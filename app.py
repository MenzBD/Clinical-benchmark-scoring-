#!/usr/bin/env python3
import json
import hashlib
import math
import os
import queue
import re
import statistics
import subprocess
import threading
import time
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from openai import OpenAI


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "gpt-5.4-nano"
SIMULATION_TEMPERATURE = float(os.environ.get("SIMULATION_TEMPERATURE", "0.7"))
EVALUATOR_TEMPERATURE = float(os.environ.get("EVALUATOR_TEMPERATURE", "0"))
IMPROVER_TEMPERATURE = float(os.environ.get("IMPROVER_TEMPERATURE", "0.3"))
# Retained for the existing UI/API response. Simulation calls use this value;
# evaluator and prompt-improver calls have their own lower temperatures.
MODEL_TEMPERATURE = SIMULATION_TEMPERATURE
SCORING_VERSION = "assigned-task-v2.1-unanchored"
TASK_METRIC_WEIGHTS = {
    "Assigned Task Fidelity": 0.50,
    "Conversational Realism": 0.20,
    "Information Behaviour": 0.20,
    "Case Fidelity": 0.10,
}
PROMPT_MIN_WORDS = 12
PROMPT_MAX_WORDS = 170
MAX_EXCHANGES = 30
MIN_EXCHANGES_BEFORE_FINAL = 10
PATIENT_TEST_MAX_EXCHANGES = 30
PATIENT_TEST_MIN_EXCHANGES = 10
THINKING_DELAY_SECONDS = 3
CLINICIAN_CHAT_STYLE = (
    "During the interaction, keep each primary-role turn conversational and short: at most two short sentences, under about 45 words. "
    "Ask about one clinical topic at a time. Do not send numbered lists, bullet lists, checklists, forms, or long questionnaires. "
    "If you need several facts, gather them over multiple turns."
)
CLINICIAN_TURN_BOUNDARY = (
    "Runtime turn boundary for this app: you are only the primary speaking role in this conversation. "
    "Return only your next message, then stop. Do not write, invent, quote, predict, or simulate the other role's response. "
    "Do not write both sides of a dialogue, a transcript, role labels, or multiple exchanges in one answer. "
    "The app will send your message to a separate counterpart simulator and will call you again after the counterpart replies. "
    "If the prompt under test asks you to simulate a whole conversation, reinterpret that as guidance for your own style only."
)
RUN_JOBS = {}
RUN_JOBS_LOCK = threading.Lock()


class ClientDisconnected(Exception):
    pass


DEFAULT_CASE_ID = "case-559"

CASE_LIBRARY = {
    "case-559": {
        "id": "case-559",
        "title": "Case 559 - Medication history and review: migraine with aura",
        "short_title": "Pharmacist-patient call",
        "patient": "Emily Carter, 29-year-old female",
        "setting": "Hospital pharmacist medication history and review interview",
        "interaction": "Patient-pharmacist medication history and review",
        "primary_role_label": "Pharmacist",
        "counterpart_role_label": "Patient",
        "primary_tune_label": "Tune Pharmacist AI",
        "counterpart_tune_label": "Tune Patient AI",
        "primary_prompt_label": "Pharmacist Prompt",
        "counterpart_prompt_label": "Patient Prompt",
        "case_summary": (
            "A pharmacist takes a medication history and reviews medication-related safety issues for a patient "
            "with worsening migraine symptoms, frequent analgesic use, allergy history and contraception safety issues."
        ),
        "chips": ["patient-pharmacist", "medication history and review", "safety", "contraception"],
        "facts": [
            "Migraine with aura diagnosed at age 23; worse over the last 5 months.",
            "9 migraine attacks and 12 headache days in the last 4 weeks.",
            "Typical aura: flickering zig-zag lights and missing visual patch for 20-25 minutes.",
            "Sensory aura: pins and needles in right hand and around lips for 10-15 minutes.",
            "Severe left temporal/retro-orbital throbbing headache with photophobia, phonophobia, nausea and occasional vomiting.",
            "No thunderclap onset, fever, neck stiffness, persistent neurological deficit, weakness, facial droop, or head injury.",
            "Uses paracetamol/codeine on 11 days in the last 4 weeks, with only partial relief.",
            "Also used plain paracetamol on 3 additional headache days.",
            "No previous triptan trial and no previous preventive migraine therapy.",
            "Allergy/adverse reaction history includes aspirin-induced wheeze, ibuprofen hives/lip swelling, naproxen facial swelling, and metoclopramide dystonic reaction.",
            "Takes combined oral contraceptive ethinylestradiol/levonorgestrel despite migraine with aura.",
            "Smokes about 4 cigarettes per day and socially vapes nicotine.",
            "Mild intermittent asthma and low-normal blood pressure, making propranolol less attractive.",
            "Not currently pregnant; would like pregnancy as an option in the next 1-2 years.",
            "Normal neurological examination and MRI brain; urine beta-hCG negative.",
            "MIDAS score 28, severe disability range.",
        ],
        "memory_facts": [
            "Headaches have been getting worse over several months and happen a lot more often than they used to.",
            "Pain is usually on one side of the head and sometimes feels like it is behind the left eye.",
            "Before some headaches, vision goes strange with bright zig-zag lights and a missing patch; sometimes there are pins and needles in the hand and around the lips.",
            "Bright light and noise bother the patient during attacks, and feeling sick or vomiting can happen.",
            "No sudden worst-ever headache, fever, stiff neck, fainting, seizure, head injury, lasting weakness, face droop or trouble speaking is remembered.",
            "The patient uses a strong painkiller on many bad headache days and sometimes another plain pain tablet, but does not know exact names or day counts from memory.",
            "The strong painkiller only partly helps.",
            "The patient has not knowingly tried a migraine-specific medicine or a daily prevention medicine before.",
            "Some anti-inflammatory pain tablets caused wheeze, hives or swelling, and one nausea medicine caused a frightening twisting/eye reaction; exact names are on an allergy list.",
            "The patient takes the pill for contraception and has a reliever inhaler for asthma; exact medicine names are on packets or a clinic list.",
            "The patient smokes a few cigarettes per day and sometimes vapes nicotine socially.",
            "Blood pressure tends to run low-normal.",
            "The patient is not trying for pregnancy immediately but wants pregnancy to remain a future option.",
            "The patient remembers being told the brain checks were reassuring but does not know exact test names or report wording from memory.",
            "The headaches are very disruptive and often require lying down in a dark quiet room, but the patient does not know a formal disability score from memory.",
        ],
        "rubric": [
            "Recognise migraine with aura and indication for preventive therapy.",
            "Identify frequent codeine/paracetamol use and medication-overuse risk.",
            "Avoid opioids as routine acute migraine therapy.",
            "Recommend migraine-specific acute therapy such as sumatriptan, with correct timing and repeat-dose counselling.",
            "Account for NSAID/aspirin allergy history and metoclopramide adverse reaction.",
            "Identify combined hormonal contraception as unsafe in migraine with aura and recommend non-oestrogen review.",
            "Recognise propranolol is a poor option because of asthma and low-normal blood pressure.",
            "Recognise topiramate is less clean because of reproductive and contraception issues.",
            "Recommend amitriptyline as a reasonable preventive option with gradual titration and review.",
            "Include headache diary, sleep, meals, hydration, exercise, stress management, and smoking cessation.",
        ],
    },
    "case-612": {
        "active": False,
        "id": "case-612",
        "title": "Case 612 - Asthma control review",
        "short_title": "Asthma control",
        "patient": "Noah Williams, 22-year-old male",
        "setting": "Community pharmacy inhaler review",
        "case_summary": (
            "Increasing shortness of breath and heavy reliever inhaler use. "
            "The pharmacist must assess control, inhaler technique, red flags and preventer treatment."
        ),
        "chips": ["respiratory", "inhalers", "safety", "adherence"],
        "facts": [
            "Asthma diagnosed in childhood; symptoms worse over the last 6 weeks after a viral illness.",
            "Uses salbutamol inhaler 6-8 times per day on bad days and wakes coughing 3 nights per week.",
            "Has budesonide/formoterol maintenance inhaler but takes it only when symptoms are bad, not every day.",
            "Poor inhaler technique: does not shake the puffer, breathes in too fast, and does not hold breath after inhaling.",
            "No spacer at home.",
            "Peak flow is around 65% of personal best today.",
            "Can speak full sentences at rest today; no blue lips, severe chest pain, confusion or silent chest.",
            "Had one emergency department visit for asthma 18 months ago but no ICU admission or intubation.",
            "Vapes nicotine most days and smokes cannabis on weekends.",
            "Lives with a cat and notices worse symptoms in dusty rooms.",
            "Uses ibuprofen sometimes for sports injuries and has noticed wheeze after it twice.",
            "No known drug allergies.",
            "Needs an asthma action plan and follow-up with GP within 1 week.",
        ],
        "memory_facts": [
            "Breathing has been worse for several weeks after a cold.",
            "The blue reliever puffer is being used a lot, especially on bad days, but exact counts are not known from memory.",
            "Coughing wakes the patient up some nights.",
            "There is another inhaler at home, but the patient is unsure how often it should be used.",
            "The patient is not confident the puffer is being used the right way.",
            "The patient does not think they have a spacer.",
            "Today the patient can talk normally while sitting down and does not feel like they are about to collapse.",
            "There was a hospital visit for asthma in the past, but details are fuzzy.",
            "The patient vapes most days and sometimes smokes cannabis.",
            "Dusty rooms and the cat seem to make breathing worse.",
            "Some pain tablets may have made wheeze worse, but the patient is not sure of the name.",
        ],
        "rubric": [
            "Recognise poor asthma control and high reliever use.",
            "Screen for acute severe asthma red flags and need for urgent referral.",
            "Identify poor adherence to preventer or maintenance therapy.",
            "Assess and correct inhaler technique and spacer use.",
            "Recommend appropriate preventer-based asthma management and GP review.",
            "Address vaping, cannabis smoke, dust and pet triggers.",
            "Consider NSAID-exacerbated respiratory symptoms.",
            "Provide an asthma action plan and safety-net advice.",
        ],
    },
    "case-704": {
        "id": "case-704",
        "title": "Case 704 - Pharmacist-doctor medication safety call",
        "short_title": "Pharmacist-doctor call",
        "patient": "Dr Arun Patel, general medicine registrar",
        "setting": "Hospital medication safety phone call",
        "interaction": "Pharmacist-doctor medication issue",
        "primary_role_label": "Pharmacist",
        "counterpart_role_label": "Doctor",
        "primary_tune_label": "Tune Pharmacist AI",
        "counterpart_tune_label": "Tune Doctor AI",
        "primary_prompt_label": "Pharmacist Prompt",
        "counterpart_prompt_label": "Doctor Prompt",
        "case_summary": (
            "A pharmacist calls a doctor about a patient with possible upper GI bleeding while taking "
            "naproxen and aspirin, requiring urgent medication safety decisions."
        ),
        "chips": ["pharmacist-doctor", "GI bleed risk", "NSAID", "handover"],
        "facts": [
            "Burning upper abdominal pain for 3 weeks, worse at night and after meals.",
            "Black sticky stools occurred twice in the last week.",
            "Feels more tired and light-headed on standing.",
            "Takes naproxen 500 mg twice daily for knee osteoarthritis for 5 weeks.",
            "Takes aspirin 100 mg daily after a transient ischaemic attack 4 years ago.",
            "Also takes ramipril, atorvastatin and sertraline.",
            "No proton pump inhibitor currently.",
            "Drinks 2 glasses of wine most nights.",
            "No vomiting blood, but nausea is present.",
            "No unexplained weight loss or swallowing difficulty.",
            "Past history of peptic ulcer 9 years ago.",
            "No known drug allergies.",
            "Needs urgent medical review because of possible upper GI bleeding.",
        ],
        "memory_facts": [
            "The doctor is covering Margaret Lewis, a 68-year-old admitted patient with upper abdominal pain.",
            "The doctor knows she has been taking naproxen for knee pain and aspirin after a previous TIA.",
            "The doctor may not have noticed the combined bleeding risk until the pharmacist raises it.",
            "The notes mention dark stools and light-headedness, but the doctor may need prompting to check exact details.",
            "There is no proton pump inhibitor on the current medication chart.",
            "The patient has a past history of peptic ulcer disease.",
            "The doctor is busy and may initially ask what the pharmacist is recommending.",
            "The doctor can agree to urgent review, stopping naproxen and arranging assessment if the concern is clearly explained.",
        ],
        "rubric": [
            "Recognise possible upper GI bleeding and need for urgent medical review.",
            "Identify NSAID plus aspirin as a high-risk medicine combination.",
            "Ask about black stools, vomiting blood, dizziness, anaemia symptoms, weight loss and swallowing difficulty.",
            "Recommend stopping NSAID pending urgent review and avoid adding OTC NSAIDs.",
            "Suggest safer pain options and non-drug osteoarthritis measures while awaiting review.",
            "Identify need for gastroprotection review if antiplatelet or NSAID therapy continues.",
            "Account for alcohol intake, past ulcer history and sertraline bleeding risk.",
            "Provide clear safety-net advice.",
        ],
    },
    "case-831": {
        "id": "case-831",
        "title": "Case 831 - Nurse-pharmacist medication advice call",
        "short_title": "Nurse-pharmacist call",
        "patient": "Jordan Lee, ward pharmacist",
        "setting": "Antenatal ward medication advice phone call",
        "interaction": "Nurse-pharmacist medication issue",
        "primary_role_label": "Nurse",
        "counterpart_role_label": "Pharmacist",
        "primary_tune_label": "Tune Nurse AI",
        "counterpart_tune_label": "Tune Pharmacist AI",
        "primary_prompt_label": "Nurse Prompt",
        "counterpart_prompt_label": "Pharmacist Prompt",
        "case_summary": (
            "A nurse calls a ward pharmacist about a pregnant patient with urinary symptoms, a penicillin "
            "allergy history and questions about safe antibiotic and symptom advice."
        ),
        "chips": ["nurse-pharmacist", "pregnancy", "antibiotics", "allergy"],
        "facts": [
            "12 weeks pregnant, confirmed by GP, first pregnancy.",
            "Burning when passing urine and going more often for 2 days.",
            "No fever, flank pain, vomiting, rigors, confusion, blood in urine or reduced fetal movement.",
            "Mild lower abdominal discomfort but no severe pelvic pain.",
            "Previous UTI 2 years ago treated successfully with nitrofurantoin.",
            "Penicillin allergy: immediate hives and lip swelling after amoxicillin as a teenager.",
            "Takes pregnancy multivitamin containing folic acid and iodine.",
            "No regular prescription medicines.",
            "No kidney disease, diabetes or immunosuppression.",
            "Urine dipstick at pharmacy positive for leukocytes and nitrites.",
            "Needs same-day GP or antenatal service review for urine culture and pregnancy-safe antibiotic.",
            "Counselling should include fluids, paracetamol if needed, avoid NSAIDs unless clinician advised, and seek urgent care if fever/flank pain/vomiting/worsening symptoms.",
        ],
        "memory_facts": [
            "The ward pharmacist is being asked about Aisha Khan, a pregnant patient with urinary symptoms.",
            "The pharmacist knows antibiotic choice needs to account for pregnancy stage and allergy history.",
            "The pharmacist may need the nurse to confirm red flags such as fever, flank pain, vomiting or systemic symptoms.",
            "The pharmacist may ask whether a urine culture has been arranged and whether prescriber review is happening.",
            "The allergy history sounds immediate, with hives and lip swelling after a penicillin-type antibiotic.",
            "The pharmacist can give general medication safety advice but should not pretend to prescribe independently if local workflow requires medical review.",
            "The pharmacist should advise escalation if red flags are present or symptoms worsen.",
            "The pharmacist can suggest pregnancy-safe symptom advice such as fluids and paracetamol if appropriate.",
        ],
        "rubric": [
            "Recognise pregnancy as a reason for medical review rather than simple OTC management.",
            "Screen for pyelonephritis and sepsis red flags: fever, flank pain, vomiting, rigors, confusion and worsening symptoms.",
            "Ask pregnancy stage and relevant obstetric symptoms.",
            "Account for penicillin allergy with immediate swelling reaction.",
            "Recommend same-day GP or antenatal review for urine culture and pregnancy-safe antibiotic.",
            "Avoid inappropriate OTC urinary alkalinisers or delayed treatment as sole management.",
            "Provide pregnancy-safe symptom advice, including fluids and paracetamol if appropriate.",
            "Safety-net urgent-care triggers clearly.",
        ],
    },
}


def case_metadata(case):
    return {
        "id": case["id"],
        "title": case["title"],
        "short_title": case.get("short_title", case["title"]),
        "patient": case["patient"],
        "setting": case["setting"],
        "interaction": case.get("interaction", ""),
        "primary_role_label": case.get("primary_role_label", "Clinician"),
        "counterpart_role_label": case.get("counterpart_role_label", "Patient"),
        "primary_tune_label": case.get("primary_tune_label", "Tune Clinician AI"),
        "counterpart_tune_label": case.get("counterpart_tune_label", "Tune Patient AI"),
        "primary_prompt_label": case.get("primary_prompt_label", "Clinician Prompt"),
        "counterpart_prompt_label": case.get("counterpart_prompt_label", "Patient Prompt"),
        "case_summary": case["case_summary"],
        "chips": case.get("chips", []),
    }


def normalise_uploaded_case(raw):
    if not isinstance(raw, dict):
        return None

    title = str(raw.get("title") or "Uploaded case").strip()
    facts = raw.get("facts") or raw.get("hidden_facts") or []
    rubric = raw.get("rubric") or []
    if not isinstance(facts, list) or not facts:
        return None
    if not isinstance(rubric, list) or not rubric:
        return None

    return {
        "id": "uploaded-case",
        "title": title,
        "short_title": str(raw.get("short_title") or title).strip(),
        "patient": str(raw.get("patient") or "Uploaded patient").strip(),
        "setting": str(raw.get("setting") or "Uploaded case setting").strip(),
        "interaction": str(raw.get("interaction") or "").strip(),
        "primary_role_label": str(raw.get("primary_role_label") or "Clinician").strip(),
        "counterpart_role_label": str(raw.get("counterpart_role_label") or "Patient").strip(),
        "primary_tune_label": str(raw.get("primary_tune_label") or "Tune Clinician AI").strip(),
        "counterpart_tune_label": str(raw.get("counterpart_tune_label") or "Tune Patient AI").strip(),
        "primary_prompt_label": str(raw.get("primary_prompt_label") or "Clinician Prompt").strip(),
        "counterpart_prompt_label": str(raw.get("counterpart_prompt_label") or "Patient Prompt").strip(),
        "case_summary": str(raw.get("case_summary") or raw.get("summary") or title).strip(),
        "chips": [str(item) for item in raw.get("chips", ["uploaded", "custom"]) if str(item).strip()],
        "facts": [str(item) for item in facts if str(item).strip()],
        "memory_facts": [
            str(item)
            for item in raw.get("memory_facts", raw.get("patient_memory_facts", facts))
            if str(item).strip()
        ],
        "rubric": [str(item) for item in rubric if str(item).strip()],
    }


def case_from_payload(payload):
    uploaded = normalise_uploaded_case(payload.get("customCase"))
    if uploaded:
        return uploaded
    return CASE_LIBRARY.get(payload.get("caseId")) or CASE_LIBRARY[DEFAULT_CASE_ID]


def case_download_payload(case):
    return {
        "id": case["id"],
        "title": case["title"],
        "patient": case["patient"],
        "setting": case["setting"],
        "interaction": case.get("interaction", ""),
        "primary_role_label": case.get("primary_role_label", "Clinician"),
        "counterpart_role_label": case.get("counterpart_role_label", "Patient"),
        "primary_tune_label": case.get("primary_tune_label", "Tune Clinician AI"),
        "counterpart_tune_label": case.get("counterpart_tune_label", "Tune Patient AI"),
        "primary_prompt_label": case.get("primary_prompt_label", "Clinician Prompt"),
        "counterpart_prompt_label": case.get("counterpart_prompt_label", "Patient Prompt"),
        "case_summary": case["case_summary"],
        "memory_facts": case.get("memory_facts", []),
        "hidden_facts": case["facts"],
        "rubric": case["rubric"],
    }


def emit(handler, payload):
    try:
        handler.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, OSError, TimeoutError):
        raise ClientDisconnected()


def read_dotenv_key():
    dotenv = ROOT / ".env"
    if not dotenv.exists():
        return None

    for line in dotenv.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "OPENAI_API_KEY":
            return value.strip().strip('"').strip("'")
    return None


def require_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = read_dotenv_key()
    if not api_key:
        try:
            result = subprocess.run(
                ["launchctl", "getenv", "OPENAI_API_KEY"],
                capture_output=True,
                text=True,
                check=False,
            )
            api_key = result.stdout.strip()
        except OSError:
            api_key = None
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not visible to this process. Export it in this terminal, "
            "save it with launchctl, or add OPENAI_API_KEY to a local .env file before restarting the prototype server."
        )
    return api_key


def chat(client, model, messages, temperature=SIMULATION_TEMPERATURE):
    request = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": 60,
    }
    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        error_text = str(exc).lower()
        temperature_unsupported = "temperature" in error_text and any(
            phrase in error_text
            for phrase in ("unsupported", "not support", "only the default", "must be 1")
        )
        if not temperature_unsupported:
            raise
        request.pop("temperature", None)
        response = client.chat.completions.create(**request)
    return response.choices[0].message.content.strip()


def emit_optional(event_callback, payload):
    if event_callback:
        event_callback(payload)


def chat_with_events(
    event_callback,
    client,
    model,
    role,
    messages,
    thinking_delay_seconds=0,
    temperature=SIMULATION_TEMPERATURE,
):
    if thinking_delay_seconds:
        emit_optional(event_callback, {"type": "typing", "role": role, "seconds": thinking_delay_seconds})
        time.sleep(thinking_delay_seconds)

    result_queue = queue.Queue(maxsize=1)

    def run_chat():
        try:
            result_queue.put(("ok", chat(client, model, messages, temperature=temperature)))
        except Exception as exc:
            result_queue.put(("error", exc))

    worker = threading.Thread(target=run_chat, daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(timeout=2)
        if worker.is_alive():
            emit_optional(event_callback, {"type": "keepalive", "role": role})

    status, result = result_queue.get()
    if status == "error":
        raise result
    return result


def chat_with_typing(handler, client, model, role, messages):
    return chat_with_events(
        lambda payload: emit(handler, payload),
        client,
        model,
        role,
        messages,
        thinking_delay_seconds=THINKING_DELAY_SECONDS,
    )


def extract_final_answer(text):
    match = re.search(r"(?ims)^\s*(?:#{1,6}\s*)?FINAL\s*:\s*", text)
    if not match:
        match = re.search(r"(?is)\bFINAL\s*:\s*", text)
    if not match:
        return None
    return text[match.end() :].strip()


def build_student_prompt(mode, custom_prompt, case):
    if mode == "custom" and custom_prompt.strip():
        return custom_prompt.strip()

    primary = primary_role_label(case)
    counterpart = counterpart_role_label(case)
    primary_lower = role_phrase(primary)
    counterpart_lower = role_phrase(counterpart)

    if mode == "poor":
        return (
            f"You are a busy {primary_lower} simulator speaking with a simulated {counterpart_lower}. "
            f"Case setting: {case['setting']}. "
            "You do not have the case facts or rubric. Keep the conversation simple and low-detail: ask or say one thing at a time, in no more than two short sentences. "
            "Do not try to cover the whole case in one message, bundle multiple safety issues together, or sound like you are working through a checklist. "
            "You may miss some follow-up and accept vague answers at first; the aim is a realistic brief conversation, not a perfect assessment in a single turn. "
            f"{CLINICIAN_CHAT_STYLE} "
            f"Speak naturally to the {counterpart_lower}. Do not preface turns with process labels, numbering, or descriptions of the question type. "
            "When you decide to finish, begin your final answer with exactly 'FINAL:'. Keep the final answer under 180 words and do not use markdown formatting."
        )

    if mode == "low":
        return (
            f"You are a basic {primary_lower} simulator speaking with a simulated {counterpart_lower}. "
            f"Case setting: {case['setting']}. "
            "You do not have the case facts or rubric. Ask relevant clinical questions or raise relevant medication issues and do some follow-up, but stay incomplete and imperfect. "
            "You may miss depth, medicine-safety details, red flags, or patient concerns, and you may stop once you think you have a reasonable outline. "
            f"{CLINICIAN_CHAT_STYLE} "
            f"Speak naturally to the {counterpart_lower}. Do not preface turns with process labels, numbering, or descriptions of the question type. "
            "When you decide to finish, begin your final answer with exactly 'FINAL:'. Keep the final answer under 200 words and do not use markdown formatting."
        )

    if mode == "high":
        return (
            f"You are an excellent {primary_lower} simulator completing a pharmacy-focused clinical interaction. "
            f"Case setting: {case['setting']}. "
            f"Speak with the {counterpart_lower} systematically, but never as a single intake form. Over several short turns, clarify the main problem, timing, severity, red flags, "
            "impact on daily life, current medicines, recently used medicines, allergies or bad reactions, "
            "medical history, pregnancy or contraception where relevant, and patient concerns. "
            f"{CLINICIAN_CHAT_STYLE} "
            "Do not give a final answer until you have enough "
            "information to address the most likely assessment, immediate safety issues, treatment options, medicine risks, "
            "counselling and follow-up. When you have enough information, begin your final "
            "answer with exactly 'FINAL:' and then provide a concise assessment and management plan. "
            f"Speak naturally to the {counterpart_lower}. Do not preface turns with process labels, numbering, or descriptions of the question type. "
            "The final answer is for the evaluator, not a patient handout: keep it under 220 words, "
            "do not add a long counselling essay, do not ask follow-up questions after finalising, "
            "and do not use markdown formatting, bold markers or headings."
        )

    if mode in {"structured", "medium"}:
        return (
            f"You are an AI {primary_lower} simulator speaking with a simulated {counterpart_lower}. Ask relevant history questions or medication-safety questions, "
            f"then provide an assessment and plan. {CLINICIAN_CHAT_STYLE} When you have enough information, begin your final answer with "
            f"exactly 'FINAL:'. Speak naturally to the {counterpart_lower}. Do not preface turns with process labels, numbering, "
            "or descriptions of the question type. Keep the final answer under 220 words and do not use markdown formatting."
        )

    return (
        f"You are an AI {primary_lower} simulator speaking with a simulated {counterpart_lower}. Ask relevant history questions or medication-safety questions, "
        f"then provide an assessment and plan. {CLINICIAN_CHAT_STYLE} When you have enough information, begin your final answer with "
        f"exactly 'FINAL:'. Speak naturally to the {counterpart_lower}. Do not preface turns with process labels, numbering, "
        "or descriptions of the question type. Keep the final answer under 220 words and do not use markdown formatting."
    )


def normalise_target_role(value):
    role = str(value or "clinician").strip().lower()
    if role in {"student", "learner", "clinician"}:
        return "clinician"
    if role == "patient":
        return "patient"
    return "clinician"


def primary_role_label(case):
    return str(case.get("primary_role_label") or "Clinician").strip() or "Clinician"


def counterpart_role_label(case):
    return str(case.get("counterpart_role_label") or "Patient").strip() or "Patient"


def counterpart_is_patient(case):
    return counterpart_role_label(case).lower() == "patient"


def role_phrase(label):
    return str(label or "role").strip().lower()


def historian_for_level(level):
    return {
        "low": "poor",
        "medium": "typical",
        "high": "clear",
        "custom": "typical",
    }.get(str(level or "medium").strip().lower(), "typical")


def clinician_mode_for_level(level):
    return {
        "low": "poor",
        "medium": "low",
        "high": "high",
        "custom": "low",
    }.get(str(level or "medium").strip().lower(), "structured")


def build_counterpart_prompt(case, historian, exact_facts_available=True, custom_patient_prompt=""):
    role = counterpart_role_label(case)
    primary = primary_role_label(case)
    role_lower = role_phrase(role)
    primary_lower = role_phrase(primary)
    has_custom_prompt = bool(custom_patient_prompt.strip())
    style = {
        "custom": (
            f"Follow the user's {role_lower} prompt exactly for behaviour, tone, cooperation level, pacing and how much information you share. "
            "The case facts below are only the factual source for the scenario; they are not a script."
        ),
        "clear": (
            f"Excellent {role_lower}: you are organised, engaged and clear. "
            "You can provide relevant details when asked and you help the interaction progress efficiently."
        ),
        "typical": (
            f"Real-world {role_lower}: you are cooperative but busy and imperfect. "
            "You answer sensible questions, may need prompting for details, and sometimes ask the other role to be specific."
        ),
        "poor": (
            f"Limited-detail {role_lower}: you are busy, brief, reactive and not very forthcoming. "
            "Keep replies short and conversational, usually no more than two sentences. "
            "You do not volunteer the whole scenario. You answer the immediate question or request, ask for clarification when needed, "
            "and may need to check the medication chart, notes, list or local workflow before giving exact details."
        ),
    }.get(historian, f"Answer naturally like a real {role_lower}.")

    source_limited = historian == "poor"
    source_guidance = ""
    if source_limited:
        source_guidance = (
            "Initial knowledge rule: before a source-check message is provided, use only the rough role knowledge facts you can recall. "
            "Do not invent, infer or volunteer exact medicine names, doses, dates, counts, investigation results or policy details from memory. "
            "If exact details are needed, say you would need to check the chart, notes, medication list, protocol or local workflow."
        )

    facts = case["facts"]
    facts_heading = "Hidden case facts"
    if source_limited and not exact_facts_available:
        facts = case.get("memory_facts") or case["facts"]
        facts_heading = f"{role} knowledge available before checking sources"

    extra_instructions = ""
    if has_custom_prompt:
        extra_instructions = f"\n\nUser {role_lower} prompt to follow exactly:\n" + custom_patient_prompt.strip()

    reveal_rules = {
        "custom": (
            f"Use the case facts to answer consistently, but let the user's {role_lower} prompt decide how much information you reveal, "
            "how quickly you reveal it, and how organised you are. "
        ),
        "clear": "You may share relevant safety details and exact information when it fits the question or call. ",
        "typical": "Do not dump the whole scenario at once. Share enough to keep the conversation realistic, then wait for follow-up. ",
        "poor": "Do not volunteer medication names, doses, exact dates, complete risk summaries or management plans unless specifically asked or asked to check a source. ",
    }.get(historian, "")

    return (
        f"You are {case['patient']} in this setting: {case['setting']}. "
        f"You are the simulated {role_lower} speaking with a {primary_lower}. "
        f"Speak in first person as the {role_lower}. Do not mention that you are an AI or that you have a hidden case file. "
        "Only use the facts provided below. Do not invent extra clinical details. "
        "Use natural spoken language appropriate to your role. Do not sound like a rubric, checklist or case note. "
        "Do not use markdown, bullet lists, bold markers or headings. "
        "Do not write both sides of the conversation or answer on behalf of the other role. "
        f"{reveal_rules}"
        f"Communication style: {style}\n\n"
        f"Case summary: {case['case_summary']}\n\n"
        f"{facts_heading}:\n- " + "\n- ".join(facts) + f"{extra_instructions}\n\n{source_guidance}"
    )


def build_patient_prompt(case, historian, exact_facts_available=True, custom_patient_prompt=""):
    if not counterpart_is_patient(case):
        return build_counterpart_prompt(case, historian, exact_facts_available, custom_patient_prompt)

    has_custom_patient_prompt = bool(custom_patient_prompt.strip())
    style = {
        "custom": (
            "Follow the user's patient prompt exactly for behaviour, attitude, tone, memory, pacing, and how forthcoming you are. "
            "The clinical facts below are only the factual source for the case; they are not a behaviour script."
        ),
        "clear": (
            "Excellent historian: you are organised, engaged and very forthcoming. "
            "You explain the main story clearly, volunteer important safety details, and can give exact medicine names, dates and counts. "
            "You still speak like a patient, but you make the interview easy because you remember the relevant information well."
        ),
        "typical": (
            "Real-world historian: you are cooperative and reasonably forthcoming, but not perfectly organised. "
            "You give about three quarters of the important information when asked sensible questions. "
            "You start with everyday words and rough estimates, forget some details, and become more precise after targeted follow-up or checking notes, packets or letters."
        ),
        "poor": (
            "Limited-detail historian: you are challenging but believable to interview. "
            "You are vague, forgetful, easily side-tracked, easily overwhelmed by multi-part questions, and not very forthcoming. "
            "Keep replies short and conversational, usually no more than two sentences. "
            "You miss or avoid most important details unless the clinician asks simple targeted follow-up questions. "
            "You do not remember exact medicine names, doses, dates, counts, test results, scores or formal diagnoses from memory. "
            "Use phrases like 'that strong painkiller', 'my pill', 'the puffer', 'a few times', 'most weeks', or 'I would need to check'. "
            "A direct broad question must not reveal the whole case. Reveal exact details only if the clinician clearly asks you to check a packet, list, diary, letter or notes."
        ),
    }.get(historian, "Answer naturally like a real patient.")

    source_locked = historian == "poor"
    poor_lock = ""
    if source_locked:
        poor_lock = (
            "Initial memory rule: before a source-check message is provided, the patient memory facts are the only facts you can recall. "
            "Do not invent, infer or volunteer exact drug names, exact day counts, exact test results, formal contraception type, formal diagnoses, "
            "named investigations, disability scores or exact pregnancy timeline. "
            "If asked a broad direct question about medicines, allergies and pregnancy, answer only from the patient memory facts. "
            "It is okay to say 'Sorry, that is a lot' and then give a partial answer. If exact names or counts are needed, offer to check packets, notes or a list. "
            "If the clinician asks again directly but "
            "does not ask you to check a source, stay approximate and offer to check a source."
        )

    facts = case["facts"]
    facts_heading = "Hidden case facts"
    if source_locked and not exact_facts_available:
        facts = case.get("memory_facts") or case["facts"]
        facts_heading = "Patient memory facts available before checking sources"
    else:
        facts = case["facts"]

    extra_patient_instructions = ""
    if has_custom_patient_prompt:
        extra_patient_instructions = (
            "\n\nUser patient prompt to follow exactly:\n" + custom_patient_prompt.strip()
        )

    fact_reveal_rules = {
        "custom": (
            "Use the case facts to answer consistently, but let the user's patient prompt decide how much information you reveal, "
            "how quickly you reveal it, and how organised you are. "
        ),
        "clear": (
            "As an excellent historian, you may volunteer relevant safety details and exact information when it fits the question. "
            "You can use names, dates and counts from the case facts, but keep the wording patient-like and plain. "
        ),
        "typical": (
            "As a real-world historian, do not volunteer the whole case at once. Give about three quarters of the important information "
            "with sensible questioning, and use rough estimates before becoming precise after follow-up or checking notes. "
        ),
        "poor": (
            "As a limited-detail historian, do not volunteer medication names, doses, exact monthly counts, formal allergy labels, investigation results, "
            "or disability scores from memory. Give approximate answers first and only provide exact details after being prompted to check a packet, list, diary or clinic letter. "
        ),
    }.get(historian, "")

    return (
        f"You are {case['patient']} in this setting: {case['setting']}. "
        "Speak in first person as the patient. Do not mention that you are an AI or that you have a hidden case file. "
        "Only use the facts provided below. Do not invent extra medical details. "
        "Use lay language only, at grade 10 reading level or below. Use short, normal sentences. "
        "Do not sound like a clinician, pharmacist, rubric or case note. "
        "Do not use markdown, bullet lists, bold markers or headings. "
        "Do not use jargon or body-location words that patients rarely use, such as temporal, retro-orbital, photophobia, "
        "phonophobia, scotoma, dyspepsia, dyspnoea, pyelonephritis, contraindication, medication overuse, dystonic reaction, "
        "or combined hormonal contraception unless the student uses the term first. Even then, answer in ordinary words. "
        "Do not open by naming the diagnosis or giving a polished summary, even if the hidden facts contain the diagnosis. Start from what you feel or what bothers you. "
        "Translate hidden facts into patient speech. Say 'this side of my head' instead of 'my temple', 'behind my eye' instead of "
        "'retro-orbital', 'bright zig-zag lights' instead of 'visual aura', 'bright light bothers me' instead of 'photophobia', "
        "'my neck twisted and my eyes rolled back' instead of 'dystonic reaction', 'burning when I pee' instead of 'dysuria', "
        "and 'short of breath' instead of 'dyspnoea'. "
        f"{fact_reveal_rules}"
        f"Historian style: {style}\n\n"
        f"Case summary: {case['case_summary']}\n\n"
        f"{facts_heading}:\n- " + "\n- ".join(facts) + f"{extra_patient_instructions}\n\n{poor_lock}"
    )


def build_patient_case_context(case, exact_facts_available=False, custom_patient_prompt=""):
    role = counterpart_role_label(case)
    role_lower = role_phrase(role)
    if not counterpart_is_patient(case):
        if exact_facts_available:
            facts = case["facts"]
            label = "Private checked source facts"
            intro = (
                f"The {role_lower} has checked the chart, notes, medication list, protocol or local workflow. "
                "These exact facts are now available as factual background. They are not a script."
            )
        else:
            facts = case.get("memory_facts") or case["facts"]
            label = f"Private rough {role_lower} knowledge"
            intro = (
                f"These are rough things the {role_lower} may know or remember without checking sources. "
                "They are factual background only, not dialogue, not a script, and not a complete scenario to recite. "
                f"The {role_lower} system prompt controls behaviour, tone, cooperation and information release."
            )

        return (
            f"{intro}\n\n"
            f"{role} identity: {case['patient']}\n"
            f"Setting: {case['setting']}\n\n"
            f"{label}:\n- " + "\n- ".join(facts)
        )

    if exact_facts_available:
        facts = case["facts"]
        label = "Private checked source facts"
        intro = (
            "The patient has checked their notes, packets, list, diary or letter. "
            "These exact facts are now available as factual background. They are not a script."
        )
    else:
        facts = case.get("memory_facts") or case["facts"]
        label = "Private rough patient memory"
        intro = (
            "These are rough things the patient may remember without checking records. "
            "They are factual background only, not dialogue, not a script, and not a complete history to recite. "
            "The patient system prompt controls behaviour, tone, cooperation and information release."
        )

    return (
        f"{intro}\n\n"
        f"Patient identity: {case['patient']}\n"
        f"Setting: {case['setting']}\n\n"
        f"{label}:\n- " + "\n- ".join(facts)
    )


def build_custom_patient_messages(case, custom_patient_prompt):
    role = counterpart_role_label(case)
    role_lower = role_phrase(role)
    prompt = custom_patient_prompt.strip() or (
        f"Act as the {role_lower} in a clinical education conversation. Use the separate case memory only for factual consistency."
    )
    runtime_boundary = (
        f"Runtime boundary for this simulation: remain the {role_lower} and return only that role's next spoken reply. "
        "The assigned system prompt is authoritative for personality, emotion, tone, cooperation, recall style, pacing and disclosure. "
        "If it assigns anger, reluctance, passivity, confusion or another difficult behaviour, show that behaviour clearly and consistently; "
        "do not become universally helpful merely because this is a healthcare conversation. "
        "Use the private case file only as factual truth. Do not invent facts, answer for the other role, or recite the whole file. "
        "Knowing a fact does not require volunteering it: reveal, withhold, approximate or defer details according to the assigned prompt. "
        "If the assigned behaviour limits recall until a packet, diary, list, chart or letter is checked, provide exact details only after the other role explicitly asks for that source check."
    )
    return [
        {"role": "system", "content": prompt},
        {"role": "system", "content": runtime_boundary},
        {"role": "user", "content": build_patient_case_context(case, exact_facts_available=True)},
    ]


def source_check_requested(text):
    lowered = text.lower()
    actions = ("check", "look at", "look up", "read", "review", "go through", "show me")
    sources = (
        "packet",
        "packets",
        "box",
        "bottle",
        "diary",
        "list",
        "letter",
        "note",
        "notes",
        "record",
        "records",
        "chart",
        "charts",
        "protocol",
        "protocols",
        "policy",
        "workflow",
        "orders",
        "symptom",
        "symptoms",
        "allergy",
        "clinic",
        "gp",
        "medication",
    )
    return any(action in lowered for action in actions) and any(source in lowered for source in sources)


def build_patient_source_unlock_prompt(case):
    if not counterpart_is_patient(case):
        role = counterpart_role_label(case)
        role_lower = role_phrase(role)
        return (
            f"The {role_lower} has now checked the relevant chart, notes, medication list, protocol or local workflow. "
            "Exact case facts are now available. Answer only the details asked for, in first person and natural spoken language. "
            f"Keep the same {role_lower} voice, not a case-note summary. "
            "Exact facts:\n- " + "\n- ".join(case["facts"])
        )

    return (
        "The patient has now checked the relevant diary or symptom notes, medication packets, allergy list, clinic letter or medication list. "
        "Exact case facts are now available. Answer only the details the clinician asked for, in first person and lay language. "
        "Use grade 10 or below English. Keep the same patient voice, not a clinical note. "
        "Use exact units carefully, for example '11 days in the last 4 weeks', not '11 weeks'. "
        "Exact facts:\n- " + "\n- ".join(case["facts"])
    )


def target_role_label(case, target_role):
    return (
        counterpart_role_label(case)
        if normalise_target_role(target_role) == "patient"
        else primary_role_label(case)
    )


def target_is_healthcare_professional(case, target_role):
    return target_role_label(case, target_role).strip().lower() not in {
        "patient",
        "caregiver",
        "family member",
        "parent",
    }


def fallback_behaviour_contract(case, target_role, prompt):
    target = target_role_label(case, target_role)
    objective = str(prompt or "").strip()
    return {
        "role": target,
        "assigned_task_summary": objective or f"Act naturally as the {role_phrase(target)} in this case.",
        "required_behaviours": [objective] if objective else [f"Remain in the {role_phrase(target)} role."],
        "prohibited_behaviours": ["Writing both sides of the conversation", "Inventing facts outside the case file"],
        "tone_and_emotion": "Follow the assigned prompt exactly; do not assume that calm or cooperative behaviour is always better.",
        "cooperation_and_disclosure": "Match the assigned prompt, including any requested anger, reluctance, passivity, uncertainty or proactive disclosure.",
        "observable_success": ["The assigned behaviour is visible repeatedly in the target role's actual turns."],
    }


def safe_behaviour_contract(raw, case, target_role, prompt):
    fallback = fallback_behaviour_contract(case, target_role, prompt)
    try:
        parsed = json.loads(extract_json_object(raw))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return fallback
    if not isinstance(parsed, dict):
        return fallback

    def string_list(key, fallback_value):
        value = parsed.get(key)
        if not isinstance(value, list):
            return fallback_value
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned[:8] or fallback_value

    return {
        "role": str(parsed.get("role") or fallback["role"]).strip(),
        "assigned_task_summary": str(parsed.get("assigned_task_summary") or fallback["assigned_task_summary"]).strip(),
        "required_behaviours": string_list("required_behaviours", fallback["required_behaviours"]),
        "prohibited_behaviours": string_list("prohibited_behaviours", fallback["prohibited_behaviours"]),
        "tone_and_emotion": str(parsed.get("tone_and_emotion") or fallback["tone_and_emotion"]).strip(),
        "cooperation_and_disclosure": str(
            parsed.get("cooperation_and_disclosure") or fallback["cooperation_and_disclosure"]
        ).strip(),
        "observable_success": string_list("observable_success", fallback["observable_success"]),
    }


def build_behaviour_contract_prompt(case, target_role, prompt):
    target = target_role_label(case, target_role)
    return [
        {
            "role": "system",
            "content": (
                "Convert an assigned role-play prompt into a concise observable evaluation contract. "
                "The assigned prompt is the authority. Do not replace difficult behaviour with generic helpfulness: if it asks for anger, anger is success; "
                "if it asks for reluctance, restricted disclosure or passivity, those are success when credibly expressed. "
                "Do not add clinical completeness, politeness or cooperation unless the prompt requests them. "
                "Return only JSON with exactly these fields: role, assigned_task_summary, required_behaviours, prohibited_behaviours, "
                "tone_and_emotion, cooperation_and_disclosure, observable_success. The three behaviour fields must be short arrays of observable items."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Case setting: {case['setting']}\n"
                f"Assigned role: {target}\n\n"
                f"ASSIGNED PROMPT:\n{str(prompt or '').strip()}"
            ),
        },
    ]


def derive_behaviour_contract(client, model, case, target_role, prompt):
    raw = chat(
        client,
        model,
        build_behaviour_contract_prompt(case, target_role, prompt),
        temperature=EVALUATOR_TEMPERATURE,
    )
    return safe_behaviour_contract(raw, case, target_role, prompt)


def evaluator_output_shape(professional):
    clinical_shape = (
        '"clinical_evaluation":{"applicable":true,"items":[{"criterion":"exact rubric criterion",'
        '"score":null,"evidence":"observable evidence"}],"safety_flags":[]}'
        if professional
        else '"clinical_evaluation":{"applicable":false,"items":[],"safety_flags":[]}'
    )
    return (
        '{"metrics":['
        '{"name":"Assigned Task Fidelity","score":null,"note":"evidence"},'
        '{"name":"Conversational Realism","score":null,"note":"evidence"},'
        '{"name":"Information Behaviour","score":null,"note":"evidence"},'
        '{"name":"Case Fidelity","score":null,"note":"evidence"}],'
        '"summary":"one sentence","critical_omissions":[],' + clinical_shape + "}"
    )


def evaluator_system_content(target, professional, paired=False):
    target_lower = role_phrase(target)
    pair_instruction = (
        "Score Transcript X and Transcript Y independently with exactly the same contract. Do not guess which is the baseline or updated prompt. "
        "After scoring, compare the turn-level evidence. A tie is allowed only when there is no meaningful observable difference in that domain. "
        "If the comparison summary says one transcript is stronger on a domain, give it a higher score on that domain unless the summary explicitly explains an offsetting weakness in the same domain. "
        if paired
        else "Score the supplied transcript against the contract. "
    )
    clinical_instruction = (
        "Because the target is a healthcare professional, also score every supplied clinical rubric item as 0, 0.5 or 1 using only observable target-role behaviour. "
        "List genuinely unsafe target-role advice as safety flags with severity critical, major or minor. "
        if professional
        else "Clinical performance is not applicable to this lay-role target; return clinical_evaluation.applicable=false. "
    )
    return (
        f"You are a strict medical-simulation evaluator. Evaluate only the observable {target_lower} turns; the other role is context. "
        + pair_instruction
        + "The assigned-task contract is authoritative. An angry role should be visibly angry; a reluctant role should be reluctant; a passive role should be passive. "
        "Never reward calmness, cooperation, disclosure, completeness or clinical usefulness unless the contract requests it. "
        "Assigned Task Fidelity (50% of the deterministic score) measures whether the requested role, emotion, tone, competence, cooperation and response pattern remain clearly visible. "
        "Conversational Realism (20%) measures believable, natural, role-appropriate speech while preserving the assigned behaviour. "
        "Information Behaviour (20%) measures whether the amount, timing and precision of disclosure or questioning match the contract; intentional withholding can earn 10/10. "
        "Case Fidelity (10%) measures factual consistency, no invented details, correct role boundaries and no writing both sides. Omission belongs here only when the contract requires disclosure. "
        "Use scores from 0 to 10 in 0.5-point increments and cite concise transcript evidence in every note. Do not return or calculate an overall score. "
        + clinical_instruction
        + "Return only valid JSON without markdown."
    )


def build_evaluator_prompt(
    case,
    transcript,
    final_answer,
    target_role="clinician",
    target_prompt="",
    evaluation_target_prompt="",
    evaluation_mode="initial",
    baseline_report=None,
    baseline_transcript="",
    behaviour_contract=None,
):
    primary = primary_role_label(case)
    target = target_role_label(case, target_role)
    professional = target_is_healthcare_professional(case, target_role)
    fixed_objective = str(evaluation_target_prompt or target_prompt or "").strip()
    contract = behaviour_contract or fallback_behaviour_contract(case, target_role, fixed_objective)
    clinical_rubric = case.get("rubric", []) if professional else []

    return [
        {
            "role": "system",
            "content": evaluator_system_content(target, professional, paired=False),
        },
        {
            "role": "user",
            "content": (
                f"Case: {case['title']}\n"
                f"Case facts for factual consistency:\n- " + "\n- ".join(case["facts"]) + "\n\n"
                f"FIXED ASSIGNED-TASK CONTRACT:\n{json.dumps(contract, ensure_ascii=False)}\n\n"
                f"Clinical rubric (separate score; empty means not applicable):\n{json.dumps(clinical_rubric, ensure_ascii=False)}\n\n"
                f"TRANSCRIPT:\n{transcript}\n\n{primary} final answer:\n{final_answer}\n\n"
                f"Return this shape exactly:\n{evaluator_output_shape(professional)}"
            ),
        },
    ]


def build_pairwise_evaluator_prompt(case, transcript_x, transcript_y, target_role, behaviour_contract):
    target = target_role_label(case, target_role)
    professional = target_is_healthcare_professional(case, target_role)
    clinical_rubric = case.get("rubric", []) if professional else []
    single_shape = evaluator_output_shape(professional)
    return [
        {
            "role": "system",
            "content": evaluator_system_content(target, professional, paired=True),
        },
        {
            "role": "user",
            "content": (
                f"Case: {case['title']}\n"
                f"Case facts for factual consistency:\n- " + "\n- ".join(case["facts"]) + "\n\n"
                f"FIXED ASSIGNED-TASK CONTRACT FOR BOTH TRANSCRIPTS:\n{json.dumps(behaviour_contract, ensure_ascii=False)}\n\n"
                f"Clinical rubric (separate score; empty means not applicable):\n{json.dumps(clinical_rubric, ensure_ascii=False)}\n\n"
                f"TRANSCRIPT X:\n{transcript_x}\n\nTRANSCRIPT Y:\n{transcript_y}\n\n"
                "Do not infer which transcript came first. Return exactly: "
                f'{{"X":{single_shape},"Y":{single_shape},"comparison_summary":"one neutral sentence"}}'
            ),
        },
    ]


def build_prompt_guidance_prompt(
    case,
    target_role,
    current_prompt,
    evaluation_target_prompt="",
    evaluation_mode="initial",
):
    primary = primary_role_label(case)
    counterpart = counterpart_role_label(case)
    target = counterpart if normalise_target_role(target_role) == "patient" else primary
    target_lower = role_phrase(target)
    fixed_objective = str(evaluation_target_prompt or current_prompt or "").strip()
    comparison = evaluation_mode == "comparison"
    if comparison:
        instruction = (
            "Compare the updated prompt with the fixed original prompt. For Fidelity, explain whether the update preserved or clarified the original behavioural intent. "
            "For Portability, explain whether the update became more or less reusable outside this app. Describe what changed; do not recommend another revision. "
        )
    else:
        instruction = (
            "For Fidelity, explain whether the current prompt clearly expresses the intended behaviour and give one practical edit. "
            "For Portability, explain whether the prompt can be copied into another AI platform together with the clinical case without hidden app logic, and give one practical edit. "
        )
    return [
        {
            "role": "system",
            "content": (
                "You are a concise medical-simulation prompt coach. Review the prompt itself separately from conversation scoring. "
                "Return only valid JSON with two qualitative pointers and no numbers, ratings or score language. "
                + instruction
                + "Appropriate role and setting details are portable; verbosity is not automatically a weakness. "
                'Return exactly: {"fidelity":"two or three concise sentences","portability":"two or three concise sentences"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Case: {case['title']}\n"
                f"Role being tuned: {target}\n\n"
                f"Fixed original behavioural objective:\n{fixed_objective}\n\n"
                f"{'Updated' if comparison else 'Current'} {target_lower} prompt:\n{current_prompt or '[No custom prompt supplied.]'}"
            ),
        },
    ]


def safe_prompt_guidance(raw):
    try:
        parsed = json.loads(extract_json_object(raw))
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    return {
        "fidelity": str(parsed.get("fidelity") or "Prompt fidelity guidance was unavailable for this run."),
        "portability": str(parsed.get("portability") or "Prompt portability guidance was unavailable for this run."),
        "raw": raw,
    }


def extract_json_object(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text


def repair_report_json(raw):
    text = extract_json_object(raw)
    text = re.sub(
        r'("name"\s*:\s*"[^"]+"\s*,)\s*"([+-]?\d+\s*%|\d+\s*/\s*\d+|n/a)"\s*,\s*"note"',
        r'\1"score":"\2","note"',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'}\s*,\s*{\s*"(critical_omissions|summary|metrics|score)"\s*:',
        r',"\1":',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def decode_json_string_fragment(fragment):
    try:
        return json.loads(f'"{fragment}"')
    except json.JSONDecodeError:
        return fragment


def extract_metric_triplets(raw, expected_names):
    text = extract_json_object(raw)
    pattern = re.compile(
        r'"name"\s*:\s*"((?:\\.|[^"\\])*)"\s*,\s*'
        r'"score"\s*:\s*"((?:\\.|[^"\\])*)"\s*,\s*'
        r'"note"\s*:\s*"((?:\\.|[^"\\])*)"',
        flags=re.DOTALL,
    )
    by_name = {}
    expected_lookup = {name.lower(): name for name in expected_names}
    for match in pattern.finditer(text):
        raw_name = decode_json_string_fragment(match.group(1)).strip()
        expected_name = expected_lookup.get(raw_name.lower())
        if not expected_name or expected_name in by_name:
            continue
        by_name[expected_name] = {
            "name": expected_name,
            "score": decode_json_string_fragment(match.group(2)),
            "note": decode_json_string_fragment(match.group(3)),
        }
    return by_name


def normalise_metric_score(score):
    if score is None:
        return "n/a"

    if isinstance(score, (int, float)) and not isinstance(score, bool):
        value = float(score)
        if 0 <= value <= 10:
            return f"{value:g} / 10"
        if 10 < value <= 100:
            return f"{value / 10:g} / 10"
        return str(score)

    text = str(score).strip()
    fraction = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)\s*$", text)
    if fraction:
        numerator = float(fraction.group(1))
        denominator = float(fraction.group(2))
        if denominator > 0:
            value = numerator if denominator == 10 else (numerator / denominator) * 10
            value = max(0, min(10, value))
            return f"{value:g} / 10"

    percent = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%\s*$", text)
    if percent:
        value = max(0, min(10, float(percent.group(1)) / 10))
        return f"{value:g} / 10"

    return text or "n/a"


def metric_score_value(score):
    normalised = normalise_metric_score(score)
    match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*/\s*10\s*$", normalised)
    if not match:
        return None
    return max(0, min(10, float(match.group(1))))


def derived_overall_score(metrics, target_role="clinician", case=None):
    values_by_name = {
        str(metric.get("name") or ""): metric_score_value(metric.get("score"))
        for metric in metrics
        if isinstance(metric, dict)
    }
    if any(values_by_name.get(name) is None for name in TASK_METRIC_WEIGHTS):
        return "--"
    return round(
        sum(values_by_name[name] * 10 * weight for name, weight in TASK_METRIC_WEIGHTS.items()),
        1,
    )


def normalise_metric(metric, fallback_name):
    if not isinstance(metric, dict):
        return {"name": fallback_name, "score": "n/a", "note": str(metric)}

    score = metric.get("score")
    note = metric.get("note") or ""

    if score is None:
        for key, value in metric.items():
            if key in {"name", "score", "note"}:
                continue
            if re.match(r"^\s*(?:[+-]?\d+\s*%|\d+\s*/\s*\d+|n/a)\s*$", str(key), re.IGNORECASE):
                score = key
                if not note and value:
                    note = value
                break

    return {
        "name": fallback_name,
        "score": normalise_metric_score(score),
        "note": str(note),
    }


def report_metric_names(target_role, case=None):
    return list(TASK_METRIC_WEIGHTS)


def normalise_clinical_evaluation(value, case, target_role):
    applicable = target_is_healthcare_professional(case or {}, target_role) if case else False
    if not applicable:
        return {
            "applicable": False,
            "score": None,
            "uncapped_score": None,
            "items": [],
            "safety_flags": [],
        }

    parsed = value if isinstance(value, dict) else {}
    raw_items = parsed.get("items") if isinstance(parsed.get("items"), list) else []
    rubric = list((case or {}).get("rubric") or [])
    items = []
    for index, criterion in enumerate(rubric):
        raw_item = raw_items[index] if index < len(raw_items) and isinstance(raw_items[index], dict) else {}
        try:
            item_score = float(raw_item.get("score", 0))
        except (TypeError, ValueError):
            item_score = 0.0
        item_score = min(1.0, max(0.0, item_score))
        items.append(
            {
                "criterion": str(criterion),
                "score": item_score,
                "evidence": str(raw_item.get("evidence") or "No supporting target-role evidence identified."),
            }
        )

    flags = []
    raw_flags = parsed.get("safety_flags") if isinstance(parsed.get("safety_flags"), list) else []
    for raw_flag in raw_flags[:8]:
        if isinstance(raw_flag, dict):
            severity = str(raw_flag.get("severity") or "major").strip().lower()
            note = str(raw_flag.get("note") or "").strip()
        else:
            severity = "major"
            note = str(raw_flag).strip()
        if severity not in {"critical", "major", "minor"}:
            severity = "major"
        if note:
            flags.append({"severity": severity, "note": note})

    uncapped_score = round((sum(item["score"] for item in items) / len(items)) * 100, 1) if items else 0.0
    score = min(49.0, uncapped_score) if any(flag["severity"] == "critical" for flag in flags) else uncapped_score
    return {
        "applicable": True,
        "score": score,
        "uncapped_score": uncapped_score,
        "items": items,
        "safety_flags": flags,
    }


def normalise_report(parsed, raw, target_role="clinician", case=None):
    expected_names = report_metric_names(target_role, case)
    if not isinstance(parsed, dict):
        raise ValueError("Evaluator report was not a JSON object.")

    metrics = parsed.get("metrics") if isinstance(parsed.get("metrics"), list) else []
    extracted_metrics = extract_metric_triplets(raw, expected_names)
    normalised_metrics = []
    for index, expected_name in enumerate(expected_names):
        metric = extracted_metrics.get(expected_name) or (metrics[index] if index < len(metrics) else {})
        normalised_metrics.append(normalise_metric(metric, expected_name))

    score = derived_overall_score(normalised_metrics, target_role, case)

    omissions = parsed.get("critical_omissions")
    if not isinstance(omissions, list):
        omissions = []

    return {
        "score": score,
        "metrics": normalised_metrics,
        "summary": str(parsed.get("summary") or "Evaluation complete."),
        "critical_omissions": [str(item) for item in omissions],
        "clinical_evaluation": normalise_clinical_evaluation(
            parsed.get("clinical_evaluation"), case or {}, target_role
        ),
        "scoring_version": SCORING_VERSION,
        "raw": raw,
    }


def safe_report(raw, target_role="clinician", case=None):
    repaired = repair_report_json(raw)
    try:
        return normalise_report(json.loads(repaired), raw, target_role, case)
    except (json.JSONDecodeError, ValueError):
        expected_names = report_metric_names(target_role, case)
        return {
            "score": None,
            "metrics": [
                {
                    "name": name,
                    "score": "n/a",
                    "note": "Evaluator returned malformed JSON; this run is excluded from numerical aggregates.",
                }
                for name in expected_names
            ],
            "summary": "Evaluator returned a malformed report. Scorecard fields were not fully available.",
            "critical_omissions": [],
            "clinical_evaluation": normalise_clinical_evaluation({}, case or {}, target_role),
            "scoring_version": SCORING_VERSION,
            "invalid_evaluation": True,
            "raw": raw,
        }


def safe_pairwise_reports(raw, target_role, case):
    try:
        parsed = json.loads(extract_json_object(raw))
        if not isinstance(parsed, dict) or not isinstance(parsed.get("X"), dict) or not isinstance(parsed.get("Y"), dict):
            raise ValueError("Pairwise evaluator did not return X and Y reports.")
        report_x_raw = json.dumps(parsed["X"], ensure_ascii=False)
        report_y_raw = json.dumps(parsed["Y"], ensure_ascii=False)
        report_x = normalise_report(parsed["X"], report_x_raw, target_role, case)
        report_y = normalise_report(parsed["Y"], report_y_raw, target_role, case)
        comparison_summary = str(parsed.get("comparison_summary") or "Blind paired evaluation complete.")
        return report_x, report_y, comparison_summary
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, None, "Blind paired evaluator returned malformed JSON."


def build_prompt_improvement_prompt(case, target, current_prompt, report, transcript, notes):
    target = normalise_target_role(target)
    primary = primary_role_label(case)
    counterpart = counterpart_role_label(case)
    tuned_role = counterpart if target == "patient" else primary
    comparison_role = primary if target == "patient" else counterpart
    tuned_lower = role_phrase(tuned_role)
    comparison_lower = role_phrase(comparison_role)
    report_metrics = [
        {
            "name": metric.get("name"),
            "score": metric.get("score"),
            "note": metric.get("note"),
        }
        for metric in (report.get("metrics", []) if isinstance(report, dict) else [])[:4]
        if isinstance(metric, dict)
    ]
    if target == "patient":
        # Behavioural feedback is sufficient to revise a simulated-patient prompt. Excluding the
        # Case Fidelity note and raw dialogue prevents private scenario facts from being copied into it.
        report_metrics = [metric for metric in report_metrics if metric.get("name") != "Case Fidelity"]
    report_summary = {
        "score": report.get("score") if isinstance(report, dict) else None,
        "metrics": report_metrics,
        "summary": "" if target == "patient" else (report.get("summary") if isinstance(report, dict) else ""),
        "critical_omissions": (
            report.get("critical_omissions", [])[:3]
            if target != "patient" and isinstance(report, dict) and isinstance(report.get("critical_omissions"), list)
            else []
        ),
        "prompt_guidance": (
            {
                "fidelity": report.get("prompt_guidance", {}).get("fidelity", ""),
                "portability": report.get("prompt_guidance", {}).get("portability", ""),
            }
            if isinstance(report, dict) and isinstance(report.get("prompt_guidance"), dict)
            else {}
        ),
        "behaviour_contract": (
            report.get("behaviour_contract", {})
            if isinstance(report, dict) and isinstance(report.get("behaviour_contract"), dict)
            else {}
        ),
    }
    transcript_excerpt = (
        "[Raw dialogue omitted for patient-prompt editing to prevent private case-fact leakage.]"
        if target == "patient"
        else (transcript[-30:] if isinstance(transcript, list) else transcript)
    )
    system = (
        f"You are a restrained medical-simulation prompt editor. Improve a {tuned_lower} role-play prompt using the evaluator feedback, "
        "but make a modest revision rather than a comprehensive rewrite. Preserve the original role, personality, competence, difficulty, cooperation level "
        "and information-sharing direction. A lazy role must remain lazy, a challenging role challenging, a supportive role supportive, "
        "and a baseline role recognisably in the middle. Select no more than two feedback findings with the greatest likely effect on the next conversation. "
        "The fixed behaviour contract in the feedback defines success. If the assigned role is angry, reluctant, passive, confused, terse or otherwise difficult, "
        "make that assigned behaviour more observable rather than making the role generically calm, cooperative or clinically useful. "
        "Add only clear, observable behaviour needed to address those findings and preserve behaviours that already worked. "
        "Keep the role and scenario identification from the original prompt. The revised prompt must be portable to another chatbot with the same case. "
        "Use the smallest effective revision: one compact paragraph of 2-6 sentences, normally 35-130 words and never more than 170 words. "
        "Do not use headings, bullet points, numbered rules, optional modes, scoring language, app-specific instructions, hidden case facts, diagnoses, "
        "worked examples, exhaustive checklists or both sides of the conversation. Do not solve every weakness mentioned in the report. "
        "Return only valid JSON with this shape: "
        '{"prompt":"concise improved prompt","rationale":"one sentence naming the one or two changes selected"}'
    )

    return [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": (
                f"Simulation: {case['interaction']}\n"
                f"Setting: {case['setting']}\n"
                f"Role being tuned: {tuned_role}\n"
                f"Conversation partner: {comparison_role}\n\n"
                f"Original {tuned_lower} prompt:\n{current_prompt}\n\n"
                f"Evaluator feedback:\n{json.dumps(report_summary, ensure_ascii=False)}\n\n"
                f"Requested focus:\n{notes or '[No single domain selected.]'}\n\n"
                f"Short transcript excerpt:\n{json.dumps(transcript_excerpt, ensure_ascii=False)}\n\n"
                f"Revise only the {tuned_lower} prompt. Prioritise the one or two changes most likely to improve the observed behaviour with the {comparison_lower}."
            ),
        },
    ]


PROMPT_LEAKAGE_GENERIC_TERMS = {
    "additional", "answer", "answers", "angry", "asked", "avoid", "because", "behaviour", "brief", "candidate", "case",
    "chatbot", "clearly", "consistent", "conversation", "cooperative", "daily", "details",
    "difficult", "disclose", "exact", "factual", "factually", "guarded", "hidden", "history",
    "impatient", "information", "medicine", "medication", "patient", "pharmacist", "polite",
    "prompt", "question", "questions", "reluctant", "remain", "repeated", "resisting", "review",
    "right", "scenario", "share", "short", "specifics", "stated", "talking", "their", "timing", "under", "without",
}


def case_specific_prompt_terms(case, original_prompt, candidate_prompt):
    """Return new terms copied from private case material into a revised prompt."""
    if not isinstance(case, dict):
        return []
    private_sections = [case.get("case_summary", "")]
    for key in ("facts", "memory_facts", "rubric"):
        values = case.get(key) or []
        if isinstance(values, list):
            private_sections.extend(str(value) for value in values)

    def terms(text):
        return {
            token
            for token in re.findall(r"[a-z][a-z0-9']{4,}", str(text or "").lower())
            if token not in PROMPT_LEAKAGE_GENERIC_TERMS
        }

    private_terms = terms(" ".join(private_sections))
    added_terms = terms(candidate_prompt) - terms(original_prompt)
    return sorted(private_terms & added_terms)


def prompt_validation_issues(original_prompt, candidate_prompt, case=None):
    original = str(original_prompt or "").strip()
    candidate = str(candidate_prompt or "").strip()
    issues = []
    words = re.findall(r"\b\S+\b", candidate)
    if len(words) < PROMPT_MIN_WORDS:
        issues.append(f"Prompt is too short ({len(words)} words; minimum {PROMPT_MIN_WORDS}).")
    if len(words) > PROMPT_MAX_WORDS:
        issues.append(f"Prompt is too long ({len(words)} words; maximum {PROMPT_MAX_WORDS}).")
    if re.search(r"(?m)^\s*(?:[-*•]|\d+[.)]|#{1,6}\s)", candidate):
        issues.append("Prompt uses headings, bullets or numbered rules instead of one portable paragraph.")
    if re.search(r"[\"“][^\"”\n]{2,}[\"”]", candidate):
        issues.append("Prompt embeds worked dialogue examples instead of portable behavioural instructions.")
    if "you are" in original.lower() and "you are" not in candidate.lower():
        issues.append("Prompt no longer explicitly preserves the assigned role introduction.")

    original_numbers = set(re.findall(r"\b\d{2,}(?:\.\d+)?\b", original))
    added_numbers = set(re.findall(r"\b\d{2,}(?:\.\d+)?\b", candidate)) - original_numbers
    if added_numbers:
        issues.append("Prompt introduces new case-like exact numbers: " + ", ".join(sorted(added_numbers)))
    leaked_terms = case_specific_prompt_terms(case, original, candidate)
    if leaked_terms:
        issues.append(
            "Prompt copies private case-specific content not present in the original: "
            + ", ".join(leaked_terms[:12])
        )
    return issues


def parse_improved_prompt(raw):
    try:
        parsed = json.loads(extract_json_object(raw))
    except json.JSONDecodeError:
        parsed = {
            "prompt": raw.strip(),
            "rationale": "The model returned a prompt in plain text rather than JSON.",
        }
    return {
        "prompt": str(parsed.get("prompt", "")).strip(),
        "rationale": str(parsed.get("rationale", "")).strip(),
    }


def improve_prompt(payload):
    api_key = require_api_key()
    client = OpenAI(api_key=api_key)
    case = case_from_payload(payload)
    model = payload.get("model") or DEFAULT_MODEL
    target = normalise_target_role(payload.get("promptTarget", "clinician"))
    current_prompt = payload.get("currentPrompt", "")
    report = payload.get("report") or {}
    transcript = payload.get("transcript") or []
    notes = payload.get("notes", "")

    messages = build_prompt_improvement_prompt(case, target, current_prompt, report, transcript, notes)
    raw = chat(client, model, messages, temperature=IMPROVER_TEMPERATURE)
    parsed = parse_improved_prompt(raw)
    prompt = parsed["prompt"]
    if not prompt:
        raise RuntimeError("Prompt improvement returned an empty prompt.")

    issues = prompt_validation_issues(current_prompt, prompt, case=case)
    repair_messages = list(messages)
    repair_attempts = 0
    while issues and repair_attempts < 3:
        repair_attempts += 1
        repair_messages.extend(
            [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "The candidate failed validation:\n- "
                        + "\n- ".join(issues)
                        + "\nReturn corrected JSON that passes every limit. Use only abstract behavioural instructions: "
                        "no quotation marks, sample utterances, diagnoses, medicines, symptoms, numbers, or other case facts. "
                        "Preserve the fixed assigned role and behaviour."
                    ),
                },
            ]
        )
        raw = chat(client, model, repair_messages, temperature=IMPROVER_TEMPERATURE)
        repaired = parse_improved_prompt(raw)
        if repaired["prompt"]:
            prompt = repaired["prompt"]
            parsed = repaired
        issues = prompt_validation_issues(current_prompt, prompt, case=case)
    if issues:
        raise RuntimeError("Prompt improvement failed validation after repair: " + "; ".join(issues))

    return {
        "prompt": prompt,
        "rationale": parsed["rationale"],
        "validation": {
            "passed": not issues,
            "issues": issues,
            "word_count": len(re.findall(r"\b\S+\b", prompt)),
            "repair_attempts": repair_attempts,
        },
    }


def score_to_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group(0)) if match else None


def transcript_payload_to_text(case, transcript):
    if isinstance(transcript, str):
        return transcript.strip()
    if not isinstance(transcript, list):
        return ""

    primary = primary_role_label(case)
    counterpart = counterpart_role_label(case)
    role_labels = {
        "student": primary,
        "clinician": primary,
        "patient": counterpart,
    }
    lines = []
    for item in transcript:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").lower()
        if role not in role_labels:
            continue
        label = role_labels[role]
        if item.get("final"):
            label = f"{label} final"
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(f"{label}: {content}")
    return "\n\n".join(lines)


def add_comparison_metadata(report, baseline_report):
    if not isinstance(baseline_report, dict):
        return report

    baseline_metrics = {
        str(metric.get("name") or ""): metric
        for metric in baseline_report.get("metrics", [])
        if isinstance(metric, dict)
    }
    for metric in report.get("metrics", []):
        baseline_metric = baseline_metrics.get(str(metric.get("name") or ""))
        if not baseline_metric:
            continue
        baseline_value = metric_score_value(baseline_metric.get("score"))
        current_value = metric_score_value(metric.get("score"))
        metric["baseline_score"] = normalise_metric_score(baseline_metric.get("score"))
        metric["delta"] = (
            round(current_value - baseline_value, 1)
            if baseline_value is not None and current_value is not None
            else None
        )

    baseline_score = score_to_number(baseline_report.get("score"))
    current_score = score_to_number(report.get("score"))
    report["comparison"] = {
        "baseline_score": baseline_score,
        "updated_score": current_score,
        "delta": (
            round(current_score - baseline_score, 1)
            if baseline_score is not None and current_score is not None
            else None
        ),
    }
    current_clinical = report.get("clinical_evaluation")
    baseline_clinical = baseline_report.get("clinical_evaluation")
    if isinstance(current_clinical, dict) and isinstance(baseline_clinical, dict):
        current_clinical_score = score_to_number(current_clinical.get("score"))
        baseline_clinical_score = score_to_number(baseline_clinical.get("score"))
        current_clinical["baseline_score"] = baseline_clinical_score
        current_clinical["delta"] = (
            round(current_clinical_score - baseline_clinical_score, 1)
            if current_clinical_score is not None and baseline_clinical_score is not None
            else None
        )
    report["evaluation_mode"] = "comparison"
    return report


def build_run_material(payload):
    case = case_from_payload(payload)
    target_role = normalise_target_role(payload.get("targetRole", "clinician"))
    prompt_level = str(payload.get("promptLevel", "low")).strip().lower()
    target_prompt = payload.get("targetPrompt", "")
    evaluation_target_prompt = payload.get("evaluationTargetPrompt") or target_prompt
    baseline_report = payload.get("baselineReport")
    baseline_transcript = payload.get("baselineTranscript")
    behaviour_contract = payload.get("behaviourContract")
    if not isinstance(behaviour_contract, dict) and isinstance(baseline_report, dict):
        behaviour_contract = baseline_report.get("behaviour_contract")
    evaluation_mode = (
        "comparison"
        if payload.get("evaluationMode") == "comparison" and isinstance(baseline_report, dict)
        else "initial"
    )

    model = payload.get("clinicianModel") or payload.get("studentModel") or DEFAULT_MODEL
    patient_model = payload.get("patientModel") or model
    evaluator_model = payload.get("evaluatorModel") or model
    prompt_mode = payload.get("promptMode", "structured")
    historian = payload.get("historian", "poor")
    custom_prompt = payload.get("customPrompt", "")
    patient_prompt = payload.get("patientPrompt", "")

    clinician_prompt_under_test = target_role == "clinician" and bool(str(target_prompt).strip())
    patient_prompt_under_test = target_role == "patient" and bool(str(target_prompt).strip())
    max_exchanges = PATIENT_TEST_MAX_EXCHANGES if target_role == "patient" else MAX_EXCHANGES
    min_exchanges = PATIENT_TEST_MIN_EXCHANGES if target_role == "patient" else MIN_EXCHANGES_BEFORE_FINAL
    primary = primary_role_label(case)
    counterpart = counterpart_role_label(case)
    primary_lower = role_phrase(primary)
    counterpart_lower = role_phrase(counterpart)

    if "targetRole" in payload:
        if target_role == "patient":
            prompt_mode = clinician_mode_for_level(prompt_level)
            custom_prompt = ""
            historian = "custom" if patient_prompt_under_test else "typical"
            patient_prompt = target_prompt
        else:
            prompt_mode = "custom" if clinician_prompt_under_test else "structured"
            custom_prompt = target_prompt
            historian = historian_for_level(prompt_level)
            patient_prompt = ""

    student_system = build_student_prompt(prompt_mode, custom_prompt, case)

    if clinician_prompt_under_test:
        starter_content = (
            f"Begin the simulated {primary_lower}-{counterpart_lower} conversation now in natural clinical language. "
            f"Case setting: {case['setting']}. "
            f"Run this as a turn-by-turn conversation with up to {max_exchanges} {primary_lower}-{counterpart_lower} exchanges. "
            f"Do not finalise before at least {min_exchanges} exchanges. "
            "Use short conversational turns. Do not send numbered lists, bullet lists, intake forms, checklists, or long questionnaires. "
            f"{CLINICIAN_TURN_BOUNDARY} "
            "Do not preface turns with process labels, numbering, or descriptions of the question type. "
            "When you are ready to finish after that point, begin your final answer with exactly 'FINAL:' so the evaluator can score it."
        )
    elif patient_prompt_under_test:
        starter_content = (
            f"Begin the simulated {primary_lower}-{counterpart_lower} conversation now in natural clinical language. "
            f"Case setting: {case['setting']}. "
            f"Start with one short opening question or call-opening. Run this as a turn-by-turn conversation with up to {max_exchanges} {primary_lower}-{counterpart_lower} exchanges. "
            f"Do not finalise before at least {min_exchanges} exchanges. "
            "Ask no more than one or two closely related details in a single turn. "
            "Do not send numbered lists, bullet lists, intake forms, checklists, or long questionnaires. "
            "Do not preface turns with process labels, numbering, or descriptions of the question type. "
            "When you are ready to finish after that point, begin your final answer with exactly 'FINAL:'."
        )
    else:
        source_expectation = (
            "Expect that a poor historian may not know exact medication names, doses or counts unless prompted to check a diary, packet or list. "
            if counterpart_is_patient(case)
            else f"Expect that a difficult {counterpart_lower} may not volunteer exact details unless asked clearly or asked to check the chart, notes or medication list. "
        )
        starter_content = (
            f"You are about to speak with a simulated {counterpart_lower}. Communicate naturally, usually one clinical topic or medication issue at a time. "
            f"Case setting: {case['setting']}. "
            f"Do not provide the final answer yet. You have up to {max_exchanges} {primary_lower}-{counterpart_lower} exchanges. "
            f"Do not finalise before at least {min_exchanges} exchanges. "
            "Before finalising, cover the main symptom pattern, timing, red flags, impact, current and recent medicines, "
            "allergies/adverse reactions, relevant medical history, pregnancy or contraception if relevant, and follow-up needs. "
            f"{source_expectation}"
            f"If the {counterpart_lower} is vague, guide them with simple follow-up questions and ask them to check available records rather than assuming details. "
            "Use short conversational turns. Do not send numbered lists, bullet lists, intake forms, checklists, or long questionnaires. "
            "Do not preface turns with process labels, numbering, or descriptions of the question type. Start the conversation now."
        )

    student_messages = [
        {"role": "system", "content": student_system},
        *(
            [{"role": "system", "content": CLINICIAN_TURN_BOUNDARY}]
            if clinician_prompt_under_test
            else []
        ),
        {
            "role": "user",
            "content": starter_content,
        },
    ]
    exact_patient_sources_unlocked = not patient_prompt_under_test and historian != "poor"
    if patient_prompt_under_test:
        patient_messages = build_custom_patient_messages(case, patient_prompt)
    else:
        patient_messages = [
            {
                "role": "system",
                "content": build_patient_prompt(
                    case,
                    historian,
                    exact_facts_available=exact_patient_sources_unlocked,
                    custom_patient_prompt=patient_prompt,
                ),
            }
        ]

    return {
        "case": case,
        "target_role": target_role,
        "prompt_level": prompt_level,
        "target_prompt": target_prompt,
        "evaluation_target_prompt": evaluation_target_prompt,
        "evaluation_mode": evaluation_mode,
        "baseline_report": baseline_report if isinstance(baseline_report, dict) else None,
        "baseline_transcript": transcript_payload_to_text(case, baseline_transcript),
        "behaviour_contract": behaviour_contract if isinstance(behaviour_contract, dict) else None,
        "model": model,
        "patient_model": patient_model,
        "evaluator_model": evaluator_model,
        "prompt_mode": prompt_mode,
        "historian": historian,
        "custom_prompt": custom_prompt,
        "patient_prompt": patient_prompt,
        "clinician_prompt_under_test": clinician_prompt_under_test,
        "patient_prompt_under_test": patient_prompt_under_test,
        "max_exchanges": max_exchanges,
        "min_exchanges": min_exchanges,
        "student_messages": student_messages,
        "patient_messages": patient_messages,
        "exact_patient_sources_unlocked": exact_patient_sources_unlocked,
    }


def format_messages_for_preview(messages):
    return [{"role": message.get("role", ""), "content": message.get("content", "")} for message in messages]


def build_prompt_preview(payload, transcript_text=None, final_answer=None):
    material = build_run_material(payload)
    case = material["case"]
    primary = primary_role_label(case)
    counterpart = counterpart_role_label(case)
    counterpart_lower = role_phrase(counterpart)
    evaluator_transcript = transcript_text or "[Transcript will be inserted after the run.]"
    evaluator_final_answer = final_answer or f"[{primary} final answer will be inserted after the run.]"
    evaluator_target_prompt = (
        material["patient_prompt"]
        if material["patient_prompt_under_test"]
        else material["custom_prompt"]
    )
    evaluator_objective_prompt = material["evaluation_target_prompt"]
    preview_contract = material["behaviour_contract"] or fallback_behaviour_contract(
        case, material["target_role"], evaluator_objective_prompt
    )
    if material["evaluation_mode"] == "comparison":
        evaluator_messages = build_pairwise_evaluator_prompt(
            case,
            material["baseline_transcript"] or "[Baseline transcript will be inserted.]",
            evaluator_transcript,
            material["target_role"],
            preview_contract,
        )
    else:
        evaluator_messages = build_evaluator_prompt(
            case,
            evaluator_transcript,
            evaluator_final_answer,
            material["target_role"],
            evaluator_target_prompt,
            evaluator_objective_prompt,
            material["evaluation_mode"],
            material["baseline_report"],
            material["baseline_transcript"],
            preview_contract,
        )
    prompt_guidance_messages = build_prompt_guidance_prompt(
        case,
        material["target_role"],
        evaluator_target_prompt,
        evaluator_objective_prompt,
        material["evaluation_mode"],
    )
    if counterpart_is_patient(case):
        prompt_note = "Patient prompts under test control behaviour and disclosure; a separate private case file supplies factual truth without requiring it to be volunteered."
    else:
        prompt_note = f"{counterpart} prompts under test control behaviour and disclosure; a separate private case file supplies factual truth without requiring it to be volunteered."
    notes = [
        prompt_note,
        f"Run cap: {material['max_exchanges']} {role_phrase(primary)}-{counterpart_lower} exchanges.",
    ]
    if material["evaluation_mode"] == "comparison":
        notes.append("Comparison run: transcripts are rescored together under anonymous X/Y labels; the earlier scorecard is hidden from the final judge.")
    if material["historian"] == "poor" and not material["patient_prompt_under_test"]:
        if counterpart_is_patient(case):
            notes.append("Default poor-historian runs may add checked source facts only after the primary role asks the patient to check records.")
        else:
            notes.append(f"Default difficult-{counterpart_lower} runs may add checked source facts only after the primary role asks the {counterpart_lower} to check records.")

    return {
        "caseId": case["id"],
        "caseTitle": case["title"],
        "targetRole": material["target_role"],
        "maxExchanges": material["max_exchanges"],
        "clinician": format_messages_for_preview(material["student_messages"]),
        "patient": format_messages_for_preview(material["patient_messages"]),
        "evaluator": format_messages_for_preview(
            evaluator_messages
            + [
                {
                    "role": "system",
                    "content": "Separate qualitative prompt-guidance pass (not included in the numerical conversation score).",
                }
            ]
            + prompt_guidance_messages
        ),
        "notes": notes,
    }


def run_case(payload, event_callback=None, thinking_delay_seconds=0):
    api_key = require_api_key()
    client = OpenAI(api_key=api_key)
    material = build_run_material(payload)
    case = material["case"]
    target_role = material["target_role"]
    model = material["model"]
    patient_model = material["patient_model"]
    evaluator_model = material["evaluator_model"]
    historian = material["historian"]
    custom_prompt = material["custom_prompt"]
    patient_prompt = material["patient_prompt"]
    evaluation_target_prompt = material["evaluation_target_prompt"]
    evaluation_mode = material["evaluation_mode"]
    baseline_report = material["baseline_report"]
    baseline_transcript = material["baseline_transcript"]
    clinician_prompt_under_test = material["clinician_prompt_under_test"]
    patient_prompt_under_test = material["patient_prompt_under_test"]
    max_exchanges = material["max_exchanges"]
    min_exchanges = material["min_exchanges"]
    student_messages = [dict(message) for message in material["student_messages"]]
    patient_messages = [dict(message) for message in material["patient_messages"]]
    exact_patient_sources_unlocked = material["exact_patient_sources_unlocked"]
    behaviour_contract = material["behaviour_contract"]
    transcript = []
    final_answer = None
    primary = primary_role_label(case)
    counterpart = counterpart_role_label(case)
    primary_lower = role_phrase(primary)
    counterpart_lower = role_phrase(counterpart)

    transcript_events = []

    if not behaviour_contract:
        emit_optional(event_callback, {"type": "status", "maxTurns": max_exchanges, "phase": "contract"})
        behaviour_contract = derive_behaviour_contract(
            client,
            evaluator_model,
            case,
            target_role,
            evaluation_target_prompt,
        )

    for turn in range(1, max_exchanges + 1):
        emit_optional(event_callback, {"type": "status", "turn": turn, "maxTurns": max_exchanges, "phase": "student"})
        student_reply = chat_with_events(event_callback, client, model, "student", student_messages, thinking_delay_seconds)

        detected_final = extract_final_answer(student_reply)

        if detected_final and turn < min_exchanges:
            emit_optional(
                event_callback,
                {
                    "type": "message",
                    "role": "evaluator",
                    "content": (
                        f"{primary} attempted to finalise at exchange {turn}. Continuing interaction until at least "
                        f"{min_exchanges} exchanges or the hard {max_exchanges}-exchange cap."
                    ),
                },
            )
            student_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Do not finalise yet. Continue the interview naturally with one short follow-up about a single clinical topic. "
                        "Do not ask a multi-part questionnaire, numbered list, bullet list, checklist, or intake form. "
                        "Do not preface turns with process labels, numbering, or descriptions of the question type. "
                        "You still need to gather enough information across symptoms, timing, red flags, impact, "
                        "medicine use, allergies/adverse reactions, relevant medical history and follow-up needs."
                    ),
                }
            )
            continue

        if detected_final:
            final_answer = detected_final
            transcript.append((f"{primary} final", final_answer))
            transcript_events.append({"role": "clinician", "content": final_answer, "final": True})
            emit_optional(event_callback, {"type": "message", "role": "student", "content": final_answer, "final": True})
            break

        transcript.append((primary, student_reply))
        transcript_events.append({"role": "clinician", "content": student_reply, "final": False})
        emit_optional(event_callback, {"type": "message", "role": "student", "content": student_reply})

        patient_messages.append({"role": "user", "content": student_reply})
        if historian == "poor" and not exact_patient_sources_unlocked and source_check_requested(student_reply):
            patient_messages.append({"role": "system", "content": build_patient_source_unlock_prompt(case)})
            exact_patient_sources_unlocked = True
        emit_optional(event_callback, {"type": "status", "turn": turn, "maxTurns": max_exchanges, "phase": "patient"})
        answer = chat_with_events(event_callback, client, patient_model, "patient", patient_messages, thinking_delay_seconds)
        patient_messages.append({"role": "assistant", "content": answer})
        transcript.append((counterpart, answer))
        transcript_events.append({"role": "patient", "content": answer, "final": False})
        emit_optional(event_callback, {"type": "message", "role": "patient", "content": answer})

        student_messages.append({"role": "assistant", "content": student_reply})
        if clinician_prompt_under_test:
            next_instruction = (
                f"{counterpart} answer: {answer}\n\n"
                f"Continue according to your system prompt. {CLINICIAN_TURN_BOUNDARY} "
                "If you are ready to finish, begin with exactly 'FINAL:'."
            )
        else:
            next_instruction = (
                f"{counterpart} answer: {answer}\n\n"
                "Continue the interview naturally with one short follow-up about a single clinical topic, unless you now have enough information to safely answer the case. "
                "Do not ask a multi-part questionnaire, numbered list, bullet list, checklist, or intake form. "
                "If you need several details, spread them over later turns. "
                "Do not preface turns with process labels, numbering, or descriptions of the question type. "
                "If you are ready to finish, begin with exactly 'FINAL:' and include the most likely assessment, immediate safety issues, "
                "medicine recommendations or referral needs, counselling and follow-up. Keep the final answer under 220 words "
                "and do not ask further questions after finalising. Do not use markdown formatting."
            )
        if turn == max_exchanges:
            next_instruction = (
                f"{counterpart} answer: {answer}\n\n"
                f"You have reached the hard {max_exchanges}-exchange limit. Begin with exactly 'FINAL:' and provide your final assessment "
                "and management plan now, including immediate safety issues, medicine recommendations or referral needs, "
                "counselling and follow-up. Keep it under 220 words and do not use markdown formatting."
            )
        student_messages.append({"role": "user", "content": next_instruction})

    if final_answer is None:
        emit_optional(event_callback, {"type": "status", "maxTurns": max_exchanges, "phase": "final-answer"})
        final_answer = chat_with_events(event_callback, client, model, "student", student_messages, thinking_delay_seconds)
        detected_final = extract_final_answer(final_answer)
        if detected_final:
            final_answer = detected_final
        transcript.append((f"{primary} final", final_answer))
        transcript_events.append({"role": "clinician", "content": final_answer, "final": True})
        emit_optional(event_callback, {"type": "message", "role": "student", "content": final_answer, "final": True})

    transcript_text = "\n\n".join(f"{speaker}: {content}" for speaker, content in transcript)
    emit_optional(event_callback, {"type": "status", "maxTurns": max_exchanges, "phase": "evaluator"})
    if evaluation_mode == "comparison":
        order_digest = hashlib.sha256(
            (baseline_transcript + "\n---PAIR---\n" + transcript_text).encode("utf-8")
        ).digest()[0]
        current_is_x = bool(order_digest % 2)
        transcript_x = transcript_text if current_is_x else baseline_transcript
        transcript_y = baseline_transcript if current_is_x else transcript_text
        raw_report = chat_with_events(
            event_callback,
            client,
            evaluator_model,
            "evaluator",
            build_pairwise_evaluator_prompt(
                case,
                transcript_x,
                transcript_y,
                target_role,
                behaviour_contract,
            ),
            thinking_delay_seconds,
            temperature=EVALUATOR_TEMPERATURE,
        )
        report_x, report_y, pair_summary = safe_pairwise_reports(raw_report, target_role, case)
        if report_x is not None and report_y is not None:
            report = report_x if current_is_x else report_y
            rescored_baseline = report_y if current_is_x else report_x
            report = add_comparison_metadata(report, rescored_baseline)
            report["comparison"]["stored_baseline_score"] = score_to_number(
                baseline_report.get("score") if isinstance(baseline_report, dict) else None
            )
            report["comparison"]["method"] = "blind-paired-rescore"
            report["comparison"]["summary"] = pair_summary
            report["comparison"]["updated_label"] = "X" if current_is_x else "Y"
        else:
            fallback_raw = chat_with_events(
                event_callback,
                client,
                evaluator_model,
                "evaluator",
                build_evaluator_prompt(
                    case,
                    transcript_text,
                    final_answer,
                    target_role,
                    patient_prompt if patient_prompt_under_test else custom_prompt,
                    evaluation_target_prompt,
                    "initial",
                    behaviour_contract=behaviour_contract,
                ),
                thinking_delay_seconds,
                temperature=EVALUATOR_TEMPERATURE,
            )
            report = add_comparison_metadata(safe_report(fallback_raw, target_role, case), baseline_report)
            report["comparison"]["method"] = "single-report-fallback"
            report["comparison"]["summary"] = pair_summary
    else:
        raw_report = chat_with_events(
            event_callback,
            client,
            evaluator_model,
            "evaluator",
            build_evaluator_prompt(
                case,
                transcript_text,
                final_answer,
                target_role,
                patient_prompt if patient_prompt_under_test else custom_prompt,
                evaluation_target_prompt,
                evaluation_mode,
                baseline_report,
                baseline_transcript,
                behaviour_contract,
            ),
            thinking_delay_seconds,
            temperature=EVALUATOR_TEMPERATURE,
        )
        report = safe_report(raw_report, target_role, case)
        report["evaluation_mode"] = "initial"
    report["behaviour_contract"] = behaviour_contract
    raw_prompt_guidance = chat_with_events(
        event_callback,
        client,
        evaluator_model,
        "evaluator",
        build_prompt_guidance_prompt(
            case,
            target_role,
            patient_prompt if patient_prompt_under_test else custom_prompt,
            evaluation_target_prompt,
            evaluation_mode,
        ),
        thinking_delay_seconds,
        temperature=EVALUATOR_TEMPERATURE,
    )
    report["prompt_guidance"] = safe_prompt_guidance(raw_prompt_guidance)
    report["evaluation_settings"] = {
        "scoring_version": SCORING_VERSION,
        "simulation_temperature": SIMULATION_TEMPERATURE,
        "evaluator_temperature": EVALUATOR_TEMPERATURE,
        "improver_temperature": IMPROVER_TEMPERATURE,
        "simulation_model": model,
        "counterpart_model": patient_model,
        "evaluator_model": evaluator_model,
    }
    prompt_preview = build_prompt_preview(
        {**payload, "behaviourContract": behaviour_contract},
        transcript_text,
        final_answer,
    )
    emit_optional(event_callback, {"type": "report", "report": report})
    return {
        "report": report,
        "score": score_to_number(report.get("score")),
        "transcript": transcript_events,
        "transcript_text": transcript_text,
        "final_answer": final_answer,
        "prompt_preview": prompt_preview,
    }


def paired_result_summary(items):
    valid = [item for item in items if isinstance(item.get("delta"), (int, float))]
    deltas = [float(item["delta"]) for item in valid]
    before_scores = [float(item["beforeScore"]) for item in valid if isinstance(item.get("beforeScore"), (int, float))]
    after_scores = [float(item["afterScore"]) for item in valid if isinstance(item.get("afterScore"), (int, float))]
    if not deltas:
        return {
            "validPairs": 0,
            "meanBefore": None,
            "meanAfter": None,
            "meanDelta": None,
            "medianDelta": None,
            "deltaCi95": [None, None],
            "winRate": None,
            "status": "invalid",
        }

    mean_delta = statistics.mean(deltas)
    if len(deltas) >= 2:
        standard_error = statistics.stdev(deltas) / math.sqrt(len(deltas))
        ci_low = mean_delta - 1.96 * standard_error
        ci_high = mean_delta + 1.96 * standard_error
    else:
        ci_low = None
        ci_high = None
    critical_regressions = sum(
        1
        for item in valid
        if any(
            flag.get("severity") == "critical"
            for flag in (item.get("afterReport", {}).get("clinical_evaluation", {}).get("safety_flags") or [])
            if isinstance(flag, dict)
        )
    )
    if len(deltas) < 3:
        status = "insufficient-evidence"
    elif ci_low is not None and ci_low > 0 and critical_regressions == 0:
        status = "promote"
    elif ci_high is not None and ci_high < 0:
        status = "reject"
    else:
        status = "inconclusive"
    return {
        "validPairs": len(deltas),
        "meanBefore": round(statistics.mean(before_scores), 2) if before_scores else None,
        "meanAfter": round(statistics.mean(after_scores), 2) if after_scores else None,
        "meanDelta": round(mean_delta, 2),
        "medianDelta": round(statistics.median(deltas), 2),
        "deltaCi95": [
            round(ci_low, 2) if ci_low is not None else None,
            round(ci_high, 2) if ci_high is not None else None,
        ],
        "winRate": round(sum(delta > 0 for delta in deltas) / len(deltas), 3),
        "criticalSafetyRegressions": critical_regressions,
        "status": status,
    }


def batch_optimise_clinician(handler, payload):
    repetitions = min(max(int(payload.get("runs") or 10), 1), 25)
    requested_levels = payload.get("levels")
    if not isinstance(requested_levels, list):
        requested_levels = ["low"]
    levels = [level for level in requested_levels if level in {"low", "medium", "high"}]
    if not levels:
        levels = ["low"]

    fallback_low_prompt = str(payload.get("lowPrompt") or "").strip() or (
        "You are interviewing a simulated patient. Ask relevant questions, then provide an assessment and plan. "
        "Speak naturally to the patient and do not preface turns with process labels, numbering, or descriptions of the question type. "
        "When ready, begin the final answer with exactly 'FINAL:'."
    )
    prompts_by_level = payload.get("promptsByLevel") if isinstance(payload.get("promptsByLevel"), dict) else {}
    default_prompts = {
        "low": fallback_low_prompt,
        "medium": (
            "Ask about one clinical topic at a time. Clarify the main problem, timing, severity, red flags, current medicines, "
            "recent medicine use, allergies or bad reactions, relevant medical history, pregnancy or contraception when relevant, "
            "and patient concerns. Speak naturally to the patient and do not preface turns with process labels, numbering, or descriptions of the question type. "
            "When ready, begin the final answer with exactly 'FINAL:' and keep it concise."
        ),
        "high": (
            "You are a careful clinician simulator completing a pharmacy-focused clinical assessment. Ask about one clinical topic at a time. "
            "Use plain, patient-friendly language. Clarify the main problem, timing, severity, red flags, impact on daily life, current medicines, "
            "recently used medicines, allergies or bad reactions, relevant medical history, pregnancy or contraception when relevant, and patient concerns. "
            "If the patient is vague, ask simple follow-up questions and ask them to check packets, notes, letters or lists. Do not finalise until you have "
            "enough information to address assessment, immediate safety issues, medicine options, medicine risks, counselling and follow-up. "
            "Speak naturally to the patient and do not preface turns with process labels, numbering, or descriptions of the question type. "
            "When ready, begin with exactly 'FINAL:' and give a concise management plan under 220 words with no markdown."
        ),
    }
    results = []
    summaries = {}
    total_pairs = len(levels) * repetitions
    index = 0

    def batch_case_events(index, total, level, repetition, batch_phase):
        def forward(event):
            if event.get("type") != "status":
                return
            emit(
                handler,
                {
                    "type": "batch_keepalive",
                    "index": index,
                    "total": total,
                    "level": level,
                    "repetition": repetition,
                    "phase": batch_phase,
                    "simulationPhase": event.get("phase"),
                    "turn": event.get("turn"),
                    "maxTurns": event.get("maxTurns"),
                },
            )

        return forward

    candidate_records = {}
    for level in levels:
        starting_prompt = str(prompts_by_level.get(level) or default_prompts[level]).strip()
        emit(
            handler,
            {
                "type": "batch_status",
                "index": index,
                "total": total_pairs,
                "runs": repetitions,
                "level": level,
                "repetition": 0,
                "phase": "pilot",
            },
        )
        pilot_payload = {
            **payload,
            "targetRole": "clinician",
            "promptLevel": "low",
            "targetPrompt": starting_prompt,
            "evaluationTargetPrompt": starting_prompt,
        }
        pilot = run_case(
            pilot_payload,
            event_callback=batch_case_events(index, total_pairs, level, 0, "pilot"),
            thinking_delay_seconds=0,
        )
        improved = improve_prompt(
            {
                **payload,
                "model": payload.get("clinicianModel") or payload.get("studentModel") or DEFAULT_MODEL,
                "promptTarget": "clinician",
                "currentPrompt": starting_prompt,
                "report": pilot["report"],
                "transcript": pilot["transcript"],
                "notes": (
                    f"Batch clinician-prompt optimisation for the {level} assigned task. Strengthen the frozen assigned behaviour "
                    "without adding hidden case facts or writing the clinical answer."
                ),
            }
        )
        candidate_records[level] = {
            "startingPrompt": starting_prompt,
            "improvedPrompt": improved["prompt"],
            "rationale": improved.get("rationale", ""),
            "validation": improved.get("validation", {}),
            "behaviourContract": pilot["report"].get("behaviour_contract"),
        }

        level_results = []
        for repetition in range(1, repetitions + 1):
            index += 1
            emit(
                handler,
                {
                    "type": "batch_status",
                    "index": index,
                    "total": total_pairs,
                    "runs": repetitions,
                    "level": level,
                    "repetition": repetition,
                    "phase": "before",
                },
            )
            base_payload = {
                **pilot_payload,
                "behaviourContract": candidate_records[level]["behaviourContract"],
            }
            before = run_case(
                base_payload,
                event_callback=batch_case_events(index, total_pairs, level, repetition, "before"),
                thinking_delay_seconds=0,
            )

            emit(
                handler,
                {
                    "type": "batch_status",
                    "index": index,
                    "total": total_pairs,
                    "runs": repetitions,
                    "level": level,
                    "repetition": repetition,
                    "phase": "after",
                },
            )
            after = run_case(
                {
                    **payload,
                    "targetRole": "clinician",
                    "promptLevel": "low",
                    "targetPrompt": improved["prompt"],
                    "evaluationTargetPrompt": starting_prompt,
                    "evaluationMode": "comparison",
                    "baselineReport": before["report"],
                    "baselineTranscript": before["transcript"],
                    "behaviourContract": candidate_records[level]["behaviourContract"],
                },
                event_callback=batch_case_events(index, total_pairs, level, repetition, "after"),
                thinking_delay_seconds=0,
            )
            comparison = after["report"].get("comparison") or {}
            paired_before_score = score_to_number(comparison.get("baseline_score"))
            paired_after_score = after["score"]
            paired_delta = score_to_number(comparison.get("delta"))
            result = {
                "index": index,
                "level": level,
                "repetition": repetition,
                "beforeScore": paired_before_score,
                "afterScore": paired_after_score,
                "delta": paired_delta,
                "initialBeforeScore": before["score"],
                "beforeReport": before["report"],
                "afterReport": after["report"],
                "startingPrompt": starting_prompt,
                "improvedPrompt": improved["prompt"],
                "rationale": improved.get("rationale", ""),
                "validation": improved.get("validation", {}),
            }
            results.append(result)
            level_results.append(result)
            emit(handler, {"type": "batch_item", "item": result})

        summaries[level] = {
            **paired_result_summary(level_results),
            **candidate_records[level],
        }

    promotable = [
        summary
        for summary in summaries.values()
        if summary.get("status") == "promote" and summary.get("validation", {}).get("passed", True)
    ]
    best = max(promotable, key=lambda item: item.get("deltaCi95", [float("-inf")])[0], default=None)
    emit(
        handler,
        {
            "type": "batch_done",
            "results": results,
            "summaries": summaries,
            "bestPrompt": best.get("improvedPrompt") if best else "",
            "selectionStatus": "promoted" if best else "no-candidate-cleared-evidence-threshold",
            "temperature": MODEL_TEMPERATURE,
            "evaluatorTemperature": EVALUATOR_TEMPERATURE,
            "settings": {
                "levels": levels,
                "repetitionsPerLevel": repetitions,
                "totalPairedTrials": total_pairs,
            },
        },
    )


def run_simulation(handler, payload):
    run_case(
        payload,
        event_callback=lambda event: emit(handler, event),
        thinking_delay_seconds=THINKING_DELAY_SECONDS,
    )
    emit(handler, {"type": "done"})


def start_run_job(payload):
    job_id = uuid.uuid4().hex
    with RUN_JOBS_LOCK:
        RUN_JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "createdAt": time.time(),
            "updatedAt": time.time(),
            "progress": {"phase": "starting", "turn": None, "maxTurns": None},
            "events": [],
            "result": None,
            "error": None,
        }

    def update_progress(event):
        with RUN_JOBS_LOCK:
            job = RUN_JOBS.get(job_id)
            if not job:
                return
            job["updatedAt"] = time.time()
            if event.get("type") == "status":
                job["progress"] = {
                    "phase": event.get("phase"),
                    "turn": event.get("turn"),
                    "maxTurns": event.get("maxTurns"),
                }
                return
            if event.get("type") == "message":
                events = job.setdefault("events", [])
                events.append({
                    "index": len(events),
                    "type": "message",
                    "role": event.get("role"),
                    "content": event.get("content"),
                    "final": bool(event.get("final")),
                })
                return
            if event.get("type") == "typing":
                events = job.setdefault("events", [])
                events.append({
                    "index": len(events),
                    "type": "typing",
                    "role": event.get("role"),
                    "seconds": event.get("seconds", THINKING_DELAY_SECONDS),
                })

    def worker():
        try:
            result = run_case(payload, event_callback=update_progress, thinking_delay_seconds=THINKING_DELAY_SECONDS)
            with RUN_JOBS_LOCK:
                if job_id in RUN_JOBS:
                    RUN_JOBS[job_id].update({
                        "status": "done",
                        "updatedAt": time.time(),
                        "progress": {"phase": "done", "turn": None, "maxTurns": None},
                        "result": result,
                    })
        except Exception as exc:
            traceback.print_exc()
            with RUN_JOBS_LOCK:
                if job_id in RUN_JOBS:
                    RUN_JOBS[job_id].update({
                        "status": "error",
                        "updatedAt": time.time(),
                        "progress": {"phase": "error", "turn": None, "maxTurns": None},
                        "error": str(exc),
                    })

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def get_run_job(job_id):
    with RUN_JOBS_LOCK:
        job = RUN_JOBS.get(job_id)
        if not job:
            return None
        return dict(job)


FRONTEND_PAGE = "/clinical-benchmark-ui-mockup.html"
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico"}


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        if self.path in {"/", FRONTEND_PAGE}:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def send_stream_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            data = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/cases":
            cases = [case_metadata(case) for case in CASE_LIBRARY.values() if case.get("active", True)]
            data = json.dumps({"defaultCaseId": DEFAULT_CASE_ID, "cases": cases}, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/case-file":
            query = parse_qs(parsed.query)
            case_id = (query.get("caseId") or [DEFAULT_CASE_ID])[0]
            case = CASE_LIBRARY.get(case_id) or CASE_LIBRARY[DEFAULT_CASE_ID]
            data = json.dumps(case_download_payload(case), indent=2).encode("utf-8")
            filename = re.sub(r"[^a-z0-9]+", "-", case["title"].lower()).strip("-")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{filename}.json"',
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/run-job":
            query = parse_qs(parsed.query)
            job_id = (query.get("jobId") or [""])[0]
            job = get_run_job(job_id)
            if not job:
                self.send_json({"error": "Run job not found."}, status=404)
                return
            self.send_json(job)
            return
        if parsed.path == "/":
            self.path = FRONTEND_PAGE
            parsed = urlparse(self.path)
        filename = parsed.path.lstrip("/")
        is_frontend = parsed.path == FRONTEND_PAGE
        is_image = (
            "/" not in filename
            and not filename.startswith(".")
            and Path(filename).suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        )
        if not (is_frontend or is_image):
            self.send_error(404, "Not found")
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/run", "/api/run-sync", "/api/run-job", "/api/prompt-preview", "/api/improve-prompt", "/api/batch-clinician"}:
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body or "{}")

        if parsed.path == "/api/prompt-preview":
            try:
                self.send_json(build_prompt_preview(payload))
            except Exception as exc:
                traceback.print_exc()
                self.send_json({"error": str(exc)})
            return

        if parsed.path == "/api/improve-prompt":
            try:
                response = improve_prompt(payload)
                self.send_json(response)
            except Exception as exc:
                traceback.print_exc()
                self.send_json({"error": str(exc)})
            return

        if parsed.path == "/api/run-sync":
            try:
                response = run_case(payload, event_callback=None, thinking_delay_seconds=0)
                self.send_json(response)
            except Exception as exc:
                traceback.print_exc()
                self.send_json({"error": str(exc)})
            return

        if parsed.path == "/api/run-job":
            try:
                job_id = start_run_job(payload)
                self.send_json({"jobId": job_id, "status": "running"})
            except Exception as exc:
                traceback.print_exc()
                self.send_json({"error": str(exc)})
            return

        if parsed.path == "/api/batch-clinician":
            self.send_stream_headers()
            try:
                emit(self, {"type": "keepalive", "phase": "batch-start"})
                batch_optimise_clinician(self, payload)
            except ClientDisconnected:
                return
            except Exception as exc:
                traceback.print_exc()
                try:
                    emit(self, {"type": "error", "message": str(exc)})
                except ClientDisconnected:
                    return
            return

        self.send_stream_headers()

        try:
            emit(self, {"type": "keepalive", "phase": "run-start"})
            run_simulation(self, payload)
        except ClientDisconnected:
            return
        except Exception as exc:
            traceback.print_exc()
            try:
                emit(self, {"type": "error", "message": str(exc)})
            except ClientDisconnected:
                return


def main():
    os.chdir(ROOT)
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Clinical Benchmark Studio running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
