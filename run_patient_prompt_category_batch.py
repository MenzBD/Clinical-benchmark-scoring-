#!/usr/bin/env python3
import csv
import json
import os
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import app


RUNS_PER_CATEGORY = int(os.environ.get("RUNS_PER_CATEGORY", "5"))
RUN_LABEL = os.environ.get("RUN_LABEL") or f"{RUNS_PER_CATEGORY}run"
RANDOM_SEED = 20260728
CASE_ID = "case-559"
TARGET_ROLE = "patient"
MODEL = app.DEFAULT_MODEL
ROLE_PREFIX = "You are the patient speaking with the pharmacist during a medication history and review."
DOMAINS = list(app.TASK_METRIC_WEIGHTS)

PROMPTS = {
    "Baseline": [
        f"{ROLE_PREFIX} Be an ordinary, cooperative patient and let the other person lead.",
        f"{ROLE_PREFIX} Answer honestly, but do not add much unless asked.",
        f"{ROLE_PREFIX} Be polite and fairly passive, with short answers.",
        f"{ROLE_PREFIX} Mostly answer only the question you were asked.",
        f"{ROLE_PREFIX} Use average recall and avoid offering extra details.",
        f"{ROLE_PREFIX} Let the other person uncover the history through questions.",
        f"{ROLE_PREFIX} Be cooperative, but do not anticipate what might matter.",
        f"{ROLE_PREFIX} Speak plainly and wait for follow-up questions.",
        f"{ROLE_PREFIX} Give one relevant piece of information at a time.",
        f"{ROLE_PREFIX} Act like a typical patient who is helpful but incomplete.",
    ],
    "Challenging": [
        f"{ROLE_PREFIX} Be difficult and make the other person work for information.",
        f"{ROLE_PREFIX} Have poor recall and often seem unsure.",
        f"{ROLE_PREFIX} Be irritated by the questions and answer reluctantly.",
        f"{ROLE_PREFIX} Give vague, brief and incomplete answers.",
        f"{ROLE_PREFIX} Get side-tracked instead of answering directly.",
        f"{ROLE_PREFIX} Misunderstand some questions and need redirecting.",
        f"{ROLE_PREFIX} Do not volunteer important details.",
        f"{ROLE_PREFIX} Be impatient when the other person keeps pressing.",
        f"{ROLE_PREFIX} Confuse some names and details.",
        f"{ROLE_PREFIX} Combine poor recall, tangents and minimal answers.",
    ],
    "Supportive": [
        f"{ROLE_PREFIX} Be helpful and volunteer information that seems useful.",
        f"{ROLE_PREFIX} Have good recall and answer confidently.",
        f"{ROLE_PREFIX} Be engaged and forthcoming.",
        f"{ROLE_PREFIX} Give clear answers and try to prevent misunderstandings.",
        f"{ROLE_PREFIX} Mention useful context when it comes to mind.",
        f"{ROLE_PREFIX} Try to help the other person understand the situation.",
        f"{ROLE_PREFIX} Add something later if you realise you forgot it.",
        f"{ROLE_PREFIX} Treat the conversation as collaborative.",
        f"{ROLE_PREFIX} Answer questions fully and openly.",
        f"{ROLE_PREFIX} Make it easy to obtain an accurate picture.",
    ],
}


def metric_scores(report):
    return {
        metric.get("name", f"Metric {index + 1}"): app.score_to_number(metric.get("score"))
        for index, metric in enumerate(report.get("metrics") or [])
    }


def run_payload(prompt, evaluation_target_prompt=None, baseline=None, behaviour_contract=None):
    payload = {
        "caseId": CASE_ID,
        "targetRole": TARGET_ROLE,
        "promptLevel": "low",
        "targetPrompt": prompt,
        "evaluationTargetPrompt": evaluation_target_prompt or prompt,
        "clinicianModel": MODEL,
        "studentModel": MODEL,
        "patientModel": MODEL,
        "evaluatorModel": MODEL,
    }
    if baseline:
        payload.update({
            "evaluationMode": "comparison",
            "baselineReport": baseline["report"],
            "baselineTranscript": baseline["transcript"],
        })
    if behaviour_contract:
        payload["behaviourContract"] = behaviour_contract
    return payload


def average(values):
    numeric = [value for value in values if isinstance(value, (int, float))]
    return round(statistics.mean(numeric), 2) if numeric else None


def summarise(results):
    domains = DOMAINS
    summaries = {}
    for category in PROMPTS:
        items = [item for item in results if item["category"] == category]
        summary = {
            "runs": len(items),
            "beforeOverallMean": average([item["beforeScore"] for item in items]),
            "afterOverallMean": average([item["afterScore"] for item in items]),
            "overallDeltaMean": average([item["delta"] for item in items]),
            "pairedEvidence": app.paired_result_summary(items),
        }
        for domain in domains:
            before_key = f"before{domain.replace(' ', '')}Mean"
            after_key = f"after{domain.replace(' ', '')}Mean"
            delta_key = f"{domain.replace(' ', '')}DeltaMean"
            before_values = [item["beforeMetricScores"].get(domain) for item in items]
            after_values = [item["afterMetricScores"].get(domain) for item in items]
            summary[before_key] = average(before_values)
            summary[after_key] = average(after_values)
            summary[delta_key] = average([
                after - before
                for before, after in zip(before_values, after_values)
                if before is not None and after is not None
            ])
        summaries[category] = summary
    return summaries


def main():
    rng = random.Random(RANDOM_SEED)
    results = []
    case_title = app.CASE_LIBRARY[CASE_ID]["title"]

    total = len(PROMPTS) * RUNS_PER_CATEGORY
    index = 0
    for category, prompts in PROMPTS.items():
        prompt_index = rng.randrange(len(prompts))
        prompt = prompts[prompt_index]
        print(f"[pilot] {category}: seed {prompt_index + 1}", flush=True)
        pilot = app.run_case(run_payload(prompt, prompt), thinking_delay_seconds=0)
        improved = app.improve_prompt(
            {
                "caseId": CASE_ID,
                "model": MODEL,
                "promptTarget": TARGET_ROLE,
                "currentPrompt": prompt,
                "report": pilot["report"],
                "transcript": pilot["transcript"],
                "notes": (
                    f"Replicated patient-prompt optimisation. Category: {category}. "
                    "Strengthen the frozen assigned behaviour. An angry, difficult or reluctant patient must remain observably angry, difficult or reluctant; "
                    "a supportive patient must remain supportive. Do not add hidden case facts or optimise for universal cooperation."
                ),
            }
        )
        behaviour_contract = pilot["report"].get("behaviour_contract")
        for repetition in range(1, RUNS_PER_CATEGORY + 1):
            index += 1
            print(f"[{index}/{total}] {category} run {repetition}: seed {prompt_index + 1}", flush=True)

            before = app.run_case(
                run_payload(prompt, prompt, behaviour_contract=behaviour_contract),
                thinking_delay_seconds=0,
            )
            after = app.run_case(
                run_payload(
                    improved["prompt"],
                    prompt,
                    baseline=before,
                    behaviour_contract=behaviour_contract,
                ),
                thinking_delay_seconds=0,
            )

            before_scores = metric_scores(before["report"])
            after_scores = metric_scores(after["report"])
            item = {
                "index": index,
                "category": category,
                "repetition": repetition,
                "promptSeedIndex": prompt_index + 1,
                "startingPrompt": prompt,
                "improvedPrompt": improved["prompt"],
                "improvementRationale": improved.get("rationale", ""),
                "beforeScore": app.score_to_number(after.get("report", {}).get("comparison", {}).get("baseline_score")),
                "afterScore": after.get("score"),
                "delta": app.score_to_number(after.get("report", {}).get("comparison", {}).get("delta")),
                "initialBeforeScore": before.get("score"),
                "beforeMetricScores": before_scores,
                "afterMetricScores": after_scores,
                "beforeReport": before["report"],
                "afterReport": after["report"],
                "beforeTranscript": before.get("transcript_text", ""),
                "afterTranscript": after.get("transcript_text", ""),
                "beforeTranscriptTurns": len(before.get("transcript") or []),
                "afterTranscriptTurns": len(after.get("transcript") or []),
                "promptValidation": improved.get("validation", {}),
            }
            results.append(item)
            delta_text = f"{item['delta']:+.2f}" if isinstance(item["delta"], (int, float)) else "n/a"
            print(
                f"    overall {item['beforeScore']} -> {item['afterScore']} "
                f"({delta_text}); metrics {before_scores} -> {after_scores}",
                flush=True,
            )

    summary = summarise(results)
    payload = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "caseId": CASE_ID,
        "caseTitle": case_title,
        "targetRole": TARGET_ROLE,
        "model": MODEL,
        "runsPerCategory": RUNS_PER_CATEGORY,
        "runLabel": RUN_LABEL,
        "randomSeed": RANDOM_SEED,
        "evaluationMethod": (
            "One candidate Prompt B is generated per category from a pilot run. Each repetition reruns the same Prompt A and Prompt B, "
            "then blindly rescores their transcripts together under anonymous X/Y labels using the frozen assigned-behaviour contract. "
            "The reported delta is the paired rescore delta, not a subtraction across separate evaluator calls."
        ),
        "scoringVersion": app.SCORING_VERSION,
        "temperatures": {
            "simulation": app.SIMULATION_TEMPERATURE,
            "evaluator": app.EVALUATOR_TEMPERATURE,
            "improver": app.IMPROVER_TEMPERATURE,
        },
        "summary": summary,
        "results": results,
    }

    root = Path(__file__).resolve().parent
    json_path = root / f"patient_prompt_category_batch_results_{RUN_LABEL}.json"
    csv_path = root / f"patient_prompt_category_batch_results_{RUN_LABEL}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    domains = DOMAINS
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "category",
                "repetition",
                "promptSeedIndex",
                "beforeScore",
                "afterScore",
                "delta",
                *[f"before_{domain}" for domain in domains],
                *[f"after_{domain}" for domain in domains],
                "startingPrompt",
                "improvedPrompt",
                "improvementRationale",
            ],
        )
        writer.writeheader()
        for item in results:
            row = {
                "index": item["index"],
                "category": item["category"],
                "repetition": item["repetition"],
                "promptSeedIndex": item["promptSeedIndex"],
                "beforeScore": item["beforeScore"],
                "afterScore": item["afterScore"],
                "delta": item["delta"],
                "startingPrompt": item["startingPrompt"],
                "improvedPrompt": item["improvedPrompt"],
                "improvementRationale": item["improvementRationale"],
            }
            for domain in domains:
                row[f"before_{domain}"] = item["beforeMetricScores"].get(domain)
                row[f"after_{domain}"] = item["afterMetricScores"].get(domain)
            writer.writerow(row)

    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"Batch failed: {exc}", file=sys.stderr, flush=True)
        raise
