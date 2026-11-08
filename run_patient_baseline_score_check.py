#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path

import app


BASELINE_PROMPT = (
    "You are a patient attending a routine medication-history appointment. "
    "Answer every question honestly, but never volunteer additional information unless you are asked directly."
)


def metric_scores(report):
    return {
        metric.get("name", f"Metric {index + 1}"): metric.get("score", "n/a")
        for index, metric in enumerate(report.get("metrics") or [])
    }


def main():
    base_payload = {
        "caseId": "case-559",
        "targetRole": "patient",
        "promptLevel": "low",
        "targetPrompt": BASELINE_PROMPT,
        "clinicianModel": app.DEFAULT_MODEL,
        "studentModel": app.DEFAULT_MODEL,
        "patientModel": app.DEFAULT_MODEL,
        "evaluatorModel": app.DEFAULT_MODEL,
    }

    before = app.run_case(base_payload, thinking_delay_seconds=0)
    improved = app.improve_prompt(
        {
            "caseId": "case-559",
            "model": app.DEFAULT_MODEL,
            "promptTarget": "patient",
            "currentPrompt": BASELINE_PROMPT,
            "report": before["report"],
            "transcript": before["transcript"],
            "notes": "Controlled test: improve the built-in Baseline patient prompt using the app prompt-improver pathway.",
        }
    )

    after_payload = {
        **base_payload,
        "targetPrompt": improved["prompt"],
        "evaluationTargetPrompt": BASELINE_PROMPT,
        "evaluationMode": "comparison",
        "baselineReport": before["report"],
        "baselineTranscript": before["transcript"],
        "behaviourContract": before["report"].get("behaviour_contract"),
    }
    after = app.run_case(after_payload, thinking_delay_seconds=0)

    result = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "caseId": "case-559",
        "caseTitle": app.CASE_LIBRARY["case-559"]["title"],
        "targetRole": "patient",
        "model": app.DEFAULT_MODEL,
        "baselinePrompt": BASELINE_PROMPT,
        "improvedPrompt": improved["prompt"],
        "improvementRationale": improved.get("rationale", ""),
        "beforeScore": app.score_to_number(after.get("report", {}).get("comparison", {}).get("baseline_score")),
        "afterScore": after.get("score"),
        "delta": app.score_to_number(after.get("report", {}).get("comparison", {}).get("delta")),
        "initialBeforeScore": before.get("score"),
        "beforeMetricScores": metric_scores(before["report"]),
        "afterMetricScores": metric_scores(after["report"]),
        "beforeReport": before["report"],
        "afterReport": after["report"],
        "beforeTranscriptTurns": len(before.get("transcript") or []),
        "afterTranscriptTurns": len(after.get("transcript") or []),
    }

    output_path = Path(__file__).with_name("patient_baseline_score_check.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({
        "output": str(output_path),
        "beforeScore": result["beforeScore"],
        "afterScore": result["afterScore"],
        "beforeMetricScores": result["beforeMetricScores"],
        "afterMetricScores": result["afterMetricScores"],
        "improvementRationale": result["improvementRationale"],
    }, indent=2))


if __name__ == "__main__":
    main()
