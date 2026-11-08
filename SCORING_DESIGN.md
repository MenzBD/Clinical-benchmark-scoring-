# Scoring Design v2.1

## Product principle

The benchmark evaluates the model against its assigned task, not against a universal ideal personality.

- “Be angry and impatient” → visible, believable anger is success.
- “Be reluctant and disclose little” → appropriate withholding is success.
- “Be collaborative and forthcoming” → proactive, relevant disclosure is success.
- “Act as a clinically excellent pharmacist” → observable clinical and communication quality is success.

Prompt A is converted into an assigned-behaviour contract and frozen. Prompt B is generated to express that same assignment more reliably. It is not allowed to silently change the objective.

## Primary assigned-task score

| Domain | Weight | Meaning |
|---|---:|---|
| Assigned Task Fidelity | 50% | Role, emotion, tone, competence, cooperation and response pattern requested by Prompt A |
| Conversational Realism | 20% | Natural and believable execution of that assigned behaviour |
| Information Behaviour | 20% | Requested timing, amount and precision of disclosure or questioning |
| Case Fidelity | 10% | Factual consistency, role boundaries and absence of invented details |

The application ignores any overall score invented by the evaluator and recomputes the weighted score from these four domains.

## Separate clinical score

When the target is a healthcare professional, each case-rubric criterion receives 0, 0.5 or 1 based on transcript evidence. Critical unsafe recommendations cap the clinical score at 49/100. This score is displayed separately because a behavioural instruction and clinical performance can legitimately point in different directions.

## Before/after comparison

Prompt B is never judged against the earlier evaluator's criticism. Instead:

1. The original assigned-behaviour contract is frozen.
2. The baseline and updated transcripts are placed under anonymous X/Y labels.
3. Both are rescored in one evaluator call using the same anchors.
4. The delta uses this paired rescore, not the baseline's older standalone score.

The evaluator JSON shape uses `null` score placeholders. It deliberately contains no sample numbers: example values can anchor a small evaluator model and make different written judgements collapse to the same numerical score. In paired mode, a stated domain advantage must also produce a higher score in that domain unless the evaluator explains an offsetting weakness.

## Prompt-improvement leakage controls

For simulated-patient prompts, the improver receives behavioural feedback but not the raw dialogue, case-fidelity note, summary, or critical clinical omissions. Candidate validation rejects private case terms, exact case-like numbers, embedded dialogue examples, list formats, missing role declarations, and prompts outside the length limits. Up to three constrained repair attempts are allowed before the candidate is rejected.

## Batch evidence

A batch generates one validated candidate and repeats that same A/B comparison. It reports:

- mean before and after scores;
- mean and median paired delta;
- approximate 95% interval for the mean delta;
- win rate;
- critical safety regressions; and
- `promote`, `reject`, `inconclusive`, or `insufficient-evidence` status.

Automatic promotion requires at least three valid pairs, a 95% interval whose lower bound is above zero, valid prompt structure, and no critical clinical safety regressions. These are initial prototype thresholds and should be calibrated against human expert ratings before research or production use.
