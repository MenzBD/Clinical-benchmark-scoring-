import json
import unittest

import app


class AssignedTaskScoringTests(unittest.TestCase):
    def setUp(self):
        self.case = app.CASE_LIBRARY["case-559"]

    def test_weighted_assigned_task_score(self):
        metrics = [
            {"name": "Assigned Task Fidelity", "score": "8 / 10"},
            {"name": "Conversational Realism", "score": "7 / 10"},
            {"name": "Information Behaviour", "score": "6 / 10"},
            {"name": "Case Fidelity", "score": "9 / 10"},
        ]
        self.assertEqual(app.derived_overall_score(metrics), 75.0)

    def test_angry_behaviour_is_explicitly_successful(self):
        contract = app.fallback_behaviour_contract(
            self.case,
            "patient",
            "You are the patient. Be angry and impatient when questioned.",
        )
        messages = app.build_evaluator_prompt(
            self.case,
            "Pharmacist: What do you take?\n\nPatient: I'm sick of these questions.",
            "",
            "patient",
            evaluation_target_prompt="You are the patient. Be angry and impatient when questioned.",
            behaviour_contract=contract,
        )
        system_text = messages[0]["content"]
        self.assertIn("An angry role should be visibly angry", system_text)
        self.assertIn("Never reward calmness", system_text)

    def test_custom_patient_receives_truth_without_forced_disclosure(self):
        messages = app.build_custom_patient_messages(
            self.case,
            "You are the patient. Be angry, terse and reluctant.",
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("anger", messages[1]["content"])
        self.assertIn("Knowing a fact does not require volunteering it", messages[1]["content"])
        self.assertIn("paracetamol/codeine", messages[2]["content"])

    def test_report_score_ignores_model_supplied_overall(self):
        parsed = {
            "score": 1,
            "metrics": [
                {"name": "Assigned Task Fidelity", "score": "8 / 10", "note": "angry throughout"},
                {"name": "Conversational Realism", "score": "7 / 10", "note": "believable"},
                {"name": "Information Behaviour", "score": "6 / 10", "note": "appropriately terse"},
                {"name": "Case Fidelity", "score": "9 / 10", "note": "no invention"},
            ],
            "clinical_evaluation": {"applicable": False, "items": [], "safety_flags": []},
        }
        report = app.normalise_report(parsed, json.dumps(parsed), "patient", self.case)
        self.assertEqual(report["score"], 75.0)
        self.assertFalse(report["clinical_evaluation"]["applicable"])

    def test_blind_pairwise_reports_are_normalised_independently(self):
        def raw_report(task_score):
            return {
                "metrics": [
                    {"name": "Assigned Task Fidelity", "score": f"{task_score} / 10", "note": "evidence"},
                    {"name": "Conversational Realism", "score": "7 / 10", "note": "evidence"},
                    {"name": "Information Behaviour", "score": "7 / 10", "note": "evidence"},
                    {"name": "Case Fidelity", "score": "8 / 10", "note": "evidence"},
                ],
                "clinical_evaluation": {"applicable": False, "items": [], "safety_flags": []},
            }

        raw = json.dumps({"X": raw_report(5), "Y": raw_report(9), "comparison_summary": "Y is stronger."})
        report_x, report_y, summary = app.safe_pairwise_reports(raw, "patient", self.case)
        self.assertEqual(report_x["score"], 61.0)
        self.assertEqual(report_y["score"], 81.0)
        self.assertEqual(summary, "Y is stronger.")

    def test_evaluator_shape_has_no_numeric_score_anchors(self):
        shape = app.evaluator_output_shape(False)
        self.assertEqual(shape.count('"score":null'), 4)
        self.assertNotIn('"score":"7.5 / 10"', shape)
        self.assertNotIn('"score":"7 / 10"', shape)
        self.assertNotIn('"score":"8 / 10"', shape)
        self.assertNotIn('"score":"9 / 10"', shape)

    def test_critical_clinical_safety_flag_caps_separate_score(self):
        clinical = {
            "applicable": True,
            "items": [
                {"criterion": criterion, "score": 1, "evidence": "covered"}
                for criterion in self.case["rubric"]
            ],
            "safety_flags": [{"severity": "critical", "note": "Unsafe recommendation"}],
        }
        result = app.normalise_clinical_evaluation(clinical, self.case, "clinician")
        self.assertEqual(result["uncapped_score"], 100.0)
        self.assertEqual(result["score"], 49.0)

    def test_prompt_validator_rejects_case_like_numbers_and_long_prompts(self):
        issues = app.prompt_validation_issues(
            "You are the patient. Be angry.",
            "You are the patient. Be angry and disclose that you used medicine on 11 days. "
            + "word " * 180,
        )
        self.assertTrue(any("too long" in issue for issue in issues))
        self.assertTrue(any("11" in issue for issue in issues))

    def test_prompt_validator_rejects_hidden_case_terms(self):
        issues = app.prompt_validation_issues(
            "You are the patient. Be angry and reluctant.",
            "You are the patient. Be angry and reluctant. Mention migraine with aura, codeine and asthma.",
            case=self.case,
        )
        self.assertTrue(any("private case-specific content" in issue for issue in issues))
        self.assertTrue(any("migraine" in issue for issue in issues))

    def test_prompt_validator_catches_hyphenated_leakage_and_dialogue_examples(self):
        issues = app.prompt_validation_issues(
            "You are the patient. Be angry and reluctant.",
            "You are the patient. Be angry and reluctant in the migraine-with-aura scenario. Say “Ugh, fine.”",
            case=self.case,
        )
        self.assertTrue(any("migraine" in issue for issue in issues))
        self.assertTrue(any("worked dialogue examples" in issue for issue in issues))

    def test_paired_summary_requires_replicated_positive_evidence(self):
        items = [
            {"beforeScore": 50, "afterScore": 60, "delta": 10, "afterReport": {}},
            {"beforeScore": 52, "afterScore": 62, "delta": 10, "afterReport": {}},
            {"beforeScore": 49, "afterScore": 59, "delta": 10, "afterReport": {}},
        ]
        summary = app.paired_result_summary(items)
        self.assertEqual(summary["status"], "promote")
        self.assertEqual(summary["meanDelta"], 10.0)
        self.assertEqual(summary["winRate"], 1.0)


if __name__ == "__main__":
    unittest.main()
