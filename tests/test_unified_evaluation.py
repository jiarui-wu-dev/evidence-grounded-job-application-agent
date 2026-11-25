import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.unified_report import build_report, cv_checks, main, markdown_report


ROOT = Path(__file__).resolve().parents[1] / "evaluation"


class TestUnifiedEvaluation(unittest.TestCase):
    @unittest.skipUnless((ROOT / "results/baseline_v1.json").exists(), "Private retrieval artifacts unavailable.")
    def test_saved_retrieval_scores_keep_system_and_dataset_provenance(self):
        report = build_report(ROOT)
        retrieval = report["retrieval"]
        self.assertTrue(all(value is None for value in retrieval["live_graph"]["metrics"].values()))
        self.assertEqual(len(retrieval["benchmarks"]), 5)
        for item in retrieval["benchmarks"]:
            with (ROOT / item["source"]).open(encoding="utf-8") as stream:
                original = json.load(stream)
            expected = original["overall"]
            if "system" in original:
                expected = {"Reranked": expected}
            actual = {name: system["metrics"] for name, system in item["systems"].items()}
            self.assertEqual(actual, expected)
            self.assertEqual(item["evaluation_file"], original.get("evaluation_file"))
        self.assertEqual(retrieval["benchmarks"][3]["candidate_k"], 10)
        self.assertEqual(retrieval["benchmarks"][4]["hybrid_config"]["dense_weight"], 2.0)

    @unittest.skipUnless((ROOT / "results/match_gap_v1.json").exists(), "Private grounding artifacts unavailable.")
    def test_grounding_and_workflow_preserve_historical_results(self):
        report = build_report(ROOT)
        grounding = report["grounding"]
        with (ROOT / "results/match_gap_v1.json").open(encoding="utf-8") as stream:
            original = json.load(stream)
        self.assertEqual(grounding["match_gap_benchmark"]["metrics"], original["metrics"])
        self.assertIsNone(grounding["independently_verified_unsupported_claim_leakage"])
        final = next(item for item in grounding["cv_validations"]
                     if item["run"] == "composition_before_targeted_revalidation"
                     and item["stage"] == "final")
        self.assertEqual(final["validator_summary"]["UNSUPPORTED"], 2)
        self.assertEqual(final["citation_coverage"], {
            "claims": 12, "claims_with_evidence_ids": 12, "coverage": 1.0,
        })
        workflow = {item["run"]: item for item in report["workflow"]}
        self.assertEqual(workflow["revision_graph"]["final_status"], "needs_human_review")
        self.assertEqual(workflow["revision_graph"]["revision_rounds"], 2)
        self.assertIsNone(workflow["full_graph"]["verdict_delta"])
        self.assertIsNone(workflow["full_graph"]["llm_calls"])
        self.assertEqual(workflow["composition_before_targeted_revalidation"]["verdict_delta"], {
            "changed_verdicts": 1, "added_claim_ids": 0, "removed_claim_ids": 0,
        })

    def test_readability_flags_only_recorded_limits_and_normalized_duplicates(self):
        cv = {
            "profile": {"text": "Two words", "evidence_ids": ["p"]},
            "experience": [{
                "source_id": "one",
                "bullets": [{"text": "Made things"}, {"text": " made  THINGS ", "evidence_ids": ["b"]}],
            }],
            "selected_projects": [{
                "source_id": "two",
                "bullets": [{"text": "Short", "evidence_ids": ["c"]}],
            }],
        }
        citations, quality = cv_checks(cv, {
            "sources": [{"source_id": "one", "max_bullets": 1, "max_words_per_bullet": 1}],
        })
        self.assertEqual(citations["coverage"], 0.75)
        self.assertEqual(quality["profile_words"], 2)
        self.assertEqual(quality["duplicate_bullets"], 1)
        self.assertEqual(quality["bullets_by_section"]["research_experience"], 0)
        limited, unrecorded = quality["sources"]
        self.assertTrue(limited["bullet_count_over_limit"])
        self.assertEqual(limited["bullets_over_word_limit"], 2)
        self.assertIsNone(unrecorded["bullets_over_word_limit"])
        self.assertIsNone(unrecorded["bullet_count_over_limit"])

    def test_missing_artifacts_and_explicit_zero_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixtures").mkdir()
            (root / "fixtures/ai_engineer_langgraph_full_result.json").write_text(
                '{"llm_calls": 0, "retry_count": 0, "warnings": []}', encoding="utf-8",
            )
            report = build_report(root)
        self.assertFalse(report["retrieval"]["benchmarks"][0]["available"])
        self.assertIn("results/baseline_v1.json", report["missing_artifacts"])
        self.assertIsNone(report["grounding"]["match_gap_benchmark"]["case_count"])
        run = report["workflow"][0]
        self.assertEqual([run["llm_calls"], run["retry_count"], run["warning_count"]], [0, 0, 0])
        self.assertIsNone(run["latency_seconds"])
        self.assertIn("`results/baseline_v1.json` (unavailable)", markdown_report(report))

    def test_cli_writes_reproducible_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["unified_report", "--evaluation-root", str(ROOT), "--output-dir", directory]
            json_path = Path(directory) / "unified_evaluation.json"
            markdown_path = Path(directory) / "unified_evaluation.md"
            with patch("sys.argv", args):
                main()
                first_output = (json_path.read_bytes(), markdown_path.read_bytes())
                main()
                self.assertEqual(first_output, (json_path.read_bytes(), markdown_path.read_bytes()))
            report = json.loads(first_output[0])
            markdown = first_output[1].decode("utf-8")
        self.assertEqual(report["measurement_mode"], "saved_artifacts")
        for heading in ("Retrieval quality", "Grounding quality", "Workflow behavior", "CV readability"):
            self.assertIn(f"## {heading}", markdown)
        self.assertIn("Live graph · end to end", markdown)
        self.assertIn("Unavailable", markdown)
