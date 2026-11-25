import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.evaluation import multi_jd_benchmark as benchmark
from src.evaluation.multi_jd_metrics import stage_a, stage_b, summarize
from src.evaluation.unified_report import build_report, markdown_report
from src.retrieval.profile_documents import load_profile_documents


ROOT = Path(__file__).resolve().parents[1] / "evaluation"
GOLD_PATH = ROOT / "multi_jd_gold.json"


class OpenAIConnectionError(Exception):
    pass


class TestMultiJDBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = GOLD_PATH if GOLD_PATH.exists() else Path(__file__).with_name("fixtures") / "public_multi_jd_gold.json"
        cls.gold = benchmark.load_json(path)

    def test_gold_against_candidate_kb(self):
        documents = load_profile_documents()
        if documents and GOLD_PATH.is_file():
            benchmark.validate_gold(self.gold, documents)
        cases = self.gold["cases"]
        self.assertEqual(len(cases), 5)
        self.assertEqual(sum(case["full_cv"] for case in cases), 2)
        rows = [row for case in cases for row in case["requirements"]]
        self.assertEqual(len(rows), 23)
        self.assertEqual(sum(row["status"] is None for row in rows), 1)
        self.assertEqual({row["status"] for row in rows}, {"SUPPORTED", "PARTIAL", "GAP", None})

    def test_stage_a_tracks_errors_by_origin_and_separate_denominators(self):
        case = self.gold["cases"][0]
        run = benchmark.new_run(self.gold, {})
        record = run["cases"][case["id"]]
        record["extracted_requirements"] = [
            {"requirement_id": "must_1", "requirement": "PyTorch"},
            {"requirement_id": "nice_1", "requirement": "NVIDIA Triton"},
        ]
        record["state"].update(
            evidence_package={
                "evidence": [{"evidence_id": "maritime_asr_skills"}],
                "query_results": [{"results": [{"evidence_id": "maritime_asr_evidence_0"}]}],
            },
            match_gap={"requirements": [
                {"requirement_id": "must_1", "status": "SUPPORTED", "evidence_ids": ["maritime_asr_skills"]},
                {"requirement_id": "nice_1", "status": "GAP", "evidence_ids": []},
            ]},
            application_strategy={"priority_sources": [{"source_id": "maritime_asr"}]},
            cv_composition={"sources": [{"source_id": "scientific_gui"}]},
        )
        result = stage_a(case, record)
        self.assertEqual(result["classification_accuracy_aligned"], 1)
        self.assertEqual(result["classification_accuracy_end_to_end"], 0.5)
        self.assertEqual(result["unsupported_correct"], 1)
        self.assertEqual(result["retrieval"]["recall@5"], 0.2)
        self.assertEqual(result["composition_sources"]["core_coverage"], 0)
        self.assertIn("evidence_aggregation", {failure["stage"] for failure in result["failures"]})
        self.assertIn("jd_extraction", {failure["stage"] for failure in result["failures"]})

    def test_stage_b_verdicts_use_actual_trace_and_leave_semantic_leakage_unknown(self):
        run = benchmark.new_run(self.gold, {})
        record = run["cases"]["qa"]
        initial = {"claim_id": "profile", "status": "SUPPORTED", "reason": "original"}
        bullet = {"claim_id": "experience:a:bullet_1", "status": "UNSUPPORTED"}
        record["validation_trace"] = [
            {"changed_claim_ids": None, "validation": {"summary": {}, "results": [initial, bullet]}},
            {"changed_claim_ids": [bullet["claim_id"]], "validation": {
                "summary": {}, "results": [initial, {**bullet, "status": "SUPPORTED"}],
            }},
        ]
        result = stage_b(record)
        self.assertEqual(result["targeted_claim_evaluations"], 1)
        self.assertEqual(result["untouched_verdict_preservation_rate"], 1)
        self.assertIsNone(result["independent_unsupported_claim_leakage"])

    def test_resolved_model_telemetry_uses_response_metadata_and_aggregates_calls(self):
        response = SimpleNamespace(
            response_metadata={"model_name": "provider/resolved-model"},
            additional_kwargs={},
        )
        self.assertEqual(
            benchmark.resolved_model_from_response(response),
            "provider/resolved-model",
        )
        run = benchmark.new_run(self.gold, {"model": "openrouter/free"})
        run["cases"]["ai_ml"]["calls"] = [
            {"phase": "A", "status": "success", "resolved_model": "provider/resolved-model"},
            {"phase": "A", "status": "success", "resolved_model": "provider/resolved-model"},
            {"phase": "A", "status": "success", "resolved_model": None},
        ]
        result = summarize(self.gold, run)
        self.assertEqual(result["configured_model"], "openrouter/free")
        self.assertEqual(result["resolved_model_distribution"], {
            "provider/resolved-model": 2,
        })
        self.assertEqual(result["successful_calls_without_resolved_model"], 1)

    def test_resume_preserves_completed_node_and_stops_on_connection_error(self):
        run = benchmark.new_run(self.gold, {})
        record = run["cases"]["ai_ml"]
        record["completed_steps"].append("analyze_jd")
        with patch.object(benchmark.graph, "analyze_jd_node") as analyze_jd, patch.object(
            benchmark.graph, "build_queries_node", return_value={"queries": ["LoRA"]},
        ), patch.object(
            benchmark.graph, "retrieve_evidence_node", side_effect=OpenAIConnectionError(),
        ):
            self.assertFalse(benchmark.run_benchmark(self.gold, run, lambda: None, 15))
        analyze_jd.assert_not_called()
        self.assertEqual(record["completed_steps"], ["analyze_jd", "build_queries"])
        self.assertEqual(record["execution_errors"][0]["stage"], "retrieval")
        self.assertFalse(run["cases"]["applied_llm"]["attempts"])

    def test_full_cv_resume_reuses_revision_and_captures_targeted_trace(self):
        record = benchmark.new_run(self.gold, {})["cases"]["qa"]
        record["completed_steps"] = ["tailor_cv", "validate_cv:0", "revise_cv:1"]
        record["state"].update(
            revision_round=1, cv_draft={"profile": {"text": "Revised CV."}},
            pending_revalidation_claim_ids=["profile"],
        )
        validation = {"summary": {"SUPPORTED": 1, "OVERSTATED": 0, "UNSUPPORTED": 0}, "results": []}
        with patch.object(benchmark.graph, "tailor_cv_node") as tailor, patch.object(
            benchmark.graph, "revise_cv_node",
        ) as revise, patch.object(
            benchmark.graph, "validate_cv_node",
            return_value={"cv_validation": validation, "pending_revalidation_claim_ids": None},
        ):
            benchmark.run_cv(record, lambda: None, [10])
        tailor.assert_not_called()
        revise.assert_not_called()
        self.assertEqual(record["completed_steps"][-2:], ["validate_cv:1", "finalize"])
        self.assertEqual(record["validation_trace"][0]["changed_claim_ids"], ["profile"])
        self.assertEqual(record["state"]["final_status"], "grounded")

    def test_unified_report_leaves_interrupted_metrics_unavailable(self):
        run = benchmark.new_run(self.gold, {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "results/multi_jd_summary.json").write_text(
                json.dumps(summarize(self.gold, run)), encoding="utf-8",
            )
            report = build_report(root)
        self.assertEqual(report["multi_jd_live_benchmark"]["metrics"]["stage_b_completed"], 0)
        self.assertIsNone(report["multi_jd_live_benchmark"]["metrics"]["roles"][0]["stage_a"]["retrieval"])
        self.assertIn("Multi-JD live benchmark", markdown_report(report))
