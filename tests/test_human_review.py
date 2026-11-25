import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph

from src.api.main import create_app
from src.cv.evidence_validator import add_summary
from src.graph.application_graph import ApplicationState
from src.services.applications import ApplicationService, ReviewValidationFailed, RunConflict, RunNotFound


CLAIM = "selected_projects:project:bullet_1"


def row(claim_id, status, reason):
    return {"claim_id": claim_id, "status": status, "reason": reason, "unsupported_phrases": []}


class TestHumanReview(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "runs.sqlite"
        self.node_calls = 0
        self.alternate = False
        self.service = self.new_service()
        self.service.initialize()
        self.run_id = self.service.create_run("Python backend and Selenium")["run_id"]
        self.service.execute_run(self.run_id)

    def new_service(self):
        return ApplicationService(self.path, graph_factory=self.build_graph)

    def build_graph(self, *, checkpointer):
        def finish(_):
            self.node_calls += 1
            draft = {
                "profile": {"text": "Built a Python workflow.", "evidence_ids": ["verified-item"]},
                "selected_projects": [{
                    "source_id": "project", "title": "Project",
                    "bullets": [{
                        "text": "Tested workflow permissions.",
                        "evidence_ids": ["provisional-item", "verified-item"] if self.alternate else ["provisional-item"],
                    }],
                }],
                "skills": [],
            }
            validation = {"results": [
                row("profile", "SUPPORTED", "Previously grounded."),
                row(CLAIM, "UNSUPPORTED", "Evidence still provisional."),
            ]}
            add_summary(validation)
            return {
                "cv_draft": draft, "final_cv": draft, "cv_validation": validation,
                "final_status": "needs_human_review",
                "expanded_sources": [{
                    "source_id": "project", "source_title": "Project", "entity_type": "project",
                    "evidence": [
                        {"evidence_id": "verified-item", "text": "Python workflow.", "evidence_status": "verified"},
                        {"evidence_id": "provisional-item", "text": "Tested permissions.", "evidence_status": "provisional"},
                    ],
                }],
                "match_gap": {"requirements": [
                    {"requirement_id": "gap", "requirement": "Selenium", "status": "GAP", "evidence_ids": []},
                    {"requirement_id": "partial", "requirement": "Permission testing", "status": "PARTIAL",
                     "evidence_ids": ["provisional-item"]},
                    {"requirement_id": "supported", "requirement": "Python", "status": "SUPPORTED",
                     "evidence_ids": ["verified-item"]},
                ]},
            }

        graph = StateGraph(ApplicationState)
        graph.add_node("finish", finish)
        graph.add_edge(START, "finish")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=checkpointer)

    def resolve_requirements(self, run_id=None):
        run_id = run_id or self.run_id
        self.service.resolve_requirement(run_id, "gap", "keep_gap")
        return self.service.resolve_requirement(run_id, "partial", "ignore_for_application")

    def test_unresolved_queue_shows_provisional_evidence_and_blocks_completion_and_export(self):
        queue = self.service.get_review_queue(self.run_id)
        item = next(item for item in queue["items"] if item["target_id"] == "provisional-item")
        self.assertEqual(item["status"], "provisional")
        self.assertEqual(item["text"], "Tested permissions.")
        self.assertEqual(set(item["allowed_actions"]), {"confirm", "reject"})
        self.assertEqual(queue["pending_required"], 3)
        self.assertFalse(queue["can_complete"])
        for operation in (self.service.complete_review, self.service.get_cv):
            with self.assertRaises(RunConflict):
                operation(self.run_id)

    def test_confirmation_audit_persists_and_original_checkpoint_stays_provisional(self):
        view = self.service.confirm_evidence(self.run_id, "provisional-item", "I reviewed the original record.")
        decision = view["decisions"][0]
        self.assertEqual((decision["previous_status"], decision["new_status"]), ("provisional", "verified"))
        self.assertEqual(decision["run_id"], self.run_id)
        self.assertIsNotNone(decision["timestamp"])
        self.assertEqual(decision["note"], "I reviewed the original record.")
        restarted = self.new_service()
        restarted.initialize()
        self.assertEqual(restarted.get_review_queue(self.run_id)["decisions"], [decision])
        evidence = restarted.get_evidence(self.run_id)["evidence"]
        self.assertEqual(evidence[-1]["evidence_status"], "verified")
        with restarted._graph() as graph:
            state = graph.get_state(restarted._config(self.run_id)).values
        self.assertEqual(state["expanded_sources"][0]["evidence"][-1]["evidence_status"], "provisional")
        other = restarted.create_run("Other application")["run_id"]
        restarted.execute_run(other)
        self.assertEqual(restarted.get_evidence(other)["evidence"][-1]["evidence_status"], "provisional")

    def test_keep_and_ignore_record_distinct_actions_without_creating_support(self):
        view = self.resolve_requirements()
        self.assertEqual([decision["new_status"] for decision in view["decisions"]], ["GAP", "ignored"])
        rows = {r["requirement_id"]: r for r in self.service.get_evidence(self.run_id)["requirements"]}
        self.assertEqual(rows["gap"]["status"], "GAP")
        self.assertEqual(rows["partial"]["status"], "PARTIAL")
        self.assertEqual(rows["partial"]["review_status"], "ignored")

    def test_invalid_target_unsupported_action_and_duplicate_are_rejected(self):
        for target in ("missing", "verified-item"):
            with self.assertRaises(RunNotFound):
                self.service.confirm_evidence(self.run_id, target)
        with self.assertRaises(RunNotFound):
            self.service.resolve_requirement(self.run_id, "supported", "keep_gap")
        with self.assertRaises(RunConflict):
            self.service.resolve_requirement(self.run_id, "gap", "confirm")
        self.service.confirm_evidence(self.run_id, "provisional-item")
        with self.assertRaises(RunConflict):
            self.service.reject_evidence(self.run_id, "provisional-item")
        self.assertEqual(len(self.service.get_review_queue(self.run_id)["decisions"]), 1)

    def test_completion_revalidates_only_affected_claim_and_unlocks_cv(self):
        self.service.confirm_evidence(self.run_id, "provisional-item")
        self.resolve_requirements()
        with patch("src.cv.evidence_validator.get_llm", return_value=object()), patch(
            "src.cv.evidence_validator.invoke_validation_batch",
            return_value={"results": [row(CLAIM, "SUPPORTED", "Confirmed evidence supports the claim.")]},
        ) as batch:
            view = self.service.complete_review(self.run_id)
        batch.assert_called_once()
        self.assertEqual([claim["claim_id"] for claim in batch.call_args.kwargs["claims"]], [CLAIM])
        self.assertEqual(batch.call_args.kwargs["claims"][0]["evidence"][0]["evidence_status"], "verified")
        self.assertEqual(view["last_validation"]["model_claim_ids"], [CLAIM])
        self.assertEqual(view["final_status"], "grounded")
        self.assertEqual(self.node_calls, 1)
        self.assertEqual(self.service.get_cv(self.run_id)["profile"]["text"], "Built a Python workflow.")
        _, result = self.service._review_records(self.run_id)
        self.assertEqual(result["validation"]["results"][0], row("profile", "SUPPORTED", "Previously grounded."))
        self.assertEqual(self.service.get_trace(self.run_id)[-1]["final_status"], "grounded")

    def test_rejected_sole_citation_blocks_cv_without_validation_call(self):
        view = self.service.reject_evidence(self.run_id, "provisional-item")
        self.assertEqual(view["decisions"][0]["new_status"], "rejected")
        self.resolve_requirements()
        with patch("src.cv.evidence_validator.get_llm") as llm:
            view = self.service.complete_review(self.run_id)
        llm.assert_not_called()
        self.assertEqual(view["last_validation"]["changed_claim_ids"], [CLAIM])
        self.assertEqual(view["last_validation"]["model_claim_ids"], [])
        self.assertEqual(view["final_status"], "needs_human_review")
        self.assertFalse(view["can_complete"])
        _, result = self.service._review_records(self.run_id)
        self.assertEqual(result["draft"]["selected_projects"][0]["bullets"][0]["evidence_ids"], [])
        self.assertEqual(self.service.get_evidence(self.run_id)["evidence"][-1]["evidence_status"], "rejected")
        with self.assertRaises(RunConflict):
            self.service.get_cv(self.run_id)

    def test_confirmation_does_not_override_a_negative_validator_verdict(self):
        self.service.confirm_evidence(self.run_id, "provisional-item")
        self.resolve_requirements()
        with patch("src.cv.evidence_validator.get_llm", return_value=object()), patch(
            "src.cv.evidence_validator.invoke_validation_batch",
            return_value={"results": [row(CLAIM, "OVERSTATED", "Wording exceeds the confirmed evidence.")]},
        ):
            view = self.service.complete_review(self.run_id)
        self.assertEqual(view["pending_required"], 0)
        self.assertEqual(view["final_status"], "needs_human_review")
        self.assertFalse(view["can_complete"])
        remaining = next(item for item in view["items"] if item["target_id"] == CLAIM)
        self.assertEqual(remaining["status"], "OVERSTATED")
        with self.assertRaises(RunConflict):
            self.service.get_cv(self.run_id)

    def test_rejection_removes_only_rejected_citations_when_other_support_remains(self):
        self.alternate = True
        run_id = self.service.create_run("Alternative evidence")["run_id"]
        self.service.execute_run(run_id)
        self.service.reject_evidence(run_id, "provisional-item")
        self.resolve_requirements(run_id)
        with patch("src.cv.evidence_validator.get_llm", return_value=object()), patch(
            "src.cv.evidence_validator.invoke_validation_batch",
            return_value={"results": [row(CLAIM, "SUPPORTED", "Other evidence supports the claim.")]},
        ) as batch:
            self.assertEqual(self.service.complete_review(run_id)["final_status"], "grounded")
        claim = batch.call_args.kwargs["claims"][0]
        self.assertEqual([item["evidence_id"] for item in claim["evidence"]], ["verified-item"])
        self.assertEqual(self.service.get_cv(run_id)["selected_projects"][0]["bullets"][0]["evidence_ids"], ["verified-item"])

    def test_http_review_validation_failure_retains_decisions_and_hides_internal_errors(self):
        path = f"/runs/{self.run_id}/review"
        with TestClient(create_app(self.service)) as client:
            self.assertEqual(client.get(path).status_code, 200)
            self.assertEqual(client.post(path + "/evidence/provisional-item", json={"decision": "confirm"}).status_code, 200)
            duplicate = client.post(path + "/evidence/provisional-item", json={"decision": "reject"})
            self.assertEqual(duplicate.status_code, 409)
            self.assertEqual(client.post(path + "/requirements/gap", json={"decision": "keep_gap"}).status_code, 200)
            self.assertEqual(client.post(path + "/requirements/partial", json={"decision": "ignore_for_application"}).status_code, 200)
            with patch.object(self.service, "review_validator", side_effect=RuntimeError("private provider diagnostic")):
                response = client.post(path + "/complete")
            self.assertEqual(response.status_code, 502)
            self.assertNotIn("private provider diagnostic", response.text)
            self.assertEqual(client.get(path).json()["pending_required"], 0)
            self.assertEqual(len(client.get(path).json()["decisions"]), 3)
        with patch.object(self.service, "review_validator", side_effect=RuntimeError):
            with self.assertRaises(ReviewValidationFailed):
                self.service.complete_review(self.run_id)
        self.assertEqual(self.service.get_run(self.run_id)["final_status"], "needs_human_review")
