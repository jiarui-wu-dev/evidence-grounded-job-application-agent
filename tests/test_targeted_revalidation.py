import json
import sqlite3
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.sqlite import SqliteSaver

from src.cv.evidence_validator import validate_cv_evidence
from src.graph.application_graph import (
    build_revision_resume_graph,
    revise_cv_node,
    route_after_validation,
    validate_cv_node,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation/fixtures/ai_engineer_composition_regression_result.json"
)


def supported_row(claim_id):
    return {
        "claim_id": claim_id,
        "status": "SUPPORTED",
        "reason": "Stubbed supported verdict.",
        "unsupported_phrases": [],
    }


class TestTargetedRevalidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = FIXTURE_PATH if FIXTURE_PATH.exists() else Path(__file__).with_name("fixtures") / "synthetic_targeted_revalidation.json"
        cls.fixture = json.loads(path.read_text(encoding="utf-8"))
        initial_cv = cls.fixture["initial_cv"]
        cls.research_label = f"research_experience:{initial_cv['research_experience'][0]['source_id']}"
        cls.experience_label = f"experience:{initial_cv['experience'][0]['source_id']}"
        cls.project_labels = [
            f"selected_projects:{entry['source_id']}" for entry in initial_cv["selected_projects"]
        ]
        cls.experience_claims = [f"{cls.experience_label}:bullet_1", f"{cls.experience_label}:bullet_2"]
        cls.unchanged_claim = f"{cls.project_labels[1]}:bullet_2"

    def setUp(self):
        self.state = {
            "cv_draft": deepcopy(self.fixture["initial_cv"]),
            "expanded_sources": deepcopy(self.fixture["expanded_sources"]),
            "cv_validation": deepcopy(self.fixture["initial_validation"]),
            "revision_round": 0,
            "max_revision_rounds": 2,
        }
        self.requested_batches = []
        client = patch("src.cv.evidence_validator.get_llm", return_value=object())
        self.client = client.start()
        self.addCleanup(client.stop)
        validator = patch(
            "src.cv.evidence_validator.invoke_validation_batch",
            side_effect=self.validate_batch,
        )
        self.validator = validator.start()
        self.addCleanup(validator.stop)
        reviser = patch(
            "src.graph.application_graph.revise_cv_draft",
            return_value=deepcopy(self.fixture["final_cv"]),
        )
        self.reviser = reviser.start()
        self.addCleanup(reviser.stop)

    def validate_batch(self, *, llm, claims, label, max_attempts):
        claim_ids = [claim["claim_id"] for claim in claims]
        self.requested_batches.append((label, claim_ids))
        return {"results": [supported_row(claim_id) for claim_id in reversed(claim_ids)]}

    def test_saved_project_verdict_and_all_untouched_rows_survive(self):
        live_regression = {
            row["claim_id"]: row for row in self.fixture["final_validation"]["results"]
        }
        self.assertEqual(live_regression[self.unchanged_claim]["status"], "OVERSTATED")

        revised_state = {**self.state, **revise_cv_node(self.state)}
        self.assertEqual(revised_state["pending_revalidation_claim_ids"], self.experience_claims)
        update = validate_cv_node(revised_state)
        self.assertEqual(self.requested_batches, [(self.experience_label, self.experience_claims)])
        self.assertIsNone(update["pending_revalidation_claim_ids"])

        previous = self.state["cv_validation"]["results"]
        merged = update["cv_validation"]["results"]
        self.assertEqual([row["claim_id"] for row in merged], [row["claim_id"] for row in previous])
        for old, new in zip(previous, merged):
            if old["claim_id"] not in self.experience_claims:
                self.assertEqual(new, old)
            else:
                self.assertEqual(new, supported_row(old["claim_id"]))
        self.assertEqual(update["cv_validation"]["summary"], {
            "SUPPORTED": 12, "OVERSTATED": 0, "UNSUPPORTED": 0, "total": 12,
            "grounded_rate": 1.0, "problem_rate": 0.0,
        })
        self.assertEqual(route_after_validation({**revised_state, **update}), "finalize")

    def test_changed_profile_and_one_modified_bullet_are_revalidated(self):
        revised = deepcopy(self.state["cv_draft"])
        revised["profile"]["text"] += " Revised profile wording."
        revised["experience"][0]["bullets"][0]["text"] += " Revised bullet wording."
        self.reviser.return_value = revised

        revised_state = {**self.state, **revise_cv_node(self.state)}
        result = validate_cv_node(revised_state)["cv_validation"]
        self.assertEqual(self.requested_batches, [
            ("profile", ["profile"]),
            (self.experience_label, [self.experience_claims[0]]),
        ])
        old_rows = {row["claim_id"]: row for row in self.state["cv_validation"]["results"]}
        new_rows = {row["claim_id"]: row for row in result["results"]}
        self.assertEqual(new_rows[self.experience_claims[1]], old_rows[self.experience_claims[1]])
        self.assertEqual(result["summary"]["UNSUPPORTED"], 1)

    def test_noop_revalidates_nothing_and_recomputes_summary(self):
        self.reviser.return_value = deepcopy(self.state["cv_draft"])
        self.state["cv_validation"]["summary"]["UNSUPPORTED"] = 0

        revised_state = {**self.state, **revise_cv_node(self.state)}
        self.assertEqual(revised_state["pending_revalidation_claim_ids"], [])
        update = validate_cv_node(revised_state)
        self.client.assert_not_called()
        self.validator.assert_not_called()
        self.assertEqual(update["cv_validation"], self.fixture["initial_validation"])
        self.assertEqual(route_after_validation({**revised_state, **update}), "revise_cv")

    def test_noop_at_round_limit_returns_human_review_warning(self):
        self.reviser.return_value = deepcopy(self.state["cv_draft"])
        result = build_revision_resume_graph().invoke(self.state)
        self.assertEqual(result["revision_round"], 2)
        self.assertEqual(self.reviser.call_count, 2)
        self.assertEqual(result["final_status"], "needs_human_review")
        self.assertEqual(result["final_cv"], self.state["cv_draft"])
        self.assertEqual(len(result["warnings"]), 1)
        for claim_id in self.experience_claims:
            self.assertIn(claim_id, result["warnings"][0])
        self.client.assert_not_called()
        self.validator.assert_not_called()

    def test_checkpoint_resume_uses_persisted_changed_ids(self):
        config = {"configurable": {"thread_id": "targeted-revalidation-test"}}
        with SqliteSaver.from_conn_string(":memory:") as checkpointer:
            graph = build_revision_resume_graph(checkpointer=checkpointer)
            graph.invoke(self.state, config, interrupt_before=["validate_cv"])
            snapshot = graph.get_state(config)
            self.assertEqual(snapshot.next, ("validate_cv",))
            self.assertEqual(snapshot.values["pending_revalidation_claim_ids"], self.experience_claims)
            self.validator.assert_not_called()

            resumed = build_revision_resume_graph(checkpointer=checkpointer)
            result = resumed.invoke(None, config)
        self.assertEqual(result["final_status"], "grounded")
        self.assertEqual(self.requested_batches, [(self.experience_label, self.experience_claims)])
        self.assertEqual(self.reviser.call_count, 1)

    def test_initial_and_older_checkpoint_states_use_full_validation(self):
        expected_labels = [
            "profile",
            self.research_label,
            self.experience_label,
            *self.project_labels,
        ]
        initial_state = {
            "cv_draft": self.state["cv_draft"],
            "expanded_sources": self.state["expanded_sources"],
        }
        legacy_state = {**self.state, "cv_draft": self.fixture["final_cv"], "revision_round": 1}
        for state in (initial_state, legacy_state):
            with self.subTest(revision_round=state.get("revision_round", 0)):
                self.requested_batches.clear()
                result = validate_cv_node(state)
                self.assertEqual([label for label, _ in self.requested_batches], expected_labels)
                self.assertEqual(result["cv_validation"]["summary"]["SUPPORTED"], 12)

    def test_incomplete_or_duplicate_previous_results_are_rejected_before_calls(self):
        rows = self.state["cv_validation"]["results"]
        for invalid_rows in (rows[:-1], rows + rows[:1], rows + [supported_row("stale")]):
            with self.subTest(row_count=len(invalid_rows)), self.assertRaises(ValueError):
                validate_cv_evidence(
                    self.state["cv_draft"], self.state["expanded_sources"],
                    changed_claim_ids=self.experience_claims,
                    previous_validation={"results": invalid_rows},
                )
        self.client.assert_not_called()

    def test_incomplete_or_malformed_revalidation_results_are_rejected(self):
        first, second = map(supported_row, self.experience_claims)
        invalid_results = [
            [first],
            [first, first],
            [first, supported_row(self.unchanged_claim)],
            [first, {**second, "claim_id": []}],
            [first, {**second, "status": "UNKNOWN"}],
            [first, {**second, "status": []}],
            [first, {**second, "unsupported_phrases": ["unexpected phrase"]}],
        ]
        for rows in invalid_results:
            self.validator.side_effect = None
            self.validator.return_value = {"results": rows}
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_cv_node({**self.state, **revise_cv_node(self.state)})
        self.assertEqual(self.state["cv_validation"], self.fixture["initial_validation"])


if __name__ == "__main__":
    unittest.main()
