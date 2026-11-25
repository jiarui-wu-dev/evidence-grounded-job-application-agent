import json
import unittest
from pathlib import Path

from src.api.evidence_schema import EvidenceResponse
from src.services.run_evidence import build_evidence_view


class TestRunEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "evaluation/results/multi_jd_run.json"
        if not path.exists():
            cls.states = None
            return
        record = json.loads(path.read_text(encoding="utf-8"))["cases"]["applied_llm"]
        first, last = record["validation_trace"]
        cls.before = {"cv_draft": record["initial_cv"], "cv_validation": first["validation"], "revision_round": 0}
        cls.revised = {
            **cls.before, "cv_draft": record["state"]["final_cv"], "revision_round": last["round"],
            "pending_revalidation_claim_ids": last["changed_claim_ids"],
        }
        cls.states = [cls.before, cls.revised, record["state"]]

    def test_projects_match_gap_provenance_and_latest_verdicts(self):
        if self.states is None:
            self.skipTest("Private historical benchmark fixture unavailable.")
        view = EvidenceResponse.model_validate(build_evidence_view(self.states, show_cv=True))
        requirements = self.states[-1]["match_gap"]["requirements"]
        self.assertEqual([row.status for row in view.requirements], [row["status"] for row in requirements])
        evidence = {row.evidence_id: row for row in view.evidence}
        cited = view.requirements[0].evidence_ids[0]
        self.assertTrue(evidence[cited].source_id)
        self.assertTrue(evidence[cited].text)
        self.assertTrue(any(row.evidence_status == "provisional" for row in view.evidence))
        self.assertEqual(next(row for row in view.claims if row.claim_id == "profile").section, "profile")
        self.assertTrue(view.education)
        self.assertEqual([row.claim_id for row in view.revisions], self.revised["pending_revalidation_claim_ids"])
        self.assertTrue(all(row.before_text != row.after_text for row in view.revisions))
        self.assertTrue(all(row.before_validation.status != "SUPPORTED" for row in view.revisions))
        self.assertTrue(all(row.after_validation.status == "SUPPORTED" for row in view.revisions))

    def test_review_keeps_diagnostics_and_withholds_revision_and_education_text(self):
        if self.states is None:
            self.skipTest("Private historical benchmark fixture unavailable.")
        view = EvidenceResponse.model_validate(build_evidence_view(self.states, show_cv=False))
        self.assertTrue(view.requirements)
        self.assertTrue(view.evidence)
        self.assertFalse(view.education)
        self.assertTrue(view.revisions)
        for row in view.revisions:
            self.assertIsNone(row.before_text)
            self.assertIsNone(row.after_text)
            self.assertIsNotNone(row.before_validation)

    def test_extracted_requirements_before_match_have_no_inferred_status(self):
        view = EvidenceResponse.model_validate(build_evidence_view([
            {"jd_analysis": {"must_have_skills": ["Python"]}},
        ], show_cv=False))
        self.assertEqual(view.requirements[0].requirement, "Python")
        self.assertIsNone(view.requirements[0].status)
        self.assertEqual(view.requirements[0].evidence_ids, [])
