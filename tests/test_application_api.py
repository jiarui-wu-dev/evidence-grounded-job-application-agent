import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph

from src.api.main import create_app
from src.graph.application_graph import ApplicationState
from src.services.applications import ApplicationService


class TestApplicationAPI(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / "runs.sqlite"
        self.calls = []
        self.fail_once = False

    def build_graph(self, *, checkpointer):
        def analyze(state):
            self.calls.append("analyze_jd")
            return {"jd_analysis": {"job_title": state["jd_text"], "must_have_skills": ["Python"]}}

        def finalize(state):
            self.calls.append("finalize")
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("private provider diagnostic")
            review = state["jd_text"] == "Needs review"
            return {
                "revision_round": 1,
                "cv_validation": {"summary": {"SUPPORTED": 1, "UNSUPPORTED": int(review)}},
                "final_status": "needs_human_review" if review else "grounded",
                "final_cv": {"profile": {"text": "An example grounded claim.", "evidence_ids": []}},
            }

        graph = StateGraph(ApplicationState)
        graph.add_node("analyze_jd", analyze)
        graph.add_node("finalize", finalize)
        graph.add_edge(START, "analyze_jd")
        graph.add_edge("analyze_jd", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=checkpointer)

    def service(self):
        return ApplicationService(
            self.database, graph_factory=self.build_graph,
            analyzer=lambda state: {"jd_analysis": {"job_title": state["jd_text"]}},
        )

    def test_create_poll_trace_and_download_grounded_cv(self):
        with TestClient(create_app(self.service())) as client:
            created = client.post("/applications", json={"jd_text": " Private job description "})
            self.assertEqual(created.status_code, 202)
            self.assertEqual(created.json()["status"], "queued")
            run = f"/runs/{created.json()['run_id']}"
            self.assertEqual(client.get(run).json()["final_status"], "grounded")
            response = client.get(f"{run}/trace")
            self.assertEqual(response.status_code, 200)
            events = response.json()["events"]
            self.assertTrue(any(event["next_nodes"] == ["finalize"] for event in events))
            self.assertEqual(events[-1]["validation_counts"], {"SUPPORTED": 1, "UNSUPPORTED": 0})
            self.assertNotIn("Private job description", response.text)
            self.assertEqual(client.get(f"{run}/cv").json()["profile"]["text"], "An example grounded claim.")
            evidence = client.get(f"{run}/evidence")
            self.assertEqual(evidence.status_code, 200)
            self.assertEqual(evidence.json()["requirements"][0]["requirement"], "Python")
            self.assertEqual(client.post(f"{run}/resume").status_code, 409)

    def test_failed_run_resumes_from_sqlite_in_a_new_app(self):
        self.fail_once = True
        with TestClient(create_app(self.service())) as client:
            run_id = client.post("/applications", json={"jd_text": "Backend"}).json()["run_id"]
            run = f"/runs/{run_id}"
            failed = client.get(run).json()
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error_code"], "application_run_failed")
            self.assertNotIn("private provider diagnostic", str(failed))
            self.assertEqual(client.get(f"{run}/cv").status_code, 409)
        with TestClient(create_app(self.service())) as client:
            self.assertEqual(client.post(f"{run}/resume").status_code, 202)
            self.assertEqual(client.get(run).json()["status"], "completed")
        self.assertEqual(self.calls, ["analyze_jd", "finalize", "finalize"])

    def test_human_review_outcome_blocks_cv(self):
        with TestClient(create_app(self.service())) as client:
            run_id = client.post("/applications", json={"jd_text": "Needs review"}).json()["run_id"]
            run = f"/runs/{run_id}"
            self.assertEqual(client.get(run).json()["final_status"], "needs_human_review")
            self.assertEqual(client.get(f"{run}/cv").status_code, 409)
            self.assertEqual(client.get(f"{run}/evidence").json()["education"], [])
            self.assertEqual(client.get(f"{run}/trace").json()["events"][-1]["validation_counts"]["UNSUPPORTED"], 1)

    def test_analysis_validation_and_missing_runs(self):
        with TestClient(create_app(self.service())) as client:
            analyzed = client.post("/analyze-job", json={"jd_text": " Backend "})
            self.assertEqual(analyzed.json(), {"job_title": "Backend"})
            blank = client.post("/applications", json={"jd_text": "   "})
            self.assertEqual(blank.status_code, 422)
            invalid = client.post("/applications", json={"jd_text": "Backend", "max_revision_rounds": 3})
            self.assertEqual(invalid.status_code, 422)
            extra = client.post("/applications", json={"jd_text": "Backend", "extra": "private value"})
            self.assertEqual(extra.status_code, 422)
            self.assertNotIn("private value", extra.text)
            for path in ("/runs/missing", "/runs/missing/trace", "/runs/missing/cv", "/runs/missing/evidence"):
                self.assertEqual(client.get(path).status_code, 404)
            self.assertEqual(client.post("/runs/missing/resume").status_code, 404)
        self.assertEqual(self.calls, [])

    def test_restart_marks_interrupted_run_resumable_and_only_one_worker_claims_it(self):
        service = self.service()
        service.initialize()
        run_id = service.create_run("Backend")["run_id"]
        service._claim(run_id)
        restarted = self.service()
        restarted.initialize()
        self.assertEqual(restarted.get_run(run_id)["error_code"], "worker_interrupted")
        self.assertEqual(restarted.get_trace(run_id), [])
        restarted.queue_resume(run_id)
        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(restarted.execute_run, [run_id, run_id]))
        self.assertEqual(self.calls, ["analyze_jd", "finalize"])
        self.assertEqual(restarted.get_run(run_id)["status"], "completed")


if __name__ == "__main__":
    unittest.main()
