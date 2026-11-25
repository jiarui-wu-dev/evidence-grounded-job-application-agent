"""Exercise the official MCP client/server protocol with a local checkpoint."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import anyio
from langgraph.graph import END, START, StateGraph
from mcp.shared.memory import create_connected_server_and_client_session

from src.graph.application_graph import ApplicationState
from src.mcp_server.server import create_server
from src.services.applications import ApplicationService, WorkerBusy


class TestMCPServer(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = Path(directory.name) / "runs.sqlite"
        self.calls = []
        self.fail_once = False
        self.service = ApplicationService(self.db, graph_factory=self.graph_factory)
        self.server = create_server(self.service)

    def graph_factory(self, *, checkpointer):
        def analyze(state):
            self.calls.append("analyze")
            return {
                "jd_analysis": {"must_have_skills": ["Python"]},
                "expanded_sources": [{
                    "source_id": "test", "entity_type": "project",
                    "evidence": [{"evidence_id": "python", "text": "Built a Python tool.",
                                  "evidence_status": "verified"}],
                }],
            }

        def finish(state):
            self.calls.append("finalize")
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("unpublished internal detail")
            review = state["jd_text"] == "Needs review"
            return {
                "final_status": "needs_human_review" if review else "grounded",
                "final_cv": {"profile": {"text": "Built a Python tool.", "evidence_ids": ["python"]}},
                "cv_validation": {"summary": {"SUPPORTED": int(not review), "UNSUPPORTED": int(review)}},
            }

        graph = StateGraph(ApplicationState)
        graph.add_node("analyze", analyze)
        graph.add_node("finish", finish)
        graph.add_edge(START, "analyze")
        graph.add_edge("analyze", "finish")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=checkpointer)

    async def success(self, client, name, payload):
        result = await client.call_tool(name, payload)
        self.assertFalse(result.isError, result.content)
        return result.structuredContent

    def test_protocol_discovers_tools_and_reads_grounded_run(self):
        async def scenario():
            async with create_connected_server_and_client_session(
                self.server, read_timeout_seconds=timedelta(seconds=10),
            ) as client:
                listed = await client.list_tools()
                tools = {tool.name: tool for tool in listed.tools}
                self.assertEqual(set(tools), {
                    "create_application_run", "resume_run", "get_run", "get_trace",
                    "get_evidence", "get_review_queue", "get_cv",
                })
                self.assertTrue(tools["get_cv"].annotations.readOnlyHint)
                self.assertFalse(tools["create_application_run"].annotations.readOnlyHint)
                self.assertEqual(tools["create_application_run"].inputSchema["properties"]["jd_text"]["maxLength"], 30000)
                created = await self.success(client, "create_application_run", {"jd_text": " Python "})
                self.assertEqual(created["final_status"], "grounded")
                args = {"run_id": created["run_id"]}
                self.assertEqual((await self.success(client, "get_run", args))["status"], "completed")
                self.assertEqual((await self.success(client, "get_trace", args))["events"][-1]["final_status"], "grounded")
                self.assertEqual((await self.success(client, "get_evidence", args))["evidence"][0]["evidence_id"], "python")
                self.assertEqual((await self.success(client, "get_cv", args))["profile"]["evidence_ids"], ["python"])
                self.assertEqual((await self.success(client, "get_review_queue", args))["items"], [])
                rejected = await client.call_tool("create_application_run", {"jd_text": " "})
                self.assertTrue(rejected.isError)
                missing = await client.call_tool("get_run", {"run_id": "missing"})
                self.assertTrue(missing.isError)
                self.assertIn("Run not found.", missing.content[0].text)
        anyio.run(scenario)
        self.assertEqual(self.calls, ["analyze", "finalize"])

    def test_protocol_blocks_review_cv_and_human_review_mutations(self):
        async def scenario():
            async with create_connected_server_and_client_session(self.server) as client:
                created = await self.success(client, "create_application_run", {"jd_text": "Needs review"})
                args = {"run_id": created["run_id"]}
                queue = await self.success(client, "get_review_queue", args)
                self.assertEqual(queue["final_status"], "needs_human_review")
                blocked = await client.call_tool("get_cv", args)
                self.assertTrue(blocked.isError)
                self.assertIn("A grounded final CV is unavailable", blocked.content[0].text)
                self.assertNotIn("Built a Python tool", str(blocked.content))
                self.assertIsNone(blocked.structuredContent)
                mutation = await client.call_tool("confirm_evidence", {**args, "evidence_id": "python"})
                self.assertTrue(mutation.isError)
                self.assertEqual(self.service.get_review_queue(args["run_id"])["decisions"], [])
        anyio.run(scenario)

    def test_protocol_resumes_checkpoint_without_repeating_completed_node(self):
        self.fail_once = True

        async def scenario():
            async with create_connected_server_and_client_session(self.server) as client:
                created = await self.success(client, "create_application_run", {"jd_text": "Backend"})
                self.assertEqual(created["status"], "failed")
                self.assertEqual(created["error_code"], "application_run_failed")
                self.assertNotIn("unpublished", str(created))
                resumed = await self.success(client, "resume_run", {"run_id": created["run_id"]})
                self.assertEqual(resumed["run_id"], created["run_id"])
                self.assertEqual(resumed["final_status"], "grounded")
        anyio.run(scenario)
        self.assertEqual(self.calls, ["analyze", "finalize", "finalize"])

    def test_unexpected_service_error_is_sanitized(self):
        async def scenario():
            async with create_connected_server_and_client_session(self.server) as client:
                with patch.object(self.service, "get_run", side_effect=RuntimeError("private detail")):
                    failed = await client.call_tool("get_run", {"run_id": "test"})
                self.assertTrue(failed.isError)
                self.assertNotIn("private detail", str(failed.content))
        anyio.run(scenario)

    def test_sibling_startup_does_not_recover_an_active_worker(self):
        self.service.initialize()
        run_id = self.service.create_run("Backend")["run_id"]
        sibling = ApplicationService(self.db, graph_factory=self.graph_factory)
        with self.service._worker():
            self.service._claim(run_id)
            sibling.initialize()
            self.assertEqual(sibling.get_run(run_id)["status"], "running")
            with self.assertRaises(WorkerBusy), sibling._worker(timeout=0):
                self.fail("Two workers acquired the same lease.")
        sibling.initialize()
        self.assertEqual(sibling.get_run(run_id)["error_code"], "worker_interrupted")
