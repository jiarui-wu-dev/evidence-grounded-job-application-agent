"""Verify the native server handshake and tool routing without making model calls."""

import json
import os
import queue
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Thread

from mcp.types import LATEST_PROTOCOL_VERSION


RESPONSE_TIMEOUT = 40


def verify_session(process):
    lines = queue.Queue()

    def collect():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    Thread(target=collect, daemon=True).start()

    def send(method, params=None, request_id=None):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if request_id is not None:
            message["id"] = request_id
        # Match the SDK client: UTF-8 JSON followed by a literal LF on every OS.
        process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        process.stdin.flush()

    def receive(method, request_id):
        try:
            line = lines.get(timeout=RESPONSE_TIMEOUT)
        except queue.Empty:
            raise RuntimeError(f"No MCP response to {method} within {RESPONSE_TIMEOUT}s.") from None
        if line is None:
            raise RuntimeError(f"MCP server closed stdout while waiting for {method}.")
        response = json.loads(line.decode("utf-8"))
        if response.get("id") != request_id or "result" not in response:
            raise RuntimeError(f"Unexpected MCP response to {method}.")
        return response["result"]

    send("initialize", {
        "protocolVersion": LATEST_PROTOCOL_VERSION, "capabilities": {},
        "clientInfo": {"name": "release-check", "version": "1.0"},
    }, 1)
    version = receive("initialize", 1)["protocolVersion"]
    send("notifications/initialized")
    send("tools/list", request_id=2)
    names = {tool["name"] for tool in receive("tools/list", 2)["tools"]}
    if {"get_run", "get_cv", "get_review_queue"} - names or "confirm_evidence" in names:
        raise RuntimeError("Unexpected MCP tool exposure.")
    send("tools/call", {"name": "get_run", "arguments": {"run_id": "missing"}}, 3)
    result = receive("tools/call", 3)
    if not result.get("isError") or "Run not found." not in result["content"][0]["text"]:
        raise RuntimeError("MCP tool call did not produce the expected typed error.")
    return version, len(names)


def main():
    root = Path(__file__).resolve().parents[1]
    # Record where the child is stuck before the parent times out and stops it.
    server_entry = (
        "import faulthandler, runpy; "
        f"faulthandler.dump_traceback_later({RESPONSE_TIMEOUT - 5}, repeat=True); "
        "runpy.run_module('src.mcp_server.server', run_name='__main__', alter_sys=True)"
    )
    with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile() as stderr:
        with subprocess.Popen(
            [sys.executable, "-B", "-c", server_entry], cwd=root,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            env={**os.environ, "JOB_AGENT_RUN_DB": str(Path(directory) / "smoke.sqlite")},
        ) as process:
            try:
                try:
                    version, tool_count = verify_session(process)
                finally:
                    process.stdin.close()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            except Exception as error:
                stderr.seek(0)
                details = stderr.read().decode("utf-8", errors="replace").strip()
                error.add_note(f"Child process exit code after cleanup: {process.returncode}")
                error.add_note(f"Child stderr:\n{details or '(empty)'}")
                raise
    print(f"Native MCP stdio verified at protocol {version}: {tool_count} tools.")


if __name__ == "__main__":
    main()
