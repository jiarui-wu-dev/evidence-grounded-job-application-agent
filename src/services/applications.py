"""Durable run lifecycle around the existing LangGraph application workflow."""

import json
import logging
import os
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from src.cv.evidence_validator import validate_cv_evidence
from src.graph.application_graph import analyze_jd_node, build_application_graph
from src.services.human_review import (
    EVIDENCE_DECISIONS, REQUIREMENT_DECISIONS, effective_state, review_queue, validate_review,
)
from src.services.run_evidence import build_evidence_view


logger = logging.getLogger(__name__)
FINAL_STATUSES = {"grounded", "needs_human_review"}
DEFAULT_DATABASE = Path(__file__).resolve().parents[2] / "output" / "service" / "application_runs.sqlite"


def configured_run_database():
    return Path(os.getenv("JOB_AGENT_RUN_DB", str(DEFAULT_DATABASE)))


class RunNotFound(Exception):
    pass


class RunConflict(Exception):
    pass


class WorkerBusy(RunConflict):
    pass


class ReviewValidationFailed(Exception):
    pass


class ApplicationService:
    def __init__(self, database: Path, *, graph_factory=build_application_graph, analyzer=analyze_jd_node,
                 review_validator=validate_cv_evidence):
        self.database = database
        self.graph_factory = graph_factory
        self.analyzer = analyzer
        self.review_validator = review_validator

    @contextmanager
    def _worker(self, timeout=10):
        # Use a separate SQLite lock file so graph checkpoints and reads can
        # proceed while REST and MCP serialize worker operations on this DB.
        lock_database = self.database.with_name(self.database.name + ".worker")
        with closing(sqlite3.connect(lock_database, timeout=timeout)) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as error:
                if error.sqlite_errorcode == sqlite3.SQLITE_BUSY:
                    raise WorkerBusy("The application worker is busy. Try again later.") from None
                raise
            yield

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.database, timeout=10)) as connection:
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection

    def initialize(self):
        """Create run tables and recover interrupted registrations when the worker is idle."""
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS application_runs (
                    run_id TEXT PRIMARY KEY,
                    jd_text TEXT NOT NULL,
                    max_revision_rounds INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    final_status TEXT,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS application_review_decisions (
                    review_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    target_type TEXT NOT NULL, target_id TEXT NOT NULL, decision TEXT NOT NULL,
                    previous_status TEXT NOT NULL, new_status TEXT NOT NULL,
                    timestamp TEXT NOT NULL, note TEXT,
                    UNIQUE (run_id, target_type, target_id)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS application_review_results (
                    run_id TEXT PRIMARY KEY, payload TEXT NOT NULL, timestamp TEXT NOT NULL
                )
            """)
        try:
            with self._worker(timeout=0), self._connect() as connection:
                connection.execute("""
                    UPDATE application_runs
                    SET status = 'failed', error_code = 'worker_interrupted', updated_at = ?
                    WHERE status = 'running'
                """, (self._now(),))
        except WorkerBusy:
            # A sibling adapter owns the worker and its run is still active.
            pass

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _config(run_id):
        return {"configurable": {"thread_id": run_id}}

    @contextmanager
    def _graph(self):
        with closing(sqlite3.connect(self.database, timeout=10, check_same_thread=False)) as connection:
            yield self.graph_factory(checkpointer=SqliteSaver(connection))

    def _lookup(self, connection, run_id):
        row = connection.execute(
            "SELECT * FROM application_runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        if row is None:
            raise RunNotFound
        return row

    def get_run(self, run_id):
        with self._connect() as connection:
            row = self._lookup(connection, run_id)
            return {key: row[key] for key in (
                "run_id", "status", "final_status", "error_code", "created_at", "updated_at",
            )}

    def create_run(self, jd_text, max_revision_rounds=2):
        run_id = uuid4().hex
        now = self._now()
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO application_runs
                    (run_id, jd_text, max_revision_rounds, status, created_at, updated_at)
                VALUES (?, ?, ?, 'queued', ?, ?)
            """, (run_id, jd_text, max_revision_rounds, now, now))
        return self.get_run(run_id)

    def queue_resume(self, run_id):
        with self._connect() as connection:
            updated = connection.execute("""
                UPDATE application_runs
                SET status = 'queued', error_code = NULL, updated_at = ?
                WHERE run_id = ? AND status IN ('queued', 'failed')
            """, (self._now(), run_id))
            if not updated.rowcount:
                self._lookup(connection, run_id)
                raise RunConflict("Only queued or failed runs can resume.")
        return self.get_run(run_id)

    def _claim(self, run_id):
        with self._connect() as connection:
            updated = connection.execute("""
                UPDATE application_runs SET status = 'running', updated_at = ?
                WHERE run_id = ? AND status = 'queued'
            """, (self._now(), run_id))
            return self._lookup(connection, run_id) if updated.rowcount else None

    def _finish(self, run_id, status, *, final_status=None, error_code=None):
        with self._connect() as connection:
            connection.execute("""
                UPDATE application_runs
                SET status = ?, final_status = ?, error_code = ?, updated_at = ?
                WHERE run_id = ? AND status = 'running'
            """, (status, final_status, error_code, self._now(), run_id))

    def execute_run(self, run_id):
        """Claim a queued run once and continue from any persisted checkpoint."""
        try:
            with self._worker():
                self._execute_claimed_run(run_id)
        except WorkerBusy:
            logger.info("Application worker is busy; run %s remains queued", run_id)

    def _execute_claimed_run(self, run_id):
        row = self._claim(run_id)
        if row is None:
            return
        try:
            config = self._config(run_id)
            with self._graph() as graph:
                snapshot = graph.get_state(config)
                if not snapshot.values:
                    result = graph.invoke({
                        "jd_text": row["jd_text"],
                        "max_revision_rounds": row["max_revision_rounds"],
                    }, config=config)
                elif snapshot.next:
                    result = graph.invoke(None, config=config)
                else:
                    # Recover when finalization was checkpointed before the run row was updated.
                    result = snapshot.values
            final_status = result.get("final_status")
            if final_status not in FINAL_STATUSES:
                raise RuntimeError("Application workflow has no terminal verdict.")
        except Exception as error:
            logger.warning("Application run %s failed (%s)", run_id, type(error).__name__)
            self._finish(run_id, "failed", error_code="application_run_failed")
        else:
            self._finish(run_id, "completed", final_status=final_status)

    def analyze_job(self, jd_text):
        return self.analyzer({"jd_text": jd_text})["jd_analysis"]

    def get_trace(self, run_id):
        self.get_run(run_id)
        with self._graph() as graph:
            snapshots = list(graph.get_state_history(self._config(run_id), limit=64))
        events = []
        for snapshot in reversed(snapshots):
            values = snapshot.values
            summary = values.get("cv_validation", {}).get("summary", {})
            events.append({
                "created_at": snapshot.created_at,
                "next_nodes": list(snapshot.next),
                "revision_round": values.get("revision_round"),
                "pending_revalidation_claim_ids": values.get("pending_revalidation_claim_ids"),
                "validation_counts": {
                    key: summary[key] for key in ("SUPPORTED", "OVERSTATED", "UNSUPPORTED")
                    if key in summary
                },
                "final_status": values.get("final_status"),
            })
        _, result = self._review_records(run_id)
        if result:
            summary = result["validation"]["summary"]
            events.append({
                "created_at": result["timestamp"], "next_nodes": [],
                "revision_round": snapshots[0].values.get("revision_round") if snapshots else None,
                "pending_revalidation_claim_ids": None,
                "validation_counts": {key: summary[key] for key in ("SUPPORTED", "OVERSTATED", "UNSUPPORTED")},
                "final_status": result["final_status"],
            })
        return events

    def get_cv(self, run_id):
        run = self.get_run(run_id)
        if run["status"] != "completed" or run["final_status"] != "grounded":
            raise RunConflict("A grounded final CV is unavailable for this run.")
        with self._graph() as graph:
            values = graph.get_state(self._config(run_id)).values
        decisions, result = self._review_records(run_id)
        if result:
            values = effective_state(values, decisions, result)
        cv = values.get("final_cv")
        if values.get("final_status") != "grounded" or not isinstance(cv, dict):
            raise RunConflict("A grounded final CV is unavailable for this run.")
        return cv

    def get_evidence(self, run_id):
        run = self.get_run(run_id)
        with self._graph() as graph:
            snapshots = list(graph.get_state_history(self._config(run_id), limit=64))
        states = [snapshot.values for snapshot in reversed(snapshots)]
        decisions, result = self._review_records(run_id)
        if states and (decisions or result):
            states.append(effective_state(states[-1], decisions, result))
        show_cv = bool(
            states and run["status"] == "completed" and run["final_status"] == "grounded"
            and states[-1].get("final_status") == "grounded"
        )
        return build_evidence_view(states, show_cv=show_cv)

    def _review_records(self, run_id):
        with self._connect() as connection:
            decisions = [dict(row) for row in connection.execute("""
                SELECT * FROM application_review_decisions WHERE run_id = ? ORDER BY rowid
            """, (run_id,))]
            result = connection.execute("""
                SELECT payload, timestamp FROM application_review_results WHERE run_id = ?
            """, (run_id,)).fetchone()
        return decisions, {**json.loads(result["payload"]), "timestamp": result["timestamp"]} if result else None

    def _review_context(self, run_id):
        run = self.get_run(run_id)
        if run["status"] != "completed":
            raise RunConflict("Review is available when the application run has finished.")
        with self._graph() as graph:
            original = graph.get_state(self._config(run_id)).values
        decisions, result = self._review_records(run_id)
        editable = run["final_status"] == "needs_human_review"
        items = review_queue(original, decisions, result, editable=editable) if editable or decisions or result else []
        pending = sum(item["required"] and not item["resolved"] for item in items)
        evidence_decisions = {row["target_id"] for row in decisions if row["target_type"] == "evidence"}
        new_evidence_decisions = result is None or bool(evidence_decisions - set(result["applied_evidence_ids"]))
        view = {
            "run_id": run_id, "scope": "run", "final_status": run["final_status"],
            "items": items, "decisions": decisions, "pending_required": pending,
            "can_complete": editable and pending == 0 and new_evidence_decisions and bool(original.get("cv_validation"))
            and bool(original.get("cv_draft") or original.get("final_cv")),
            "last_validation": {
                key: result[key] for key in ("timestamp", "changed_claim_ids", "model_claim_ids", "final_status")
            } if result else None,
        }
        return original, result, view

    def get_review_queue(self, run_id):
        return self._review_context(run_id)[2]

    def _record_review(self, run_id, target_type, target_id, decision, options, note):
        if decision not in options:
            raise RunConflict("Unsupported review decision.")
        if note is not None and (not isinstance(note, str) or len(note) > 2000):
            raise RunConflict("Review notes must be at most 2,000 characters.")
        with self._worker():
            _, _, view = self._review_context(run_id)
            if view["final_status"] != "needs_human_review":
                raise RunConflict("This run is closed for review.")
            record = next((item for item in view["items"]
                           if item["target_type"] == target_type and item["target_id"] == target_id), None)
            if record is None:
                raise RunNotFound
            if decision not in record["allowed_actions"]:
                raise RunConflict("This target already has a recorded decision.")
            audit = {
                "review_id": uuid4().hex, "run_id": run_id, "target_type": target_type,
                "target_id": target_id, "decision": decision, "previous_status": record["status"],
                "new_status": options[decision], "timestamp": self._now(), "note": note,
            }
            try:
                with self._connect() as connection:
                    connection.execute("""
                        INSERT INTO application_review_decisions
                        (review_id, run_id, target_type, target_id, decision, previous_status, new_status, timestamp, note)
                        VALUES (:review_id, :run_id, :target_type, :target_id, :decision,
                                :previous_status, :new_status, :timestamp, :note)
                    """, audit)
            except sqlite3.IntegrityError as error:
                raise RunConflict("This target already has a recorded decision.") from error
            return self.get_review_queue(run_id)

    def confirm_evidence(self, run_id, evidence_id, note=None):
        return self._record_review(run_id, "evidence", evidence_id, "confirm", EVIDENCE_DECISIONS, note)

    def reject_evidence(self, run_id, evidence_id, note=None):
        return self._record_review(run_id, "evidence", evidence_id, "reject", EVIDENCE_DECISIONS, note)

    def resolve_requirement(self, run_id, requirement_id, decision, note=None):
        return self._record_review(run_id, "requirement", requirement_id, decision, REQUIREMENT_DECISIONS, note)

    def complete_review(self, run_id):
        """Persist only affected verdicts. The graph checkpoint stays immutable."""
        with self._worker():
            original, previous, view = self._review_context(run_id)
            if not view["can_complete"]:
                raise RunConflict("Resolve required review items before completing review.")
            try:
                result = validate_review(original, view["decisions"], previous, self.review_validator)
            except Exception as error:
                logger.warning("Review validation %s failed (%s)", run_id, type(error).__name__)
                raise ReviewValidationFailed from error
            now = self._now()
            with self._connect() as connection:
                connection.execute("""
                    INSERT INTO application_review_results (run_id, payload, timestamp) VALUES (?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET payload = excluded.payload, timestamp = excluded.timestamp
                """, (run_id, json.dumps(result), now))
                connection.execute("""
                    UPDATE application_runs SET final_status = ?, updated_at = ? WHERE run_id = ?
                """, (result["final_status"], now, run_id))
            return self.get_review_queue(run_id)
