"""Run or resume the frozen multi-JD evaluation using the production graph nodes."""

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from urllib.parse import urlsplit

from src.analysis.evidence_retriever import EvidenceRetriever
from src.analysis.match_gap_analyzer import build_requirements
from src.evaluation.multi_jd_metrics import summarize
from src.graph import application_graph as graph
from src.llm.client import ChatOpenAI
from src.retrieval.profile_documents import load_profile_documents


STAGE_A = (
    "analyze_jd", "build_queries", "retrieve_evidence",
    "analyze_match_gap", "build_strategy", "expand_sources",
)
FAILURE_STAGE = {
    "analyze_jd": "jd_extraction", "build_queries": "query_building",
    "retrieve_evidence": "retrieval", "analyze_match_gap": "match_gap",
    "build_strategy": "strategy", "expand_sources": "source_composition",
    "tailor_cv": "generation", "validate_cv": "validation", "revise_cv": "revision",
    "finalize": "finalization", "finalize_with_warning": "finalization",
}


class CallBudgetReached(Exception):
    pass


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def validate_gold(gold, documents):
    evidence = {doc["evidence_id"]: doc for doc in documents}
    source_ids = {doc["project_id"] for doc in documents}
    cases = gold["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate role IDs in the gold file.")
    for case in cases:
        if not set(case["core_sources"]) <= set(case["plausible_sources"]) <= source_ids:
            raise ValueError(f"Invalid source annotation in {case['id']}.")
        rows = case["requirements"]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate gold requirement IDs in {case['id']}.")
        for row in rows:
            status = row["status"]
            ids = row["evidence_ids"]
            if status not in {None, "SUPPORTED", "PARTIAL", "GAP"} or not row["rationale"]:
                raise ValueError(f"Invalid annotation for {case['id']}/{row['id']}.")
            if not set(ids) <= evidence.keys() or bool(ids) != (status in {"SUPPORTED", "PARTIAL"}):
                raise ValueError(f"Invalid evidence references for {case['id']}/{row['id']}.")
            if status == "SUPPORTED" and not any(evidence[eid]["evidence_status"] == "verified" for eid in ids):
                raise ValueError(f"No verified evidence for {case['id']}/{row['id']}.")


def new_run(gold, provenance):
    return {
        "schema_version": 1,
        "provenance": provenance,
        "cases": {
            case["id"]: {
                "state": {"jd_text": case["jd"], "max_revision_rounds": 2},
                "completed_steps": [],
                "attempts": [],
                "calls": [],
                "execution_errors": [],
                "validation_trace": [],
            }
            for case in gold["cases"]
        },
    }


def error_details(error):
    reason = error
    while reason.__cause__ is not None:
        reason = reason.__cause__
    return {
        "error_type": type(error).__name__,
        "cause_type": type(reason).__name__,
        "http_status": getattr(error, "status_code", None),
    }


def resolved_model_from_response(response):
    """Return the provider-resolved model reported on a successful AI message."""
    for metadata in (
        getattr(response, "response_metadata", None),
        getattr(response, "additional_kwargs", None),
    ):
        if not isinstance(metadata, dict):
            continue
        for key in ("model_name", "model", "model_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def execute_step(record, phase, node, step, save, remaining_calls):
    if step in record["completed_steps"]:
        return
    state = record["state"]
    pending_ids = state.get("pending_revalidation_claim_ids")
    attempt = {"phase": phase, "node": node, "step": step, "status": "started"}
    record["attempts"].append(attempt)
    save()
    invoke = ChatOpenAI.invoke

    def track_call(client, *args, **kwargs):
        if remaining_calls[0] == 0:
            raise CallBudgetReached
        remaining_calls[0] -= 1
        call = {"phase": phase, "node": node, "step": step, "status": "started"}
        record["calls"].append(call)
        save()
        started = perf_counter()
        try:
            result = invoke(client, *args, **kwargs)
        except Exception as error:
            call.update(status="error", **error_details(error))
            raise
        else:
            call.update(
                status="success",
                resolved_model=resolved_model_from_response(result),
            )
            return result
        finally:
            call["elapsed_seconds"] = round(perf_counter() - started, 4)
            save()

    try:
        with patch.object(ChatOpenAI, "invoke", track_call):
            update = getattr(graph, f"{node}_node")(state)
        if node == "analyze_jd":
            record["extracted_requirements"] = build_requirements(update["jd_analysis"])
        if node == "tailor_cv":
            record["initial_cv"] = deepcopy(update["cv_draft"])
        if node == "validate_cv":
            record["validation_trace"].append({
                "round": state.get("revision_round", 0),
                "changed_claim_ids": pending_ids,
                "validation": deepcopy(update["cv_validation"]),
            })
        state.update(update)
    except Exception as error:
        attempt.update(status="error", **error_details(error))
        record["execution_errors"].append({
            "stage": FAILURE_STAGE[node], "step": step, "issue": "node interrupted",
            **error_details(error),
        })
        raise
    else:
        record["completed_steps"].append(step)
        attempt["status"] = "success"
    finally:
        save()


def run_cv(record, save, remaining):
    def step(node, key=None):
        execute_step(record, "B", node, key or node, save, remaining)

    step("tailor_cv")
    state = record["state"]
    while not state.get("final_status"):
        round_number = state["revision_round"]
        step("validate_cv", f"validate_cv:{round_number}")
        next_node = graph.route_after_validation(state)
        if next_node == "revise_cv":
            step(next_node, f"revise_cv:{round_number + 1}")
        else:
            step(next_node)


def provider_unavailable(error):
    details = error_details(error)
    return (
        isinstance(error, CallBudgetReached)
        or details["http_status"] in {401, 403, 429, 502, 503, 504}
        or details["error_type"] in {"APIConnectionError", "OpenAIConnectionError", "APITimeoutError"}
        or details["cause_type"] in {"ConnectError", "ConnectTimeout"}
    )


def run_benchmark(gold, run, save, call_budget):
    remaining = [call_budget]
    for phase in ("A", "B"):
        for case in gold["cases"]:
            record = run["cases"][case["id"]]
            if phase == "B" and (
                not case["full_cv"] or "expand_sources" not in record["completed_steps"]
            ):
                continue
            print(f"Stage {phase} / {case['id']}", flush=True)
            try:
                if phase == "A":
                    for node in STAGE_A:
                        execute_step(record, phase, node, node, save, remaining)
                else:
                    run_cv(record, save, remaining)
            except Exception as error:
                print(f"Interrupted {case['id']}: {error_details(error)}", flush=True)
                if provider_unavailable(error):
                    return False
    return True


def check_retrieval():
    """Probe local dependencies with an unscored query and no model API."""
    try:
        retriever = EvidenceRetriever()
        try:
            hits = retriever.retriever.search("Qdrant retrieval", top_k=2, candidate_k=4)
            return {"status": "success", "hits": len(hits)}
        finally:
            retriever.close()
    except Exception as error:
        details = error_details(error)
        if "Could not load model BAAI/bge-small-en-v1.5" in str(error):
            details["dependency"] = "BAAI/bge-small-en-v1.5"
        return {"status": "error", **details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=Path("evaluation"))
    parser.add_argument("--max-calls", type=int, default=50, help="Logical call cap for this invocation.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summarize-only", action="store_true", help="Regenerate metrics from the checkpoint.")
    mode.add_argument("--check-retrieval", action="store_true", help="Check local retrieval without LLM calls.")
    args = parser.parse_args()
    root = args.evaluation_root
    gold_path = root / "multi_jd_gold.json"
    gold = load_json(gold_path)
    documents = load_profile_documents()
    validate_gold(gold, documents)
    if args.max_calls < 1:
        parser.error("--max-calls must be positive.")
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    checkpoint = results / "multi_jd_run.json"
    gold_digest = sha256(gold_path.read_bytes())
    provenance = {
        "gold_sha256": gold_digest,
        "candidate_kb_sha256": sha256(json.dumps(documents, sort_keys=True).encode("utf-8")),
        "model": os.getenv("LLM_MODEL"),
        "provider": urlsplit(os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")).hostname,
        "sdk_automatic_retries": 0,
    }
    run = load_json(checkpoint) if checkpoint.exists() else new_run(gold, provenance)
    if run["provenance"]["gold_sha256"] != gold_digest:
        parser.error("Gold labels changed after the checkpoint was created.")
    if not args.summarize_only and run["provenance"] != provenance:
        parser.error("The Candidate KB or model/provider changed after checkpoint creation.")

    def save():
        write_json(checkpoint, run)

    def save_summary():
        write_json(results / "multi_jd_summary.json", summarize(gold, run))

    try:
        if args.check_retrieval:
            diagnostic = check_retrieval()
            run.setdefault("retrieval_preflight", []).append(diagnostic)
            save()
            print(f"Retrieval preflight: {diagnostic}", flush=True)
            return int(diagnostic["status"] != "success")
        if not args.summarize_only:
            with patch.dict(os.environ, {"LLM_MAX_RETRIES": "0"}):
                if not run_benchmark(gold, run, save, args.max_calls):
                    print("Provider or call budget interrupted the run. Resume with the same command.")
    finally:
        save_summary()
    if not args.summarize_only and not all(
        "expand_sources" in run["cases"][case["id"]]["completed_steps"]
        and (not case["full_cv"] or run["cases"][case["id"]]["state"].get("final_status"))
        for case in gold["cases"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
