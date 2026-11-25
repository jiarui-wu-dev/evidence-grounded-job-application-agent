"""Metrics for frozen multi-JD annotations and saved application state."""

import re
from collections import Counter
from statistics import mean

from src.evaluation.evaluate_retrieval import evaluate_results
from src.evaluation.unified_report import cv_checks


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def align_requirement(gold, rows):
    aliases = {normalize(value) for value in [gold["requirement"], *gold["aliases"]]}
    matches = [
        row for row in rows
        if any(f" {alias} " in f" {normalize(row['requirement'])} " for alias in aliases)
    ]
    exact = [row for row in matches if normalize(row["requirement"]) in aliases]
    return exact or matches


def source_metrics(items, case):
    if items is None:
        return None
    actual = {item["source_id"] for item in items}
    core = set(case["core_sources"])
    outside = actual - set(case["plausible_sources"])
    return {
        "selected": sorted(actual),
        "missing_core": sorted(core - actual),
        "outside_plausible": sorted(outside),
        "core_coverage": ratio(len(actual & core), len(core)),
        "plausible_precision": ratio(len(actual - outside), len(actual)),
    }


def stage_a(case, record):
    state = record["state"]
    failures = []
    package = state.get("evidence_package")
    retrieved = set()
    per_query = set()
    retrieval = None
    if package is not None:
        retrieved = {item["evidence_id"] for item in package["evidence"]}
        per_query = {
            item["evidence_id"]
            for query in package["query_results"] for item in query["results"]
        }
        relevance = {eid: 1 for row in case["requirements"] for eid in row["evidence_ids"]}
        if relevance:
            retrieval = evaluate_results(package["evidence"], relevance)
            retrieval["gold_evidence_count"] = len(relevance)
    analysis = record.get("extracted_requirements")
    match_gap = state.get("match_gap")
    predictions = {
        row["requirement_id"]: row for row in (match_gap or {}).get("requirements", [])
    }
    rows = []
    for gold in case["requirements"]:
        expected = gold["status"]
        options = None if analysis is None else align_requirement(gold, analysis)
        aligned = options[0] if options and len(options) == 1 else None
        predicted = predictions.get(aligned["requirement_id"]) if aligned else None
        acceptable = set(gold["evidence_ids"])
        cited = set(predicted["evidence_ids"]) if predicted else set()
        status_correct = predicted["status"] == expected if expected and predicted else None
        if predicted is None or expected is None:
            attribution = None
        elif expected == "GAP":
            attribution = not cited
        else:
            attribution = bool(cited) and cited <= acceptable
        if expected and options is not None and aligned is None:
            failures.append({
                "stage": "jd_extraction", "requirement_id": gold["id"],
                "issue": "ambiguous alignment" if options else "requirement absent",
            })
        if acceptable and package is not None and not (retrieved & acceptable):
            failures.append({
                "stage": "evidence_aggregation" if per_query & acceptable else "retrieval",
                "requirement_id": gold["id"], "issue": "no gold evidence in final package",
            })
        if predicted is not None and (status_correct is False or attribution is False):
            if acceptable and not (retrieved & acceptable):
                origin = "evidence_aggregation" if per_query & acceptable else "retrieval"
            else:
                origin = "match_gap"
            failures.append({
                "stage": origin, "requirement_id": gold["id"],
                "issue": "status or citation differs from gold",
            })
        rows.append({
            "id": gold["id"],
            "expected_status": expected,
            "expected_evidence_ids": gold["evidence_ids"],
            "matches": None if options is None else [item["requirement_id"] for item in options],
            "predicted_status": predicted["status"] if predicted else None,
            "cited_evidence_ids": sorted(cited) if predicted else None,
            "status_correct": status_correct,
            "attribution_correct": attribution,
        })
    scored = [row for row in rows if row["expected_status"] is not None]
    aligned_rows = [row for row in scored if row["matches"] and len(row["matches"]) == 1]
    compared = [row for row in scored if row["status_correct"] is not None]
    gaps = [row for row in scored if row["expected_status"] == "GAP"]
    mapped_gaps = [row for row in compared if row["expected_status"] == "GAP"]
    correct = sum(row["status_correct"] for row in compared)
    strategy = source_metrics(state.get("application_strategy", {}).get("priority_sources"), case)
    composition = source_metrics(state.get("cv_composition", {}).get("sources"), case)
    for stage, selection in (("strategy", strategy), ("source_composition", composition)):
        if selection and (selection["missing_core"] or selection["outside_plausible"]):
            failures.append({
                "stage": stage, "issue": "source selection diverges from gold",
                "missing_core": selection["missing_core"],
                "outside_plausible": selection["outside_plausible"],
            })
    return {
        "gold_requirements": len(rows),
        "scored_requirements": len(scored),
        "aligned_requirements": len(aligned_rows) if analysis is not None else None,
        "compared_requirements": len(compared) if match_gap is not None else None,
        "classification_accuracy_aligned": ratio(correct, len(compared)),
        "classification_accuracy_end_to_end": (
            ratio(correct, len(scored)) if match_gap is not None else None
        ),
        "attribution_accuracy_aligned": ratio(
            sum(row["attribution_correct"] for row in compared), len(compared),
        ),
        "unsupported_requirements": len(gaps),
        "unsupported_correct": sum(row["status_correct"] for row in mapped_gaps) if match_gap is not None else None,
        "unsupported_accuracy_aligned": ratio(sum(row["status_correct"] for row in mapped_gaps), len(mapped_gaps)),
        "retrieval": retrieval,
        "strategy_sources": strategy,
        "composition_sources": composition,
        "requirements": rows,
        "failures": failures,
    }


def stage_b(record):
    state = record["state"]
    trace = record["validation_trace"]
    final_citations, readability = cv_checks(state.get("final_cv"), state.get("cv_composition"))
    initial_citations, _ = cv_checks(record.get("initial_cv"), state.get("cv_composition"))
    initial = trace[0]["validation"]["summary"] if trace else None
    final = state.get("cv_validation", {}).get("summary") if state.get("final_status") else None
    checked = preserved = targeted = 0
    changed_ids = set()
    for before, after in zip(trace, trace[1:]):
        changed = set(after["changed_claim_ids"])
        changed_ids.update(changed)
        targeted += len(changed)
        old = {row["claim_id"]: row for row in before["validation"]["results"]}
        new = {row["claim_id"]: row for row in after["validation"]["results"]}
        for claim_id in old.keys() - changed:
            checked += 1
            preserved += old[claim_id] == new.get(claim_id)
    return {
        "generated_claim_count": initial_citations["claims"] if initial_citations is not None else None,
        "initial_validation": initial,
        "initial_citation_coverage": initial_citations,
        "final_citation_coverage": final_citations,
        "final_validation": final,
        "revision_rounds": state.get("revision_round"),
        "changed_claim_ids": sorted(changed_ids) if len(trace) > 1 else None,
        "targeted_revalidation_rounds": len(trace) - 1 if trace else None,
        "targeted_claim_evaluations": targeted if trace else None,
        "untouched_verdicts_checked": checked if trace else None,
        "untouched_verdicts_preserved": preserved if trace else None,
        "untouched_verdict_preservation_rate": ratio(preserved, checked),
        "final_status": state.get("final_status"),
        "readability": readability,
        "independent_unsupported_claim_leakage": None,
    }


def summarize(gold, run):
    roles = []
    for case in gold["cases"]:
        record = run["cases"][case["id"]]
        a_metrics = stage_a(case, record)
        failures = a_metrics.pop("failures") + record["execution_errors"]
        b_metrics = stage_b(record) if case["full_cv"] else None
        if b_metrics and b_metrics["final_validation"]:
            final = b_metrics["final_validation"]
            if final["OVERSTATED"] or final["UNSUPPORTED"]:
                failures.append({
                    "stage": "validation", "issue": "final Validator flagged claims",
                    "overstated": final["OVERSTATED"], "unsupported": final["UNSUPPORTED"],
                })
        roles.append({
            "id": case["id"], "role": case["role"], "stage_a": a_metrics,
            "stage_a_complete": "expand_sources" in record["completed_steps"],
            "stage_b": b_metrics, "failures": failures,
        })

    def across_roles(key):
        values = {
            role["id"]: role["stage_a"][key] for role in roles if role["stage_a"][key] is not None
        }
        return {"roles_measured": len(values), "macro_mean": mean(values.values()) if values else None,
                "min": min(values.values(), default=None), "max": max(values.values(), default=None)}

    calls = [call for record in run["cases"].values() for call in record["calls"]]
    successful_calls = [call for call in calls if call["status"] == "success"]
    resolved_models = Counter(
        call["resolved_model"] for call in successful_calls if call.get("resolved_model")
    )
    return {
        "schema_version": 1,
        "provenance": run["provenance"],
        "retrieval_path": "live graph nodes: BM25/dense union, Cross-Encoder and confidence aggregation",
        "gold_requirements": sum(item["stage_a"]["gold_requirements"] for item in roles),
        "scored_requirements": sum(item["stage_a"]["scored_requirements"] for item in roles),
        "stage_a_completed": sum(item["stage_a_complete"] for item in roles),
        "stage_b_completed": sum(bool(item["stage_b"] and item["stage_b"]["final_status"]) for item in roles),
        "cross_role": {
            key: across_roles(key) for key in (
                "classification_accuracy_end_to_end", "attribution_accuracy_aligned",
                "unsupported_accuracy_aligned",
            )
        },
        "logical_calls": len(calls),
        "logical_calls_by_phase": dict(Counter(call["phase"] for call in calls)),
        "interrupted_calls": sum(call["status"] != "success" for call in calls),
        "configured_model": run["provenance"].get("model"),
        "resolved_model_distribution": dict(sorted(resolved_models.items())),
        "successful_calls_without_resolved_model": sum(
            not call.get("resolved_model") for call in successful_calls
        ),
        "sdk_automatic_retries": 0,
        "repeated_node_attempts": sum(
            count - 1 for record in run["cases"].values()
            for count in Counter(item["step"] for item in record["attempts"]).values()
        ),
        "rate_limited_calls": sum(call.get("http_status") == 429 for call in calls),
        "retrieval_preflight": run.get("retrieval_preflight", []),
        "failures_by_stage": dict(Counter(item["stage"] for role in roles for item in role["failures"])),
        "roles": roles,
    }
