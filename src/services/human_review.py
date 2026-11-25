"""Run-scoped evidence overlays and review-only validation."""

from copy import deepcopy

from src.cv.cv_reviser import build_claim_map
from src.cv.evidence_validator import add_summary, validate_batch_result, build_validation_batches


EVIDENCE_DECISIONS = {"confirm": "verified", "reject": "rejected"}
REQUIREMENT_DECISIONS = {"keep_gap": "GAP", "ignore_for_application": "ignored"}


def decisions_by_type(decisions, target_type):
    return {row["target_id"]: row["new_status"] for row in decisions if row["target_type"] == target_type}


def effective_state(original, decisions, result=None):
    state = deepcopy(original)
    evidence_statuses = decisions_by_type(decisions, "evidence")
    rejected = {item for item, status in evidence_statuses.items() if status == "rejected"}
    requirements = decisions_by_type(decisions, "requirement")
    sources = state.get("expanded_sources", [])
    for evidence in [
        *state.get("evidence_package", {}).get("evidence", []),
        *(item for source in sources for item in source["evidence"]),
    ]:
        if evidence["evidence_id"] in evidence_statuses:
            evidence["evidence_status"] = evidence_statuses[evidence["evidence_id"]]
    for row in state.get("match_gap", {}).get("requirements", []):
        old_ids = row.get("evidence_ids", [])
        remaining = [item for item in old_ids if item not in rejected]
        decision_status = requirements.get(row["requirement_id"])
        if decision_status is not None:
            row["review_status"] = decision_status
        if decision_status == "GAP":
            row.update(status="GAP", evidence_ids=[], explanation="A human reviewer kept this requirement as a gap.")
        elif remaining != old_ids:
            row.update(
                status="PARTIAL" if remaining else "GAP", evidence_ids=remaining,
                explanation="A human reviewer rejected evidence cited for this requirement.",
            )
    if result is not None:
        state.update(cv_draft=result["draft"], final_cv=result["draft"],
                     cv_validation=result["validation"], final_status=result["final_status"])
    return state


def review_queue(original, decisions, result, *, editable):
    state = effective_state(original, decisions, result)
    recorded = {(row["target_type"], row["target_id"]): row for row in decisions}
    draft = original.get("cv_draft") or original.get("final_cv")
    cited = set()
    if draft:
        cited.update(item for claim in build_claim_map(draft).values() for item in claim["evidence_ids"])
        cited.update(item for group in draft.get("skills", [])
                     for skill in group["items"] for item in skill["evidence_ids"])
    items = []

    def add(target_type, target_id, text, current, reason, actions, required):
        decision = recorded.get((target_type, target_id))
        items.append({
            "target_type": target_type, "target_id": target_id, "text": text,
            "status": decision["new_status"] if decision else current,
            "reason": reason, "allowed_actions": list(actions) if editable and not decision else [],
            "required": required, "resolved": decision is not None,
        })

    evidence = {item["evidence_id"]: item for item in original.get("evidence_package", {}).get("evidence", [])}
    for source in original.get("expanded_sources", []):
        evidence.update({item["evidence_id"]: item for item in source["evidence"]})
    for evidence_id, item in evidence.items():
        if item.get("evidence_status") == "provisional":
            add("evidence", evidence_id, item["text"], "provisional",
                "This evidence was recorded as provisional.", EVIDENCE_DECISIONS, evidence_id in cited)
    for row in state.get("match_gap", {}).get("requirements", []):
        if row["status"] in {"GAP", "PARTIAL", None} or ("requirement", row["requirement_id"]) in recorded:
            add("requirement", row["requirement_id"], row["requirement"], row["status"] or "ambiguous",
                row.get("explanation") or "The stored mapping does not establish full support.",
                REQUIREMENT_DECISIONS, True)
    for row in state.get("cv_validation", {}).get("results", []):
        if row["status"] in {"OVERSTATED", "UNSUPPORTED"}:
            add("claim", row["claim_id"], row["claim_id"], row["status"],
                row.get("reason") or "The Validator flagged this claim.", (), False)
    return items


def reviewed_draft(original, evidence_statuses):
    draft = deepcopy(original.get("cv_draft") or original["final_cv"])
    rejected = {item for item, status in evidence_statuses.items() if status == "rejected"}

    def remove_rejected(item):
        item["evidence_ids"] = [eid for eid in item["evidence_ids"] if eid not in rejected]

    remove_rejected(draft["profile"])
    for section in ("research_experience", "experience", "selected_projects"):
        for entry in draft.get(section, []):
            for bullet in entry["bullets"]:
                remove_rejected(bullet)
    for group in draft.get("skills", []):
        for item in group["items"]:
            remove_rejected(item)
        group["items"] = [item for item in group["items"] if item["evidence_ids"]]
    if "skills" in draft:
        draft["skills"] = [group for group in draft["skills"] if group["items"]]
    return draft


def validate_review(original, decisions, previous, validator):
    statuses = decisions_by_type(decisions, "evidence")
    applied = set(previous["applied_evidence_ids"]) if previous else set()
    changed_evidence = statuses.keys() - applied
    base_draft = original.get("cv_draft") or original["final_cv"]
    changed_ids = [
        claim_id for claim_id, item in build_claim_map(base_draft).items()
        if changed_evidence.intersection(item["evidence_ids"])
    ]
    draft = reviewed_draft(original, statuses)
    updated = build_claim_map(draft)
    expanded = effective_state(original, decisions).get("expanded_sources", [])
    model_ids = [claim_id for claim_id in changed_ids if updated[claim_id]["evidence_ids"]]
    prior = previous["validation"] if previous else original["cv_validation"]
    merged = validator(draft, expanded, changed_claim_ids=model_ids, previous_validation=prior) if model_ids else prior
    validation = deepcopy(merged)
    exhausted = set(changed_ids) - set(model_ids)
    for row in validation["results"]:
        if row["claim_id"] in exhausted:
            row.update(status="UNSUPPORTED", reason="All cited evidence was rejected during review.", unsupported_phrases=[])
    claims = [claim for batch in build_validation_batches(draft, expanded) for claim in batch["claims"]]
    validate_batch_result(validation, claims)
    add_summary(validation)
    final_status = (
        "grounded" if claims and all(claim["evidence"] for claim in claims)
        and all(row["status"] == "SUPPORTED" for row in validation["results"])
        else "needs_human_review"
    )
    return {
        "draft": draft, "validation": validation, "final_status": final_status,
        "applied_evidence_ids": sorted(statuses), "changed_claim_ids": changed_ids,
        "model_claim_ids": model_ids,
    }
