"""Read-only projections of checkpoint data for the run inspector."""

from src.analysis.match_gap_analyzer import build_requirements
from src.cv.cv_reviser import build_claim_map
from src.rendering.latex_renderer import STATIC_PROFILE_PATH, load_json


def verdicts(validation):
    return {
        row["claim_id"]: {"status": row["status"], "reason": row.get("reason")}
        for row in validation.get("results", [])
    }


def revision_history(states, show_text):
    entries = []
    round_entries = []
    previous = {}
    for state in states:
        pending = state.get("pending_revalidation_claim_ids")
        round_number = state.get("revision_round", 0)
        if isinstance(pending, list) and round_number > previous.get("revision_round", 0):
            prior = verdicts(previous.get("cv_validation", {}))
            draft = state.get("cv_draft")
            old_draft = previous.get("cv_draft")
            before = build_claim_map(old_draft) if show_text and old_draft else {}
            after = build_claim_map(draft) if show_text and draft else {}
            round_entries = [
                {
                    "round": round_number,
                    "claim_id": claim_id,
                    "before_text": before.get(claim_id, {}).get("text"),
                    "after_text": after.get(claim_id, {}).get("text"),
                    "before_validation": prior.get(claim_id),
                    "after_validation": None,
                }
                for claim_id in pending
            ]
            entries.extend(round_entries)
        elif previous.get("pending_revalidation_claim_ids") is not None and pending is None:
            updated = verdicts(state.get("cv_validation", {}))
            for entry in round_entries:
                entry["after_validation"] = updated.get(entry["claim_id"])
            round_entries = []
        previous = state
    return entries


def build_evidence_view(states, *, show_cv):
    state = states[-1] if states else {}
    evidence = {}

    def add(item, source_id, source_type):
        evidence[item["evidence_id"]] = {
            "evidence_id": item["evidence_id"],
            "source_id": source_id,
            "source_type": source_type,
            "text": item["text"],
            "evidence_status": item.get("evidence_status"),
        }

    for item in state.get("evidence_package", {}).get("evidence", []):
        add(item, item.get("project_id"), item.get("entity_type"))
    for source in state.get("expanded_sources", []):
        for item in source["evidence"]:
            add(item, source["source_id"], source.get("entity_type"))

    match_gap = state.get("match_gap")
    mapped = (
        match_gap["requirements"] if match_gap is not None
        else build_requirements(state.get("jd_analysis", {}))
    )
    requirements = [
        {
            **{key: row.get(key) for key in (
                "requirement_id", "requirement", "status", "explanation", "review_status",
            )},
            "evidence_ids": row.get("evidence_ids", []),
        }
        for row in mapped
    ]
    validation = verdicts(state.get("cv_validation", {}))
    positions = build_claim_map(state["final_cv"]) if show_cv and state.get("final_cv") else {}
    claims = [
        {
            "claim_id": claim_id,
            **verdict,
            **{key: positions.get(claim_id, {}).get(key) for key in ("section", "source_id", "bullet_index")},
        }
        for claim_id, verdict in validation.items()
    ]
    return {
        "requirements": requirements,
        "evidence": list(evidence.values()),
        "claims": claims,
        "revisions": revision_history(states, show_cv),
        "education": load_json(STATIC_PROFILE_PATH).get("education", [])
        if show_cv and STATIC_PROFILE_PATH.is_file() else [],
    }
