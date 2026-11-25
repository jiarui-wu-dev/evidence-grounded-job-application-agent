"""Project a recorded, grounded benchmark run into the frontend demo contract."""

import json
import re
from pathlib import Path

from src.services.run_evidence import build_evidence_view


ROOT = Path(__file__).resolve().parents[1]

ALIASES_PATH = ROOT / "data/candidate_profile/demo_aliases.json"


def redact(value, aliases):
    if isinstance(value, str):
        for original, replacement in aliases.items():
            value = value.replace(original, replacement)
        return re.sub(r"[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}", "[email withheld]", value)
    if isinstance(value, list):
        return [redact(item, aliases) for item in value]
    if isinstance(value, dict):
        return {redact(key, aliases): redact(item, aliases) for key, item in value.items()}
    return value


def main():
    if not ALIASES_PATH.is_file():
        raise SystemExit("Demo export requires the local candidate alias file at data/candidate_profile/demo_aliases.json.")
    aliases = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(aliases, dict) or not aliases or any(
        not old.strip() or not isinstance(new, str) or not new.strip() or old.casefold() in new.casefold()
        for old, new in aliases.items()
    ):
        raise ValueError("Demo aliases must map private identifiers to distinct, nonempty strings.")
    source = ROOT / "evaluation/results/multi_jd_run.json"
    record = json.loads(source.read_text(encoding="utf-8"))["cases"]["applied_llm"]
    state = record["state"]
    if state["final_status"] != "grounded" or len(record["validation_trace"]) != 2:
        raise ValueError("The saved demo requires a grounded run with one recorded revision.")
    first, last = record["validation_trace"]
    before = {"cv_draft": record["initial_cv"], "cv_validation": first["validation"], "revision_round": 0}
    revised = {
        **before, "cv_draft": state["final_cv"], "revision_round": last["round"],
        "pending_revalidation_claim_ids": last["changed_claim_ids"],
    }
    view = build_evidence_view([before, revised, state], show_cv=True)
    view["education"] = []
    run_id = "saved-applied-llm"
    events = []
    revision_round = 0
    pending = None
    counts = {}
    for step in record["completed_steps"]:
        node = step.split(":", 1)[0]
        events.append({
            "created_at": None, "next_nodes": [node], "revision_round": revision_round,
            "pending_revalidation_claim_ids": pending, "validation_counts": counts, "final_status": None,
        })
        if node == "revise_cv":
            revision_round = last["round"]
            pending = last["changed_claim_ids"]
        if node == "validate_cv":
            summary = (last if revision_round else first)["validation"]["summary"]
            counts = {key: summary[key] for key in ("SUPPORTED", "OVERSTATED", "UNSUPPORTED")}
            pending = None
    events.append({**events[-1], "next_nodes": [], "final_status": "grounded"})
    data = {
        "jd_text": state["jd_text"],
        "run": {
            "run_id": run_id, "status": "completed", "final_status": "grounded", "error_code": None,
            "created_at": "", "updated_at": "",
        },
        "trace": {"run_id": run_id, "events": events},
        "evidence": view,
        "cv": state["final_cv"],
    }
    summary = json.loads((ROOT / "evaluation/results/multi_jd_summary.json").read_text(encoding="utf-8"))
    data["benchmark"] = {
        key: summary[key] for key in (
            "stage_a_completed", "stage_b_completed", "gold_requirements", "scored_requirements",
            "logical_calls", "interrupted_calls", "configured_model",
        )
    }
    data["benchmark"]["accuracy"] = summary["cross_role"]["classification_accuracy_end_to_end"]["macro_mean"]
    data["benchmark"]["roles"] = [
        {"role": row["role"], "accuracy": row["stage_a"]["classification_accuracy_end_to_end"],
         "final_status": (row.get("stage_b") or {}).get("final_status")}
        for row in summary["roles"]
    ]
    target = ROOT / "frontend/src/demo.json"
    rendered = json.dumps(redact(data, aliases), indent=2, ensure_ascii=False)
    if any(original.casefold() in rendered.casefold() for original in aliases):
        raise ValueError("A private identifier remains in the demo export.")
    target.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
