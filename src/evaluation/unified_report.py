"""Summarize saved evaluation artifacts without running models or retrieval."""

import argparse
import json
from collections import Counter
from pathlib import Path


RETRIEVAL_BENCHMARKS = {
    "baseline_v1": "Initial BM25, dense and RRF benchmark",
    "challenge_v1": "Challenge BM25, dense and RRF benchmark",
    "challenge_v2_weighted_rrf": "Challenge weighted RRF benchmark",
    "challenge_v3_reranked": "Per-query Cross-Encoder benchmark",
    "holdout_v1_weighted_rrf": "Holdout weighted RRF benchmark",
}
RETRIEVAL_METRICS = ("hit@1", "hit@3", "recall@5", "mrr", "ndcg@5")
MATCH_GAP_METRICS = ("status_accuracy", "attribution_accuracy", "fully_correct_rate")
CV_SECTIONS = ("research_experience", "experience", "selected_projects")
WORKFLOW_FILES = {
    "full_graph": "fixtures/ai_engineer_langgraph_full_result.json",
    "revision_graph": "fixtures/ai_engineer_langgraph_result.json",
    "composition_before_targeted_revalidation": (
        "fixtures/ai_engineer_composition_regression_result.json"
    ),
    "composition_with_targeted_revalidation": (
        "results/ai_engineer_composition_targeted_revalidation.json"
    ),
}
NOTES = {
    "retrieval": (
        "Benchmark scores use labeled single queries. Live graph retrieval pools BM25 "
        "and dense candidates, reranks them with a Cross-Encoder, then aggregates "
        "results across JD queries using confidence weights. The saved artifacts "
        "lack end-to-end relevance labels for those evidence packages. The reranked "
        "benchmark measures the per-query component. RRF results are benchmark-only."
    ),
    "grounding": (
        "CV grounding scores are saved Validator judgments for Profile and source "
        "bullets. Citation coverage checks ID presence only. Independent final "
        "unsupported-claim leakage and claim-level attribution labels are unavailable."
    ),
    "workflow": (
        "Saved artifacts report rounds and terminal status when present. LLM calls, "
        "retries, elapsed time, and fresh versus resumed execution are unrecorded "
        "in the supplied results. Final snapshots alone cannot show a verdict delta."
    ),
    "readability": (
        "Word counts use whitespace splitting, as in CV Tailor. Exact repetition "
        "ignores case and whitespace. Source limits are reported only from saved "
        "composition plans. These checks do not assess writing or semantic quality."
    ),
    "coverage": (
        "Historical CV snapshots concern one AI Engineer JD and may describe related runs. "
        "The multi-JD evaluation is reported separately from these snapshots and retrieval datasets."
    ),
}


def metrics_from(values, keys):
    return {key: (values or {}).get(key) for key in keys}


def retrieval_summary(read):
    benchmarks = []
    for name, description in RETRIEVAL_BENCHMARKS.items():
        source = f"results/{name}.json"
        saved = read(source)
        data = saved or {}
        overall = data.get("overall", {})
        systems = data.get("systems", {})
        if "system" in data:
            systems = {"Reranked": data["system"]}
            overall = {"Reranked": overall}
        benchmarks.append({
            "id": name,
            "source": source,
            "available": saved is not None,
            "description": description,
            "evaluation_file": data.get("evaluation_file"),
            "query_count": data.get("query_count"),
            "corpus_size": data.get("corpus_size"),
            "hybrid_config": data.get("hybrid_config"),
            "reranker_model": data.get("reranker_model"),
            "candidate_k": data.get("candidate_k"),
            "systems": {
                system: {
                    "description": systems.get(system),
                    "metrics": metrics_from(overall.get(system), RETRIEVAL_METRICS),
                }
                for system in dict.fromkeys([*systems, *overall])
            },
        })
    return {
        "live_graph": {
            "description": NOTES["retrieval"],
            "metrics": metrics_from(None, RETRIEVAL_METRICS),
        },
        "benchmarks": benchmarks,
    }


def cv_checks(cv, composition):
    if cv is None:
        return None, None

    profile = cv.get("profile") or {}
    section_counts = {section: 0 for section in CV_SECTIONS}
    claims = [profile] if profile.get("text") else []
    word_counts = []
    repetitions = Counter()
    source_checks = []
    limits = {item["source_id"]: item for item in (composition or {}).get("sources", [])}
    for section in CV_SECTIONS:
        for source in cv.get(section, []):
            bullets = source.get("bullets", [])
            lengths = [len(bullet["text"].split()) for bullet in bullets]
            word_counts.extend(lengths)
            claims.extend(bullets)
            repetitions.update(" ".join(bullet["text"].casefold().split()) for bullet in bullets)
            section_counts[section] += len(bullets)
            limit = limits.get(source["source_id"], {})
            max_bullets = limit.get("max_bullets")
            max_words = limit.get("max_words_per_bullet")
            source_checks.append({
                "source_id": source["source_id"],
                "section": section,
                "bullet_count": len(bullets),
                "max_bullets": max_bullets,
                "bullet_count_over_limit": None if max_bullets is None else len(bullets) > max_bullets,
                "longest_bullet_words": max(lengths, default=None),
                "max_words_per_bullet": max_words,
                "bullets_over_word_limit": (
                    None if max_words is None else sum(length > max_words for length in lengths)
                ),
            })

    cited = sum(bool(claim.get("evidence_ids")) for claim in claims)
    return {
        "claims": len(claims),
        "claims_with_evidence_ids": cited,
        "coverage": cited / len(claims) if claims else None,
    }, {
        "profile_words": len(profile["text"].split()) if profile.get("text") else None,
        "bullets_by_section": section_counts,
        "bullet_count": len(word_counts),
        "mean_bullet_words": sum(word_counts) / len(word_counts) if word_counts else None,
        "longest_bullet_words": max(word_counts, default=None),
        "duplicate_bullets": sum(count - 1 for count in repetitions.values()),
        "sources": source_checks,
    }


def validation_delta(before, after):
    if "results" not in (before or {}) or "results" not in (after or {}):
        return None
    old, new = (
        {row["claim_id"]: row["status"] for row in validation["results"]}
        for validation in (before, after)
    )
    return {
        "changed_verdicts": sum(old[key] != new[key] for key in old.keys() & new.keys()),
        "added_claim_ids": len(new.keys() - old.keys()),
        "removed_claim_ids": len(old.keys() - new.keys()),
    }


def build_report(root):
    missing = []

    def read(relative_path):
        try:
            with (root / relative_path).open(encoding="utf-8") as stream:
                return json.load(stream)
        except FileNotFoundError:
            missing.append(relative_path)
            return None

    retrieval = retrieval_summary(read)
    match_gap_source = "results/match_gap_v1.json"
    match_gap = read(match_gap_source) or {}
    grounding = {
        "match_gap_benchmark": {
            "source": match_gap_source,
            "case_count": match_gap.get("case_count"),
            "metrics": metrics_from(match_gap.get("metrics"), MATCH_GAP_METRICS),
            "confusion_matrix": match_gap.get("confusion_matrix"),
        },
        "cv_validations": [],
        "independently_verified_unsupported_claim_leakage": None,
    }
    workflow = []
    readability = []

    def add_snapshot(run, stage, cv, validation, composition, sources):
        citations, quality = cv_checks(cv, composition)
        grounding["cv_validations"].append({
            "run": run,
            "stage": stage,
            "sources": sources,
            "validator_summary": (validation or {}).get("summary"),
            "citation_coverage": citations,
        })
        readability.append({
            "run": run, "stage": stage, "sources": sources, "metrics": quality,
        })

    for run, source in WORKFLOW_FILES.items():
        data = read(source) or {}
        before = data.get("initial_validation")
        after = data.get("final_validation", data.get("cv_validation"))
        if "initial_cv" in data or before is not None:
            add_snapshot(run, "initial", data.get("initial_cv"), before,
                         data.get("cv_composition"), [source])
        add_snapshot(run, "final", data.get("final_cv"), after,
                     data.get("cv_composition"), [source])
        warnings = data.get("warnings")
        workflow.append({
            "run": run,
            "source": source,
            "final_status": data.get("final_status"),
            "revision_rounds": data.get("revision_rounds", data.get("revision_round")),
            "max_revision_rounds": data.get("max_revision_rounds"),
            "warning_count": None if warnings is None else len(warnings),
            "verdict_delta": validation_delta(before, after),
            **metrics_from(data, ("llm_calls", "retry_count", "latency_seconds", "run_mode")),
        })

    for stage, cv_name, validation_name in (
        ("initial", "draft", "validation"),
        ("revised", "revised", "revalidation"),
    ):
        cv_path = f"fixtures/ai_engineer_cv_{cv_name}.json"
        validation_path = f"fixtures/ai_engineer_cv_{validation_name}.json"
        add_snapshot("standalone_cv", stage, read(cv_path), read(validation_path),
                     None, [cv_path, validation_path])

    report = {
        "schema_version": 1,
        "measurement_mode": "saved_artifacts",
        "retrieval": retrieval,
        "grounding": grounding,
        "workflow": workflow,
        "cv_readability": readability,
        "missing_artifacts": sorted(missing),
        "measurement_notes": NOTES,
    }
    multi_jd_path = root / "results/multi_jd_summary.json"
    if multi_jd_path.exists():
        with multi_jd_path.open(encoding="utf-8") as stream:
            report["multi_jd_live_benchmark"] = {
                "source": "results/multi_jd_summary.json",
                "checkpoint": "results/multi_jd_run.json",
                "gold": "multi_jd_gold.json",
                "metrics": json.load(stream),
            }
    return report


def markdown_report(report):
    lines = ["# Unified evaluation report", "", NOTES["coverage"]]

    def table(headers, rows):
        def cell(value):
            if value is None:
                return "Unavailable"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value).replace("|", r"\|").replace("\n", " ")

        lines.append("")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)

    lines.extend(["", "## Retrieval quality", "", NOTES["retrieval"]])
    live_metrics = report["retrieval"]["live_graph"]["metrics"]
    retrieval_rows = [
        ("Live graph · end to end", None, None, *(live_metrics[key] for key in RETRIEVAL_METRICS))
    ]
    for item in report["retrieval"]["benchmarks"]:
        for name, system in (item["systems"] or {None: {"metrics": {}}}).items():
            retrieval_rows.append((
                item["id"], name, item["query_count"],
                *(system["metrics"].get(metric) for metric in RETRIEVAL_METRICS),
            ))
    table(("Evaluation", "System", "Queries", *RETRIEVAL_METRICS), retrieval_rows)
    table(("Benchmark", "Dataset", "Recorded RRF settings", "Recorded reranker", "Candidate k"), [
        (item["id"], item["evaluation_file"],
         json.dumps(item["hybrid_config"]) if item["hybrid_config"] is not None else None,
         item["reranker_model"], item["candidate_k"])
        for item in report["retrieval"]["benchmarks"]
    ])

    lines.extend(["", "## Grounding quality", "", NOTES["grounding"]])
    match_gap = report["grounding"]["match_gap_benchmark"]
    table(("Match/gap cases", "Status accuracy", "Attribution accuracy", "Fully correct"), [
        (match_gap["case_count"], *(match_gap["metrics"][name] for name in MATCH_GAP_METRICS)),
    ])
    statuses = ("SUPPORTED", "OVERSTATED", "UNSUPPORTED", "total", "grounded_rate")
    table(("CV run", "Stage", *statuses, "Citation coverage"), [
        (item["run"], item["stage"], *metrics_from(item["validator_summary"], statuses).values(),
         (item["citation_coverage"] or {}).get("coverage"))
        for item in report["grounding"]["cv_validations"]
    ])

    lines.extend(["", "## Workflow behavior", "", NOTES["workflow"]])
    table(("Run", "Final status", "Rounds", "Limit", "Warnings", "Changed verdicts",
           "LLM calls", "Retries", "Latency (s)", "Run mode"), [
        (item["run"], item["final_status"], item["revision_rounds"], item["max_revision_rounds"],
         item["warning_count"], (item["verdict_delta"] or {}).get("changed_verdicts"),
         item["llm_calls"], item["retry_count"], item["latency_seconds"], item["run_mode"])
        for item in report["workflow"]
    ])

    lines.extend(["", "## CV readability", "", NOTES["readability"]])
    quality_keys = ("profile_words", "bullet_count", "mean_bullet_words",
                    "longest_bullet_words", "duplicate_bullets")
    table(("Run", "Stage", "Profile words", "Bullets", "Mean words per bullet",
           "Longest bullet words", "Duplicate bullets"), [
        (item["run"], item["stage"], *metrics_from(item["metrics"], quality_keys).values())
        for item in report["cv_readability"]
    ])
    source_rows = [
        (item["run"], item["stage"], source["source_id"], source["bullet_count"],
         source["max_bullets"], source["longest_bullet_words"],
         source["max_words_per_bullet"], source["bullets_over_word_limit"])
        for item in report["cv_readability"]
        for source in (item["metrics"] or {}).get("sources", [])
        if source["max_bullets"] is not None or source["max_words_per_bullet"] is not None
    ]
    if source_rows:
        table(("Run", "Stage", "Source", "Bullets", "Bullet limit", "Longest (words)",
               "Word limit", "Bullets over word limit"), source_rows)

    live = report.get("multi_jd_live_benchmark")
    if live is not None:
        data = live["metrics"]
        lines.extend([
            "", "## Multi-JD live benchmark", "",
            f"{data['gold_requirements']} annotated requirements, "
            f"{data['scored_requirements']} scorable. Stage A completed for "
            f"{data['stage_a_completed']}/{len(data['roles'])} roles. "
            f"Stage B completed for {data['stage_b_completed']}/"
            f"{sum(role['stage_b'] is not None for role in data['roles'])} selected roles.",
            "",
            f"Retrieval uses {data['retrieval_path']}. Scores use the aggregate evidence "
            "package against frozen binary relevance labels. Accuracy across all gold requirements "
            "counts missing and ambiguous extraction as misses when Match/Gap completed. "
            "Attribution requires citations to be within the frozen acceptable set. "
            "Unobserved outcomes are unavailable.",
            "",
            f"Logical calls {data['logical_calls']} · interrupted calls {data['interrupted_calls']} "
            f"· rate limits {data['rate_limited_calls']} · SDK retries {data['sdk_automatic_retries']} "
            f"· repeated node attempts {data['repeated_node_attempts']}.",
        ])
        lines.extend([
            "",
            f"Configured model/router: "
            f"`{data.get('configured_model', data['provenance'].get('model'))}`. "
            f"Resolved model calls: "
            f"{json.dumps(data.get('resolved_model_distribution', {}), sort_keys=True)}. "
            f"Successful calls without resolved-model metadata: "
            f"{data.get('successful_calls_without_resolved_model', 0)}.",
        ])
        a_rows = [(role["role"], role["stage_a"]) for role in data["roles"]]
        table(("Role", "Gold aligned", "Accuracy, all gold", "Accuracy, aligned",
               "Attribution, aligned", "Gaps correct", "Core source coverage", "Plausible source precision"), [
            (role, row["aligned_requirements"], row["classification_accuracy_end_to_end"],
             row["classification_accuracy_aligned"], row["attribution_accuracy_aligned"],
             row["unsupported_correct"], (row["composition_sources"] or {}).get("core_coverage"),
             (row["composition_sources"] or {}).get("plausible_precision"))
            for role, row in a_rows
        ])
        table(("Live graph role", *RETRIEVAL_METRICS), [
            (role, *metrics_from(row["retrieval"], RETRIEVAL_METRICS).values())
            for role, row in a_rows
        ])
        table(("Cross-role metric", "Roles measured", "Macro mean", "Min", "Max"), [
            (metric, values["roles_measured"], values["macro_mean"], values["min"], values["max"])
            for metric, values in data["cross_role"].items()
        ])
        table(("Full CV role", "Generated claims", "Final citation coverage", "S / O / U",
               "Grounded rate", "Rounds", "Changed claims", "Targeted evaluations",
               "Untouched verdict preservation", "Final status"), [
            (role["role"], row["generated_claim_count"],
             (row["final_citation_coverage"] or {}).get("coverage"),
             " / ".join(str(row["final_validation"][key]) for key in ("SUPPORTED", "OVERSTATED", "UNSUPPORTED"))
             if row["final_validation"] else None,
             (row["final_validation"] or {}).get("grounded_rate"), row["revision_rounds"],
             len(row["changed_claim_ids"]) if row["changed_claim_ids"] is not None else None,
             row["targeted_claim_evaluations"], row["untouched_verdict_preservation_rate"], row["final_status"])
            for role in data["roles"] if (row := role["stage_b"]) is not None
        ])
        lines.extend(["", "Stage B readability is computed deterministically from the final CV. "
                      "Independently adjudicated unsupported-claim leakage is unavailable."])
        table(("Full CV role", "Profile words", "Bullets", "Longest bullet words", "Duplicate bullets"), [
            (role["role"], *metrics_from(row["readability"], (
                "profile_words", "bullet_count", "longest_bullet_words", "duplicate_bullets",
            )).values())
            for role in data["roles"] if (row := role["stage_b"]) is not None
        ])
        if data["failures_by_stage"]:
            table(("Failure stage", "Recorded issues"), sorted(data["failures_by_stage"].items()))
            table(("Role", "Stage", "Requirement or node", "Issue"), [
                (role["role"], failure["stage"], failure.get("requirement_id", failure.get("step")),
                 " / ".join(filter(None, [failure["issue"], failure.get("error_type"), failure.get("cause_type")])))
                for role in data["roles"] for failure in role["failures"]
            ])
        if data.get("retrieval_preflight"):
            lines.extend(["", "Unscored local retrieval dependency checks"])
            table(("Status", "Hits", "Dependency", "Error"), [
                (check["status"], check.get("hits"), check.get("dependency"), check.get("error_type"))
                for check in data["retrieval_preflight"]
            ])
        lines.extend(["", *[f"- `{live[key]}`" for key in ("gold", "checkpoint", "source")]])

    lines.extend(["", "## Saved inputs", ""])
    sources = {item["source"] for item in report["retrieval"]["benchmarks"]}
    sources.add(match_gap["source"])
    sources.update(source for item in report["grounding"]["cv_validations"] for source in item["sources"])
    missing = set(report["missing_artifacts"])
    lines.extend(f"- `{source}`{ ' (unavailable)' if source in missing else ''}" for source in sorted(sources))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=Path("evaluation"))
    parser.add_argument("--output-dir", type=Path, help="Defaults to <evaluation-root>/results.")
    args = parser.parse_args()
    report = build_report(args.evaluation_root)
    output_dir = args.output_dir or args.evaluation_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "unified_evaluation.json"
    markdown_path = output_dir / "unified_evaluation.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"Wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
