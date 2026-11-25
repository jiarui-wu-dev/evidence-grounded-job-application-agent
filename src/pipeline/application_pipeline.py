import json
from typing import Any, Dict

from src.analysis.evidence_retriever import (
    EvidenceRetriever,
)
from src.analysis.match_gap_analyzer import (
    analyze_match_gap,
)
from src.analysis.application_strategy import (
    build_application_strategy,
)


def run_application_pipeline(
    jd_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Sequential baseline pipeline with progress logging.
    """

    print("[INIT] Loading evidence retriever...")

    evidence_retriever = EvidenceRetriever()

    try:
        print("[1/3] Retrieving candidate evidence...")

        evidence_package = (
            evidence_retriever.retrieve(
                jd_analysis=jd_analysis,
                max_queries=8,
                per_query_top_k=5,
                candidate_k=10,
                final_top_k=10,
            )
        )

        print(
            "[1/3] Done. "
            f"Retrieved "
            f"{evidence_package['evidence_count']} "
            f"evidence items."
        )

        print(
            "[2/3] Running match / gap analysis..."
        )

        match_gap_result = (
            analyze_match_gap(
                jd_analysis=jd_analysis,
                evidence_package=(
                    evidence_package
                ),
            )
        )

        print(
            "[2/3] Done. "
            f"SUPPORTED="
            f"{match_gap_result['summary']['SUPPORTED']}, "
            f"PARTIAL="
            f"{match_gap_result['summary']['PARTIAL']}, "
            f"GAP="
            f"{match_gap_result['summary']['GAP']}"
        )

        print(
            "[3/3] Building application strategy..."
        )

        strategy = (
            build_application_strategy(
                jd_analysis=jd_analysis,
                evidence_package=(
                    evidence_package
                ),
                match_gap_result=(
                    match_gap_result
                ),
            )
        )

        print("[3/3] Done.")

        return {
            "job_title": (
                jd_analysis.get(
                    "job_title"
                )
            ),
            "jd_analysis": (
                jd_analysis
            ),
            "evidence_package": (
                evidence_package
            ),
            "match_gap": (
                match_gap_result
            ),
            "strategy": strategy,
        }

    finally:
        print("[CLEANUP] Closing retriever...")
        evidence_retriever.close()
        print("[CLEANUP] Done.")

if __name__ == "__main__":
    example_analysis = {
        "job_title": (
            "Working Student AI Engineer"
        ),
        "must_have_skills": [
            "Python",
            "machine learning",
            "NLP",
            "RAG",
        ],
        "nice_to_have_skills": [
            "Docker",
            "LangGraph",
        ],
        "responsibilities": [
            (
                "Develop and evaluate "
                "LLM-based applications"
            ),
            (
                "Build retrieval-augmented "
                "generation workflows"
            ),
            (
                "Analyze model failures and "
                "improve system reliability"
            ),
        ],
        "experience_requirements": [
            (
                "hands-on experience with "
                "machine learning projects"
            )
        ],
        "education_requirements": [],
        "languages": [],
        "keywords": [
            "Python",
            "LLM",
            "RAG",
            "Docker",
            "evaluation",
        ],
    }

    result = run_application_pipeline(
        jd_analysis=example_analysis
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MATCH / GAP SUMMARY"
    )

    print(
        json.dumps(
            result[
                "match_gap"
            ]["summary"],
            indent=2,
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "APPLICATION STRATEGY"
    )

    print(
        json.dumps(
            result["strategy"],
            indent=2,
            ensure_ascii=False,
        )
    )