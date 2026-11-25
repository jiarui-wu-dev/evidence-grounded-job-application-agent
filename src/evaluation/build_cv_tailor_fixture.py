import json
from pathlib import Path

from src.analysis.evidence_retriever import (
    EvidenceRetriever,
)
from src.analysis.match_gap_analyzer import (
    analyze_match_gap,
)
from src.analysis.application_strategy import (
    build_application_strategy,
)
from src.cv.source_evidence_expander import (
    expand_priority_sources,
)


FIXTURE_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_tailor_fixture.json"
)


def main():
    jd_analysis = {
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
            "machine learning",
            "NLP",
            "LLM",
            "RAG",
            "Docker",
            "LangGraph",
            "evaluation",
        ],
    }

    print(
        "[INIT] Loading evidence retriever..."
    )

    retriever = EvidenceRetriever()

    try:
        # ---------------------------------------------
        # Step 1
        # ---------------------------------------------

        print(
            "[1/4] Retrieving candidate evidence..."
        )

        evidence_package = retriever.retrieve(
            jd_analysis
        )

        print(
            f"[1/4] Done. Retrieved "
            f"{evidence_package['evidence_count']} "
            f"evidence items."
        )

        # ---------------------------------------------
        # Step 2
        # ---------------------------------------------

        print(
            "[2/4] Running match / gap analysis..."
        )

        match_gap_result = analyze_match_gap(
            jd_analysis=jd_analysis,
            evidence_package=evidence_package,
        )

        summary = match_gap_result[
            "summary"
        ]

        print(
            "[2/4] Done. "
            f"SUPPORTED={summary['SUPPORTED']}, "
            f"PARTIAL={summary['PARTIAL']}, "
            f"GAP={summary['GAP']}"
        )

        # ---------------------------------------------
        # Step 3
        # ---------------------------------------------

        print(
            "[3/4] Building application strategy..."
        )

        application_strategy = (
            build_application_strategy(
                jd_analysis=jd_analysis,
                evidence_package=evidence_package,
                match_gap_result=(
                    match_gap_result
                ),
            )
        )

        print(
            "[3/4] Done."
        )

        # ---------------------------------------------
        # Step 4
        # ---------------------------------------------

        print(
            "[4/4] Expanding priority sources..."
        )

        expanded_sources = (
            expand_priority_sources(
                application_strategy
            )
        )

        print(
            f"[4/4] Done. Expanded "
            f"{len(expanded_sources)} sources."
        )

        for source in expanded_sources:
            print(
                f"  - {source['source_id']} "
                f"[{source['entity_type']}] "
                f"({len(source['evidence'])} evidence)"
            )

        # ---------------------------------------------
        # Freeze fixture
        # ---------------------------------------------

        fixture = {
            "jd_analysis": (
                jd_analysis
            ),

            "evidence_package": (
                evidence_package
            ),

            "match_gap_result": (
                match_gap_result
            ),

            "application_strategy": (
                application_strategy
            ),

            "expanded_sources": (
                expanded_sources
            ),
        }

        FIXTURE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with FIXTURE_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                fixture,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\n"
            + "=" * 100
        )

        print(
            "FIXTURE SAVED"
        )

        print(
            FIXTURE_PATH
        )

        print(
            "=" * 100
        )

    finally:
        print(
            "\n[CLEANUP] Closing retriever..."
        )

        retriever.close()

        print(
            "[CLEANUP] Done."
        )


if __name__ == "__main__":
    main()