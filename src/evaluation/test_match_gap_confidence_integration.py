import json

from src.analysis.match_gap_analyzer import (
    analyze_match_gap,
)

def build_evidence_package(
    evidence,
):
    return {
        "job_title": (
            "Confidence Integration Test"
        ),
        "queries": [],
        "query_count": 0,
        "evidence_count": len(
            evidence
        ),
        "evidence": evidence,
        "query_results": [],
    }


def main():
    # -------------------------------------------------
    # Only provisional evidence from a synthetic experience is supplied.
    #
    # Even if the LLM considers the requirement
    # directly supported, final output must not remain
    # SUPPORTED without verified evidence.
    # -------------------------------------------------

    evidence = [
        {
            "evidence_id": evidence_id,
            "project_title": "Example Software Testing Placement",
            "entity_type": "professional_experience",
            "type": evidence_type,
            "evidence_status": "provisional",
            "text": description,
        }
        for evidence_id, evidence_type, description in (
            ("candidate_experience_1_evidence_0", "professional_experience_evidence", "Performed functional software tests on example workflows."),
            ("candidate_experience_1_skills", "professional_experience_skills", "Skills: functional testing and defect reporting."),
        )
    ]

    evidence_package = (
        build_evidence_package(
            evidence
        )
    )

    jd_analysis = {
        "job_title": (
            "Software Testing Working Student"
        ),

        "must_have_skills": [
            "functional software testing"
        ],

        "nice_to_have_skills": [],

        "responsibilities": [],

        "experience_requirements": [],

        "education_requirements": [],

        "languages": [],

        "keywords": [
            "functional testing",
            "software testing",
        ],
    }

    print(
        "=" * 100
    )

    print(
        "PROVISIONAL EVIDENCE"
    )

    for item in evidence:
        print(
            f"{item['evidence_id']} | "
            f"{item['entity_type']} | "
            f"{item['evidence_status']}"
        )

        print(
            item["text"]
        )

        print()

    print(
        "=" * 100
    )

    print(
        "RUNNING LLM-BACKED "
        "MATCH / GAP ANALYSIS"
    )

    result = analyze_match_gap(
        jd_analysis=jd_analysis,
        evidence_package=(
            evidence_package
        ),
    )

    print(
        "\nFINAL RESULT"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    requirement = (
        result[
            "requirements"
        ][0]
    )

    final_status = requirement[
        "status"
    ]

    if final_status != "PARTIAL":
        raise AssertionError(
            "Provisional-only evidence "
            "must result in PARTIAL. "
            f"Got: {final_status}"
        )

    print(
        "\n"
        + "#" * 100
    )

    print(
        "PASS: provisional-only evidence "
        "did not produce final SUPPORTED"
    )

    if (
        "confidence_adjustment"
        in requirement
    ):
        print(
            "Guard action: "
            + requirement[
                "confidence_adjustment"
            ]
        )

    else:
        print(
            "The LLM itself classified the "
            "requirement as PARTIAL, so no "
            "deterministic downgrade was needed."
        )

    print(
        "#" * 100
    )


if __name__ == "__main__":
    main()