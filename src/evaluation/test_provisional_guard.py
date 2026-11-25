import copy
import json

from src.analysis.match_gap_analyzer import (
    enforce_evidence_confidence,
)

def run_case(
    name: str,
    original_result,
    evidence,
    expected_status: str,
):
    """
    Apply deterministic evidence-confidence
    enforcement and verify the resulting status.
    """

    result = copy.deepcopy(
        original_result
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"CASE: {name}"
    )

    print(
        "\nBefore:"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nEvidence:"
    )

    for item in evidence:
        print(
            f"- {item['evidence_id']} | "
            f"{item.get('entity_type')} | "
            f"{item.get('evidence_status')}"
        )

    enforce_evidence_confidence(
        result=result,
        evidence=evidence,
    )

    print(
        "\nAfter:"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    actual_status = (
        result[
            "requirements"
        ][0]["status"]
    )

    if actual_status != expected_status:
        raise AssertionError(
            f"{name}: expected "
            f"{expected_status}, "
            f"got {actual_status}"
        )

    print(
        f"\nPASS: "
        f"{actual_status}"
    )


def main():
    # -------------------------------------------------
    # Case 1:
    # Provisional-only evidence.
    #
    # LLM hypothetically claims SUPPORTED.
    # Deterministic guard must downgrade to PARTIAL.
    # -------------------------------------------------

    provisional_testing = {
        "evidence_id": "candidate_experience_1_evidence_0",
        "entity_type": "professional_experience",
        "evidence_status": "provisional",
    }

    provisional_result = {
        "requirements": [
            {
                "requirement_id": (
                    "must_1"
                ),
                "status": (
                    "SUPPORTED"
                ),
                "evidence_ids": [
                    provisional_testing["evidence_id"]
                ],
                "explanation": (
                    "The candidate has "
                    "functional testing "
                    "experience."
                ),
            }
        ]
    }

    run_case(
        name=(
            "Provisional-only evidence "
            "must not remain SUPPORTED"
        ),
        original_result=(
            provisional_result
        ),
        evidence=[
            provisional_testing
        ],
        expected_status=(
            "PARTIAL"
        ),
    )

    # -------------------------------------------------
    # Case 2:
    # Verified-only evidence.
    #
    # SUPPORTED should remain SUPPORTED.
    # -------------------------------------------------

    verified_skills = {
        "evidence_id": "research_project_alpha_skills",
        "entity_type": "research_experience",
        "evidence_status": "verified",
    }

    verified_result = {
        "requirements": [
            {
                "requirement_id": (
                    "must_1"
                ),
                "status": (
                    "SUPPORTED"
                ),
                "evidence_ids": [
                    verified_skills["evidence_id"]
                ],
                "explanation": (
                    "The candidate has "
                    "verified Python and "
                    "LLM experience."
                ),
            }
        ]
    }

    run_case(
        name=(
            "Verified evidence may "
            "remain SUPPORTED"
        ),
        original_result=(
            verified_result
        ),
        evidence=[
            verified_skills
        ],
        expected_status=(
            "SUPPORTED"
        ),
    )

    # -------------------------------------------------
    # Case 3:
    # Mixed verified + provisional evidence.
    #
    # At least one verified supporting item exists,
    # so SUPPORTED may remain SUPPORTED.
    # -------------------------------------------------

    mixed_result = {
        "requirements": [
            {
                "requirement_id": (
                    "must_1"
                ),
                "status": (
                    "SUPPORTED"
                ),
                "evidence_ids": [
                    provisional_testing["evidence_id"],
                    verified_skills["evidence_id"],
                ],
                "explanation": (
                    "The requirement is "
                    "supported by both "
                    "verified and provisional "
                    "candidate evidence."
                ),
            }
        ]
    }

    run_case(
        name=(
            "Mixed evidence with at least "
            "one verified item may remain "
            "SUPPORTED"
        ),
        original_result=(
            mixed_result
        ),
        evidence=[
            provisional_testing,
            verified_skills,
        ],
        expected_status=(
            "SUPPORTED"
        ),
    )

    print(
        "\n"
        + "#" * 100
    )

    print(
        "ALL PROVISIONAL GUARD "
        "TESTS PASSED"
    )

    print(
        "#" * 100
    )


if __name__ == "__main__":
    main()