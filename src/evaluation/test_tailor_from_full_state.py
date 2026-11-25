import json
from pathlib import Path

from src.cv.cv_tailor import (
    build_cv_draft,
)

from src.cv.evidence_validator import (
    validate_cv_evidence,
)


INPUT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_langgraph_full_result.json"
)

DRAFT_OUTPUT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_prompt_regression_draft.json"
)

VALIDATION_OUTPUT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_prompt_regression_validation.json"
)


def main():

    print(
        "[1/4] Loading saved full-run state..."
    )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        state = json.load(
            file
        )

    jd_analysis = state[
        "jd_analysis"
    ]

    match_gap = state[
        "match_gap"
    ]

    application_strategy = state[
        "application_strategy"
    ]

    expanded_sources = state[
        "expanded_sources"
    ]

    print(
        "[1/4] Done."
    )

    print(
        "\nPriority sources:"
    )

    for source in expanded_sources:

        print(
            "  - "
            f"{source['source_id']} "
            f"[{source['entity_type']}]"
        )

    # ========================================================
    # Generate a fresh CV with the NEW Tailor prompt.
    # ========================================================

    print(
        "\n[2/4] Generating fresh CV draft "
        "with updated prompt..."
    )

    cv_draft = build_cv_draft(
        jd_analysis=(
            jd_analysis
        ),
        match_gap_result=(
            match_gap
        ),
        application_strategy=(
            application_strategy
        ),
        expanded_sources=(
            expanded_sources
        ),
    )

    with DRAFT_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            cv_draft,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[2/4] Done."
    )

    # ========================================================
    # Validate fresh initial draft.
    # No revision yet.
    # ========================================================

    print(
        "\n[3/4] Validating fresh draft..."
    )

    validation = validate_cv_evidence(
        cv_draft=cv_draft,
        expanded_sources=(
            expanded_sources
        ),
    )

    with VALIDATION_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            validation,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[3/4] Done."
    )

    # ========================================================
    # Show result.
    # ========================================================

    summary = validation[
        "summary"
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        "UPDATED PROMPT REGRESSION RESULT"
    )

    print(
        f"SUPPORTED:   "
        f"{summary['SUPPORTED']}"
    )

    print(
        f"OVERSTATED:  "
        f"{summary['OVERSTATED']}"
    )

    print(
        f"UNSUPPORTED: "
        f"{summary['UNSUPPORTED']}"
    )

    print(
        f"Grounded rate: "
        f"{summary['grounded_rate']:.1%}"
    )

    print(
        "\nProblematic claims:"
    )

    problematic = [
        item
        for item
        in validation[
            "results"
        ]
        if item[
            "status"
        ] != "SUPPORTED"
    ]

    if not problematic:

        print(
            "  None"
        )

    else:

        for item in problematic:

            print(
                "\n"
                f"  {item['claim_id']}"
            )

            print(
                f"    Status: "
                f"{item['status']}"
            )

            print(
                f"    Reason: "
                f"{item['reason']}"
            )

            phrases = item.get(
                "unsupported_phrases",
                [],
            )

            if phrases:

                print(
                    "    Unsupported:"
                )

                for phrase in phrases:

                    print(
                        f"      - {phrase}"
                    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"Draft saved to: "
        f"{DRAFT_OUTPUT_PATH}"
    )

    print(
        f"Validation saved to: "
        f"{VALIDATION_OUTPUT_PATH}"
    )

    print(
        "[4/4] Done."
    )


if __name__ == "__main__":
    main()