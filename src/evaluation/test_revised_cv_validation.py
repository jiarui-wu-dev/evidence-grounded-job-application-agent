import json
from pathlib import Path

from src.cv.evidence_validator import (
    validate_cv_evidence,
)


FIXTURE_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_tailor_fixture.json"
)

ORIGINAL_VALIDATION_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_validation.json"
)

REVISED_CV_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_revised.json"
)

OUTPUT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_revalidation.json"
)


def main():
    print(
        "[1/4] Loading revised CV "
        "and frozen evidence..."
    )

    with FIXTURE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        fixture = json.load(
            file
        )

    with REVISED_CV_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        revised_cv = json.load(
            file
        )

    expanded_sources = fixture[
        "expanded_sources"
    ]

    print(
        "[1/4] Done."
    )

    print(
        "[2/4] Re-validating revised CV..."
    )

    result = validate_cv_evidence(
        cv_draft=revised_cv,
        expanded_sources=(
            expanded_sources
        ),
    )

    print(
        "[2/4] Done."
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[3/4] Revalidation saved."
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "REVISED CV VALIDATION"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    # -------------------------------------------------
    # Compare before vs after revision.
    # -------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "BEFORE / AFTER"
    )

    if ORIGINAL_VALIDATION_PATH.exists():

        with ORIGINAL_VALIDATION_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            original_validation = (
                json.load(
                    file
                )
            )

        before = original_validation[
            "summary"
        ]

        after = result[
            "summary"
        ]

        print(
            "\nBefore revision:"
        )

        print(
            f"  SUPPORTED:   "
            f"{before['SUPPORTED']}"
        )

        print(
            f"  OVERSTATED:  "
            f"{before['OVERSTATED']}"
        )

        print(
            f"  UNSUPPORTED: "
            f"{before['UNSUPPORTED']}"
        )

        print(
            f"  Grounded rate: "
            f"{before['grounded_rate']:.1%}"
        )

        print(
            "\nAfter revision:"
        )

        print(
            f"  SUPPORTED:   "
            f"{after['SUPPORTED']}"
        )

        print(
            f"  OVERSTATED:  "
            f"{after['OVERSTATED']}"
        )

        print(
            f"  UNSUPPORTED: "
            f"{after['UNSUPPORTED']}"
        )

        print(
            f"  Grounded rate: "
            f"{after['grounded_rate']:.1%}"
        )

        improvement = (
            after[
                "grounded_rate"
            ]
            - before[
                "grounded_rate"
            ]
        )

        print(
            "\nGrounded-rate improvement: "
            f"{improvement:+.1%}"
        )

    else:
        print(
            "Original validation result "
            "not found."
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()