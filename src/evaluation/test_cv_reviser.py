import json
from pathlib import Path

from src.cv.cv_reviser import (
    revise_cv_draft,
)


FIXTURE_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_tailor_fixture.json"
)

CV_DRAFT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_draft.json"
)

VALIDATION_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_validation.json"
)

OUTPUT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_revised.json"
)


def main():

    print(
        "[1/3] Loading draft, validation "
        "and evidence fixture..."
    )

    with FIXTURE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        fixture = json.load(
            file
        )

    with CV_DRAFT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        cv_draft = json.load(
            file
        )

    with VALIDATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        validation = json.load(
            file
        )

    expanded_sources = fixture[
        "expanded_sources"
    ]

    print(
        "[1/3] Done."
    )

    print(
        "[2/3] Revising problematic "
        "claims only..."
    )

    revised_draft = revise_cv_draft(
        cv_draft=cv_draft,
        validation_result=(
            validation
        ),
        expanded_sources=(
            expanded_sources
        ),
    )

    print(
        "[2/3] Done."
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            revised_draft,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[3/3] Revised CV saved."
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "REVISED CV"
    )

    print(
        json.dumps(
            revised_draft,
            indent=2,
            ensure_ascii=False,
        )
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