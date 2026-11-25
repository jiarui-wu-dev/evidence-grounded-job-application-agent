import json
from pathlib import Path

from src.cv.evidence_validator import (
    validate_cv_evidence,
)


FIXTURE_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_tailor_fixture.json"
)

CV_DRAFT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_draft.json"
)

OUTPUT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_validation.json"
)


def main():

    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(
            f"Fixture not found: "
            f"{FIXTURE_PATH}"
        )

    if not CV_DRAFT_PATH.exists():
        raise FileNotFoundError(
            f"CV draft not found: "
            f"{CV_DRAFT_PATH}"
        )

    print(
        "[1/3] Loading frozen fixture "
        "and CV draft..."
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

    expanded_sources = fixture[
        "expanded_sources"
    ]

    print(
        "[1/3] Done."
    )

    print(
        "[2/3] Running semantic "
        "evidence validation..."
    )

    result = validate_cv_evidence(
        cv_draft=cv_draft,
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
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[3/3] Validation result saved."
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "EVIDENCE VALIDATION"
    )

    print(
        json.dumps(
            result,
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