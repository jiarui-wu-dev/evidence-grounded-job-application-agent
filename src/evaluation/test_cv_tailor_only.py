import json
from pathlib import Path

from src.cv.cv_tailor import (
    build_cv_draft,
)


FIXTURE_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_tailor_fixture.json"
)


def main():
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(
            "CV Tailor fixture does not exist. "
            "Run:\n"
            "python -m "
            "src.evaluation.build_cv_tailor_fixture"
        )

    print(
        "[1/3] Loading frozen fixture..."
    )

    with FIXTURE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        fixture = json.load(
            file
        )

    jd_analysis = fixture[
        "jd_analysis"
    ]

    match_gap_result = fixture[
        "match_gap_result"
    ]

    application_strategy = fixture[
        "application_strategy"
    ]

    expanded_sources = fixture[
        "expanded_sources"
    ]

    print(
        "[1/3] Done."
    )

    print(
        "\nFrozen priority sources:"
    )

    for source in expanded_sources:
        print(
            f"  - {source['source_id']} "
            f"[{source['entity_type']}] "
            f"({len(source['evidence'])} evidence)"
        )

    print(
        "\n[2/3] Generating CV draft..."
    )

    cv_draft = build_cv_draft(
        jd_analysis=jd_analysis,
        match_gap_result=(
            match_gap_result
        ),
        application_strategy=(
            application_strategy
        ),
        expanded_sources=(
            expanded_sources
        ),
    )

    print(
        "[2/3] Done."
    )

    output_path = Path(
        "evaluation/fixtures/"
        "ai_engineer_cv_draft.json"
    )

    with output_path.open(
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
        "[3/3] Saved CV draft."
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CV DRAFT"
    )

    print(
        json.dumps(
            cv_draft,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()