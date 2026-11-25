import json
from pathlib import Path

from src.cv.composition_policy import (
    build_cv_composition,
)

from src.cv.source_evidence_expander import (
    expand_priority_sources,
)


FIXTURE_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_langgraph_full_result.json"
)


def main():

    print(
        "[1/3] Loading strategy..."
    )

    with FIXTURE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        state = json.load(file)

    strategy = state[
        "application_strategy"
    ]

    print(
        "[1/3] Done."
    )

    print(
        "\n[2/3] Building composition..."
    )

    composition = (
        build_cv_composition(
            application_strategy=(
                strategy
            )
        )
    )

    print(
        "[2/3] Done."
    )

    print(
        "\n[3/3] Expanding composition sources..."
    )

    expanded = (
        expand_priority_sources(
            composition
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "EXPANDED CV SOURCES"
    )

    print(
        "=" * 100
    )

    for index, source in enumerate(
        expanded,
        start=1,
    ):

        print(
            f"\n{index}. "
            f"{source['source_id']}"
        )

        print(
            "   entity_type:",
            source[
                "entity_type"
            ],
        )

        print(
            "   role:",
            source[
                "composition_role"
            ],
        )

        print(
            "   evidence_statuses:",
            source[
                "evidence_statuses"
            ],
        )

        print(
            "   evidence_count:",
            len(
                source[
                    "evidence"
                ]
            ),
        )

        print(
            "   max_bullets:",
            source[
                "max_bullets"
            ],
        )

        print(
            "   max_words_per_bullet:",
            source[
                "max_words_per_bullet"
            ],
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "Expected sources: 4"
    )

    print(
        "Actual sources:",
        len(expanded),
    )

    print(
        "[3/3] Done."
    )


if __name__ == "__main__":
    main()