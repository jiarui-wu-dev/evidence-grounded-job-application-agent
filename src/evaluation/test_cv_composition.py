import json
from pathlib import Path

from src.cv.composition_policy import (
    build_cv_composition,
)


FIXTURE_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_langgraph_full_result.json"
)


def main():

    print(
        "[1/2] Loading latest application strategy..."
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
        "[1/2] Done."
    )

    composition = (
        build_cv_composition(
            application_strategy=(
                strategy
            )
        )
    )

    print(
        "\n[2/2] CV COMPOSITION PLAN"
    )

    print(
        "=" * 100
    )

    print(
        "\nSECTION ORDER:"
    )

    for section in composition[
        "section_order"
    ]:
        print(
            f"  - {section}"
        )

    print(
        "\nCORE SOURCES:"
    )

    for source in composition[
        "core_sources"
    ]:

        print(
            f"  - {source['source_id']} "
            f"[{source['entity_type']}]"
        )

        print(
            f"    max bullets: "
            f"{source['max_bullets']}"
        )

        print(
            f"    max words/bullet: "
            f"{source['max_words_per_bullet']}"
        )

    print(
        "\nSUPPLEMENTAL SOURCES:"
    )

    for source in composition[
        "supplemental_sources"
    ]:

        print(
            f"  - {source['source_id']} "
            f"[{source['entity_type']}]"
        )

        print(
            f"    evidence status: "
            f"{source.get('evidence_status')}"
        )

        print(
            f"    max bullets: "
            f"{source['max_bullets']}"
        )

        print(
            f"    max words/bullet: "
            f"{source['max_words_per_bullet']}"
        )

    print(
        "\nFINAL SOURCES:"
    )

    for index, source in enumerate(
        composition["sources"],
        start=1,
    ):

        print(
            f"  {index}. "
            f"{source['source_id']} "
            f"[{source['composition_role']}]"
        )

    print(
        "\n"
        + "=" * 100
    )


if __name__ == "__main__":
    main()