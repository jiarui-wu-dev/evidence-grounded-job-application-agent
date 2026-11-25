import json
from pathlib import Path

from src.graph.application_graph import (
    retrieve_evidence_node,
    analyze_match_gap_node,
    build_strategy_node,
)


FIXTURE_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_langgraph_full_result.json"
)

TARGET_PROJECT = (
    "evidence_grounded_job_application_agent"
)


def main():

    print(
        "[1/4] Loading saved AI Engineer state..."
    )

    with FIXTURE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        saved_state = json.load(f)

    state = {
        "jd_text": saved_state.get(
            "jd_text",
            "",
        ),

        "jd_analysis": saved_state[
            "jd_analysis"
        ],

        # Keep the previous query count/context
        # if available.
        "queries": saved_state.get(
            "queries",
            [],
        ),
    }

    print(
        "[1/4] Done."
    )

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    print(
        "\n[2/4] Running retrieval..."
    )

    retrieval_update = (
        retrieve_evidence_node(
            state
        )
    )

    state.update(
        retrieval_update
    )

    evidence_package = state[
        "evidence_package"
    ]

    evidence = evidence_package[
        "evidence"
    ]

    print(
        f"[2/4] Retrieved "
        f"{len(evidence)} final evidence units."
    )

    print(
        "\nTOP RETRIEVED EVIDENCE"
    )

    print(
        "=" * 100
    )

    for rank, item in enumerate(
        evidence,
        start=1,
    ):

        print(
            f"\n#{rank}"
        )

        print(
            "Project:",
            item["project_id"],
        )

        print(
            "Evidence:",
            item["evidence_id"],
        )

        print(
            "Support count:",
            item.get(
                "support_count",
            ),
        )

        print(
            "Best rank:",
            item.get(
                "best_rank",
            ),
        )

        print(
            "Text:",
            item["text"],
        )

    target_retrieval_hits = [
        item
        for item in evidence
        if item["project_id"]
        == TARGET_PROJECT
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AGENT PROJECT IN RETRIEVAL:",
        "YES"
        if target_retrieval_hits
        else "NO",
    )

    if target_retrieval_hits:

        print(
            "Agent evidence hits:",
            len(
                target_retrieval_hits
            ),
        )

        for item in (
            target_retrieval_hits
        ):

            print(
                "  -",
                item[
                    "evidence_id"
                ],
            )

    # --------------------------------------------------
    # Match / Gap
    # --------------------------------------------------

    print(
        "\n[3/4] Running Match/Gap..."
    )

    match_update = (
        analyze_match_gap_node(
            state
        )
    )

    state.update(
        match_update
    )

    match_gap = state[
        "match_gap"
    ]

    print(
        "[3/4] Done."
    )

    print(
        "\nMATCH / GAP SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        json.dumps(
            match_gap.get(
                "summary",
                {},
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------
    # Strategy
    # --------------------------------------------------

    print(
        "\n[4/4] Building application strategy..."
    )

    strategy_update = (
        build_strategy_node(
            state
        )
    )

    state.update(
        strategy_update
    )

    strategy = state[
        "application_strategy"
    ]

    priority_sources = (
        strategy.get(
            "priority_sources",
            [],
        )
    )

    print(
        "[4/4] Done."
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PRIORITY SOURCES"
    )

    print(
        "=" * 100
    )

    for rank, source in enumerate(
        priority_sources,
        start=1,
    ):

        print(
            f"\n#{rank}"
        )

        print(
            "Source ID:",
            source.get(
                "source_id"
            ),
        )

        print(
            "Entity type:",
            source.get(
                "entity_type"
            ),
        )

        print(
            "Reason:",
            source.get(
                "reason",
                source.get(
                    "selection_reason",
                    "",
                ),
            ),
        )

    selected_ids = {
        source.get(
            "source_id"
        )
        for source
        in priority_sources
    }

    print(
        "\n"
        + "=" * 100
    )

    if (
        TARGET_PROJECT
        in selected_ids
    ):

        print(
            "RESULT: AGENT PROJECT SELECTED"
        )

    elif target_retrieval_hits:

        print(
            "RESULT: AGENT PROJECT RETRIEVED "
            "BUT NOT SELECTED"
        )

    else:

        print(
            "RESULT: AGENT PROJECT NOT RETRIEVED"
        )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()