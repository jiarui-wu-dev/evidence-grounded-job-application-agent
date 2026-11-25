from typing import Any, Dict, List

from src.retrieval.profile_documents import (
    load_profile_documents,
)


def expand_priority_sources(
    source_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Expand selected CV sources into their complete
    candidate evidence.

    Supports both:

    1. Application Strategy:
       {
           "priority_sources": [...]
       }

    2. CV Composition Plan:
       {
           "sources": [...]
       }

    This keeps backward compatibility with existing
    tests while allowing composition-aware CV generation.
    """

    documents = load_profile_documents()

    # -------------------------------------------------
    # Group Candidate KB evidence by source.
    # -------------------------------------------------

    source_map: Dict[
        str,
        List[Dict[str, Any]]
    ] = {}

    for document in documents:

        source_id = document[
            "project_id"
        ]

        source_map.setdefault(
            source_id,
            [],
        )

        source_map[
            source_id
        ].append(
            document
        )

    # -------------------------------------------------
    # Prefer composition sources when available.
    #
    # Fall back to strategy priority_sources so older
    # callers and tests continue to work.
    # -------------------------------------------------

    selected_sources = (
        source_plan.get(
            "sources"
        )
    )

    if selected_sources is None:

        selected_sources = (
            source_plan.get(
                "priority_sources",
                [],
            )
        )

    expanded_sources = []

    # -------------------------------------------------
    # Preserve source order from the supplied plan.
    # -------------------------------------------------

    for selected_source in (
        selected_sources
    ):

        source_id = (
            selected_source.get(
                "source_id"
            )
        )

        expected_entity_type = (
            selected_source.get(
                "entity_type"
            )
        )

        if source_id not in source_map:

            raise ValueError(
                "Selected source not found "
                f"in Candidate KB: "
                f"{source_id}"
            )

        source_documents = (
            source_map[
                source_id
            ]
        )

        actual_entity_types = {
            document.get(
                "entity_type",
                "project",
            )
            for document
            in source_documents
        }

        if len(
            actual_entity_types
        ) != 1:

            raise ValueError(
                "Inconsistent entity types "
                f"for source {source_id}: "
                f"{actual_entity_types}"
            )

        actual_entity_type = next(
            iter(
                actual_entity_types
            )
        )

        if (
            expected_entity_type
            != actual_entity_type
        ):

            raise ValueError(
                "Selected source entity type "
                "does not match Candidate KB "
                f"for {source_id}: expected "
                f"{expected_entity_type}, "
                f"got {actual_entity_type}"
            )

        evidence_statuses = {
            document.get(
                "evidence_status",
                "verified",
            )
            for document
            in source_documents
        }

        expanded_sources.append({
            "source_id": (
                source_id
            ),

            "source_title": (
                source_documents[
                    0
                ][
                    "project_title"
                ]
            ),

            "entity_type": (
                actual_entity_type
            ),

            "strategy_reason": (
                selected_source.get(
                    "reason",
                    "",
                )
            ),

            # Composition metadata.
            #
            # These fields are ignored by older
            # callers but can now be consumed by
            # CV Tailor.
            "composition_role": (
                selected_source.get(
                    "composition_role",
                    "core",
                )
            ),

            "max_bullets": (
                selected_source.get(
                    "max_bullets"
                )
            ),

            "max_words_per_bullet": (
                selected_source.get(
                    "max_words_per_bullet"
                )
            ),

            "evidence_statuses": (
                sorted(
                    evidence_statuses
                )
            ),

            "evidence": [
                {
                    "evidence_id": (
                        document[
                            "evidence_id"
                        ]
                    ),

                    "type": (
                        document[
                            "type"
                        ]
                    ),

                    "evidence_status": (
                        document.get(
                            "evidence_status",
                            "verified",
                        )
                    ),

                    "text": (
                        document[
                            "text"
                        ]
                    ),
                }
                for document
                in source_documents
            ],
        })

    return expanded_sources

def print_expanded_sources(
    expanded_sources: List[
        Dict[str, Any]
    ],
) -> None:
    """
    Human-readable diagnostic output.
    """

    print(
        "=" * 100
    )

    print(
        "EXPANDED PRIORITY SOURCES"
    )

    for index, source in enumerate(
        expanded_sources,
        start=1,
    ):
        print(
            "\n"
            + "-" * 100
        )

        print(
            f"Source {index}"
        )

        print(
            f"ID: "
            f"{source['source_id']}"
        )

        print(
            f"Title: "
            f"{source['source_title']}"
        )

        print(
            f"Entity type: "
            f"{source['entity_type']}"
        )

        print(
            "Evidence statuses: "
            + ", ".join(
                source[
                    "evidence_statuses"
                ]
            )
        )

        print(
            f"Evidence count: "
            f"{len(source['evidence'])}"
        )

        print(
            f"Strategy reason: "
            f"{source['strategy_reason']}"
        )

        print(
            "\nEvidence:"
        )

        for evidence in source[
            "evidence"
        ]:
            print(
                "\n"
                f"[{evidence['type']}] "
                f"[{evidence['evidence_status']}] "
                f"{evidence['evidence_id']}"
            )

            print(
                evidence[
                    "text"
                ]
            )


if __name__ == "__main__":
    # Standalone diagnostic test.
    #
    # This intentionally does not call an LLM
    # and therefore consumes no OpenRouter quota.

    examples = {"research_experience": {}, "project": {}}
    limits = {"research_experience": 1, "project": 2}
    for document in load_profile_documents():
        entity_type = document.get("entity_type", "project")
        if entity_type in examples and len(examples[entity_type]) < limits[entity_type]:
            source_id = document["project_id"]
            examples[entity_type].setdefault(source_id, {
                "source_id": source_id,
                "entity_type": entity_type,
            })

    example_strategy = {
        "priority_sources": [source for group in examples.values() for source in group.values()]
    }
    if not example_strategy["priority_sources"]:
        raise SystemExit("Add a local candidate profile to inspect expanded evidence.")

    expanded = (
        expand_priority_sources(
            example_strategy
        )
    )

    print_expanded_sources(
        expanded
    )