import argparse
import json
from pathlib import Path

from src.cv.composition_policy import (
    build_cv_composition,
)

from src.cv.source_evidence_expander import (
    expand_priority_sources,
)

from src.cv.cv_tailor import (
    SOURCE_SECTION_PROMPT,
    invoke_json_with_retry,
    validate_source_composition_result,
)

from src.llm.client import (
    get_llm,
)


INPUT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_langgraph_full_result.json"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load saved state
    # --------------------------------------------------

    print(
        "[1/4] Loading saved full-run state..."
    )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        state = json.load(file)

    jd_analysis = state[
        "jd_analysis"
    ]

    application_strategy = state[
        "application_strategy"
    ]

    print(
        "[1/4] Done."
    )

    # --------------------------------------------------
    # Build current composition
    # --------------------------------------------------

    print(
        "[2/4] Building current CV composition..."
    )

    composition = (
        build_cv_composition(
            application_strategy=(
                application_strategy
            )
        )
    )

    expanded_sources = (
        expand_priority_sources(
            composition
        )
    )

    source = next(
        (
            item
            for item
            in expanded_sources
            if item[
                "source_id"
            ]
            == args.source
        ),
        None,
    )

    if source is None:

        raise ValueError(
            f"Source not found in "
            f"composition: {args.source}"
        )

    print(
        "[2/4] Done."
    )

    print(
        "\nSOURCE COMPOSITION:"
    )

    print(
        "  source_id:",
        source[
            "source_id"
        ],
    )

    print(
        "  entity_type:",
        source[
            "entity_type"
        ],
    )

    print(
        "  role:",
        source[
            "composition_role"
        ],
    )

    print(
        "  evidence_statuses:",
        source[
            "evidence_statuses"
        ],
    )

    print(
        "  max_bullets:",
        source[
            "max_bullets"
        ],
    )

    print(
        "  max_words_per_bullet:",
        source[
            "max_words_per_bullet"
        ],
    )

    # --------------------------------------------------
    # Build same payload used by production Tailor
    # --------------------------------------------------

    source_payload = {
        "job_title": (
            jd_analysis.get(
                "job_title"
            )
        ),

        "source_id": (
            source[
                "source_id"
            ]
        ),

        "source_title": (
            source[
                "source_title"
            ]
        ),

        "entity_type": (
            source[
                "entity_type"
            ]
        ),

        "strategy_reason": (
            source.get(
                "strategy_reason",
                "",
            )
        ),

        "composition": {
            "role": (
                source.get(
                    "composition_role",
                    "core",
                )
            ),

            "max_bullets": (
                source.get(
                    "max_bullets",
                    3,
                )
            ),

            "max_words_per_bullet": (
                source.get(
                    "max_words_per_bullet",
                    42,
                )
            ),

            "evidence_statuses": (
                source.get(
                    "evidence_statuses",
                    [
                        "verified"
                    ],
                )
            ),
        },

        "job_requirements": {
            "must_have_skills": (
                jd_analysis.get(
                    "must_have_skills",
                    [],
                )
            ),

            "responsibilities": (
                jd_analysis.get(
                    "responsibilities",
                    [],
                )
            ),

            "experience_requirements": (
                jd_analysis.get(
                    "experience_requirements",
                    [],
                )
            ),

            "nice_to_have_skills": (
                jd_analysis.get(
                    "nice_to_have_skills",
                    [],
                )
            ),
        },

        "keyword_policy": {
            "emphasize": (
                application_strategy.get(
                    "cv_keywords_to_emphasize",
                    [],
                )
            ),

            "frame_carefully": (
                application_strategy.get(
                    "keywords_to_frame_carefully",
                    [],
                )
            ),

            "do_not_claim": (
                application_strategy.get(
                    "keywords_not_to_claim",
                    [],
                )
            ),
        },

        "source_evidence": (
            source[
                "evidence"
            ]
        ),
    }

    # --------------------------------------------------
    # Generate one source only
    # --------------------------------------------------

    print(
        "\n[3/4] Generating ONLY:",
        source[
            "source_id"
        ],
    )

    llm = get_llm(
        temperature=0.0
    )

    result = (
        invoke_json_with_retry(
            llm=llm,
            system_prompt=(
                SOURCE_SECTION_PROMPT
            ),
            payload=(
                source_payload
            ),
            label=(
                "single source "
                + source[
                    "source_id"
                ]
            ),
            max_attempts=1,
            result_validator=(
                validate_source_composition_result
            ),
        )
    )

    print(
        "[3/4] Done."
    )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RAW PARSED RESULT"
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

    bullets = result[
        "bullets"
    ]

    print(
        "BULLET DIAGNOSTICS"
    )

    for index, bullet in enumerate(
        bullets,
        start=1,
    ):

        text = bullet[
            "text"
        ]

        word_count = len(
            text.split()
        )

        print(
            f"\nBullet {index}: "
            f"{word_count} words"
        )

        print(
            text
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RESULT: PASS"
    )

    print(
        "Bullets returned:",
        len(
            bullets
        ),
    )

    print(
        "[4/4] Done."
    )


if __name__ == "__main__":
    main()