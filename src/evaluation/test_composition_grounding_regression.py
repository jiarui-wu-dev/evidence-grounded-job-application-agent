import json
from pathlib import Path

from src.cv.composition_policy import (
    build_cv_composition,
)

from src.cv.source_evidence_expander import (
    expand_priority_sources,
)

from src.cv.cv_tailor import (
    build_cv_draft,
    validate_draft_structure,
)

from src.cv.evidence_validator import (
    find_changed_claim_ids,
    validate_cv_evidence,
)

from src.cv.cv_reviser import (
    revise_cv_draft,
)


INPUT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_langgraph_full_result.json"
)

OUTPUT_PATH = Path(
    "evaluation/results/"
    "ai_engineer_composition_targeted_revalidation.json"
)


def count_problems(
    validation,
):

    summary = validation.get(
        "summary",
        {},
    )

    return (
        int(
            summary.get(
                "OVERSTATED",
                0,
            )
        )
        + int(
            summary.get(
                "UNSUPPORTED",
                0,
            )
        )
    )


def print_validation(
    label,
    validation,
):

    summary = validation.get(
        "summary",
        {},
    )

    supported = int(
        summary.get(
            "SUPPORTED",
            0,
        )
    )

    overstated = int(
        summary.get(
            "OVERSTATED",
            0,
        )
    )

    unsupported = int(
        summary.get(
            "UNSUPPORTED",
            0,
        )
    )

    total = (
        supported
        + overstated
        + unsupported
    )

    grounded_rate = (
        supported / total
        if total
        else 0.0
    )

    print(
        f"\n{label}"
    )

    print(
        "=" * 100
    )

    print(
        "SUPPORTED:",
        supported,
    )

    print(
        "OVERSTATED:",
        overstated,
    )

    print(
        "UNSUPPORTED:",
        unsupported,
    )

    print(
        "Grounded rate:",
        f"{grounded_rate:.1%}",
    )

    problems = [
        result
        for result in validation.get(
            "results",
            [],
        )
        if result.get(
            "status"
        )
        in {
            "OVERSTATED",
            "UNSUPPORTED",
        }
    ]

    if problems:

        print(
            "\nProblematic claims:"
        )

        for result in problems:

            print(
                "\n-",
                result.get(
                    "claim_id"
                ),
            )

            print(
                "  status:",
                result.get(
                    "status"
                ),
            )

            print(
                "  claim:",
                result.get(
                    "claim_text",
                    result.get(
                        "text",
                        "",
                    ),
                ),
            )

            print(
                "  reason:",
                result.get(
                    "reason",
                    "",
                ),
            )


def print_cv_structure(
    cv,
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL CV STRUCTURE"
    )

    print(
        "=" * 100
    )

    section_names = [
        (
            "research_experience",
            "Research Experience",
        ),
        (
            "experience",
            "Experience",
        ),
        (
            "selected_projects",
            "Selected Projects",
        ),
    ]

    for key, label in section_names:

        print(
            f"\n{label}:"
        )

        entries = cv.get(
            key,
            [],
        )

        if not entries:

            print(
                "  (empty)"
            )

            continue

        for entry in entries:

            print(
                f"\n  {entry.get('source_id')}"
            )

            for index, bullet in enumerate(
                entry.get(
                    "bullets",
                    [],
                ),
                start=1,
            ):

                text = bullet.get(
                    "text",
                    "",
                )

                print(
                    f"    {index}. "
                    f"[{len(text.split())} words] "
                    f"{text}"
                )


def main():

    # --------------------------------------------------
    # Load existing upstream state
    # --------------------------------------------------

    print(
        "[1/7] Loading latest grounded full-run state..."
    )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        state = json.load(file)

    jd_analysis = state[
        "jd_analysis"
    ]

    match_gap = state[
        "match_gap"
    ]

    strategy = state[
        "application_strategy"
    ]

    print(
        "[1/7] Done."
    )

    # --------------------------------------------------
    # Composition
    # --------------------------------------------------

    print(
        "\n[2/7] Building CV composition..."
    )

    composition = (
        build_cv_composition(
            application_strategy=(
                strategy
            )
        )
    )

    print(
        "[2/7] Done."
    )

    print(
        "\nComposition sources:"
    )

    for source in composition[
        "sources"
    ]:

        print(
            "  - "
            f"{source['source_id']} "
            f"[{source['entity_type']}] "
            f"[{source['composition_role']}]"
        )

    # --------------------------------------------------
    # Expand all composition sources
    # --------------------------------------------------

    print(
        "\n[3/7] Expanding composition sources..."
    )

    expanded_sources = (
        expand_priority_sources(
            composition
        )
    )

    print(
        "[3/7] Done."
    )

    print(
        "Expanded sources:",
        len(
            expanded_sources
        ),
    )

    # --------------------------------------------------
    # Tailor CV
    # --------------------------------------------------

    print(
        "\n[4/7] Generating composition-aware CV..."
    )

    cv_draft = (
        build_cv_draft(
            jd_analysis=(
                jd_analysis
            ),
            match_gap_result=(
                match_gap
            ),
            application_strategy=(
                strategy
            ),
            expanded_sources=(
                expanded_sources
            ),
        )
    )

    print(
        "[4/7] Done."
    )

    # --------------------------------------------------
    # Initial validation
    # --------------------------------------------------

    print(
        "\n[5/7] Running initial evidence validation..."
    )

    validation_1 = (
        validate_cv_evidence(
            cv_draft=(
                cv_draft
            ),
            expanded_sources=(
                expanded_sources
            ),
        )
    )

    print(
        "[5/7] Done."
    )

    print_validation(
        "INITIAL VALIDATION",
        validation_1,
    )

    initial_problems = (
        count_problems(
            validation_1
        )
    )

    # --------------------------------------------------
    # Maximum ONE revision round
    # --------------------------------------------------

    revision_rounds = 0

    final_cv = cv_draft
    final_validation = (
        validation_1
    )

    if initial_problems > 0:

        print(
            "\n[6/7] Running ONE targeted revision round..."
        )

        revised_cv = (
            revise_cv_draft(
                cv_draft=(
                    cv_draft
                ),
                validation_result=(
                    validation_1
                ),
                expanded_sources=(
                    expanded_sources
                ),
            )
        )

        validate_draft_structure(
            draft=(
                revised_cv
            ),
            expanded_sources=(
                expanded_sources
            ),
        )

        revision_rounds = 1

        print(
            "[6/7] Revision complete."
        )

        print(
            "\n[6/7] Re-validating changed claims..."
        )

        validation_2 = (
            validate_cv_evidence(
                cv_draft=(
                    revised_cv
                ),
                expanded_sources=(
                    expanded_sources
                ),
                changed_claim_ids=find_changed_claim_ids(
                    previous_draft=cv_draft,
                    revised_draft=revised_cv,
                    expanded_sources=expanded_sources,
                ),
                previous_validation=validation_1,
            )
        )

        final_cv = revised_cv
        final_validation = (
            validation_2
        )

        print(
            "[6/7] Re-validation complete."
        )

        print_validation(
            "FINAL VALIDATION",
            validation_2,
        )

    else:

        print(
            "\n[6/7] No problematic claims."
        )

        print(
            "[6/7] Revision and second "
            "validation skipped."
        )

    # --------------------------------------------------
    # Final status + save
    # --------------------------------------------------

    final_problems = (
        count_problems(
            final_validation
        )
    )

    final_status = (
        "grounded"
        if final_problems == 0
        else "needs_human_review"
    )

    print_cv_structure(
        final_cv
    )

    result = {
        "jd_analysis": (
            jd_analysis
        ),

        "match_gap": (
            match_gap
        ),

        "application_strategy": (
            strategy
        ),

        "cv_composition": (
            composition
        ),

        "expanded_sources": (
            expanded_sources
        ),

        "initial_cv": (
            cv_draft
        ),

        "initial_validation": (
            validation_1
        ),

        "final_cv": (
            final_cv
        ),

        "final_validation": (
            final_validation
        ),

        "revision_rounds": (
            revision_rounds
        ),

        "final_status": (
            final_status
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
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

        file.write(
            "\n"
        )

    print(
        "\n[7/7] Done."
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "REGRESSION RESULT"
    )

    print(
        "=" * 100
    )

    print(
        "Sources:",
        len(
            expanded_sources
        ),
    )

    print(
        "Revision rounds:",
        revision_rounds,
    )

    print(
        "Final problems:",
        final_problems,
    )

    print(
        "Final status:",
        final_status,
    )

    print(
        "Saved to:",
        OUTPUT_PATH,
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
