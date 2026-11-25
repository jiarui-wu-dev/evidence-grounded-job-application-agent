import copy
from unittest.mock import patch

from src.cv.cv_tailor import (
    validate_draft_structure,
)
from src.cv.source_evidence_expander import (
    expand_priority_sources,
)


def build_expanded_sources():
    """
    Expand three synthetic priority sources through the production expander.

    No LLM call is involved.
    """

    strategy = {
        "priority_sources": [
            {
                "source_id": "research_project_alpha",
                "entity_type": "research_experience",
                "reason": (
                    "Primary LLM research and "
                    "evaluation experience."
                ),
            },
            {
                "source_id": "candidate_project_1",
                "entity_type": "project",
                "reason": (
                    "Hands-on machine-learning "
                    "evaluation and adaptation."
                ),
            },
            {
                "source_id": "candidate_project_2",
                "entity_type": "project",
                "reason": (
                    "Backend engineering and "
                    "Docker experience."
                ),
            },
        ]
    }

    documents = [
        {
            "project_id": source_id,
            "project_title": title,
            "entity_type": entity_type,
            "evidence_id": f"{source_id}_{suffix}",
            "type": "skills" if suffix == "skills" else "evidence",
            "evidence_status": "verified",
            "text": "Synthetic evidence for structural validation.",
        }
        for source_id, title, entity_type, suffixes in (
            ("research_project_alpha", "Automated Design Validation", "research_experience", ("skills", "evidence_1", "evidence_2")),
            ("candidate_project_1", "Speech Recognition Model Evaluation", "project", ("skills", "evidence_3")),
            ("candidate_project_2", "Example Workflow Service", "project", ("skills", "evidence_3")),
        )
        for suffix in suffixes
    ]
    with patch("src.cv.source_evidence_expander.load_profile_documents", return_value=documents):
        return expand_priority_sources(strategy)


def build_valid_draft():
    """
    Build a small but structurally valid CV draft.

    The wording here is intentionally simple.
    Semantic correctness will later be handled by the
    Evidence Validator.
    """

    return {
        "profile": {
            "text": (
                "Junior AI engineer with hands-on "
                "experience in Python, LLM evaluation, "
                "model adaptation, and automated "
                "validation workflows."
            ),
            "evidence_ids": [
                "research_project_alpha_skills",
                "candidate_project_1_skills",
            ],
        },

        "skills": [
            {
                "category": "Programming",
                "items": [
                    {
                        "name": "Python",
                        "evidence_ids": [
                            "research_project_alpha_skills",
                            "candidate_project_1_skills",
                        ],
                    }
                ],
            },
            {
                "category": "AI / ML",
                "items": [
                    {
                        "name": "Machine Learning",
                        "evidence_ids": [
                            "candidate_project_1_skills",
                        ],
                    },
                    {
                        "name": "LLM Evaluation",
                        "evidence_ids": [
                            "research_project_alpha_skills",
                        ],
                    },
                ],
            },
            {
                "category": "Engineering",
                "items": [
                    {
                        "name": "Docker",
                        "evidence_ids": [
                            "candidate_project_2_skills",
                        ],
                    }
                ],
            },
        ],

        "research_experience": [
            {
                "source_id": (
                    "research_project_alpha"
                ),
                "title": (
                    "Research Project — "
                    "Automated Design "
                    "Validation"
                ),
                "organization": "Example University",
                "unit": (
                    "Example Computing Lab"
                ),
                "bullets": [
                    {
                        "text": (
                            "Built a candidate-validation "
                            "pipeline covering structural, "
                            "behavioral, and "
                            "security-semantic checks."
                        ),
                        "evidence_ids": [
                            (
                                "research_project_alpha_"
                                "evidence_1"
                            )
                        ],
                    },
                    {
                        "text": (
                            "Identified and rejected "
                            "invalid and semantically "
                            "inconsistent LLM-generated "
                            "candidates."
                        ),
                        "evidence_ids": [
                            (
                                "research_project_alpha_"
                                "evidence_2"
                            )
                        ],
                    },
                ],
            }
        ],

        "experience": [],

        "selected_projects": [
            {
                "source_id": "candidate_project_1",
                "title": (
                    "Speech Recognition "
                    "Model Evaluation"
                ),
                "bullets": [
                    {
                        "text": (
                            "Reduced catastrophic "
                            "decoding failures from "
                            "12/100 to 3/100."
                        ),
                        "evidence_ids": [
                            "candidate_project_1_evidence_3"
                        ],
                    }
                ],
            },
            {
                "source_id": "candidate_project_2",
                "title": (
                    "Example Workflow "
                    "Service"
                ),
                "bullets": [
                    {
                        "text": (
                            "Containerized the "
                            "application using Docker "
                            "and Docker Compose."
                        ),
                        "evidence_ids": [
                            (
                                "candidate_project_2_"
                                "evidence_3"
                            )
                        ],
                    }
                ],
            },
        ],
    }


def expect_pass(
    name,
    draft,
    expanded_sources,
):
    print(
        "\n"
        + "=" * 100
    )

    print(
        f"CASE: {name}"
    )

    validate_draft_structure(
        draft=draft,
        expanded_sources=(
            expanded_sources
        ),
    )

    print(
        "PASS"
    )


def expect_failure(
    name,
    draft,
    expanded_sources,
    expected_message_fragment,
):
    print(
        "\n"
        + "=" * 100
    )

    print(
        f"CASE: {name}"
    )

    try:
        validate_draft_structure(
            draft=draft,
            expanded_sources=(
                expanded_sources
            ),
        )

    except ValueError as error:
        message = str(
            error
        )

        print(
            "Correctly rejected:"
        )

        print(
            message
        )

        if (
            expected_message_fragment
            not in message
        ):
            raise AssertionError(
                "Expected error containing "
                f"{expected_message_fragment!r}, "
                f"got {message!r}"
            )

        print(
            "PASS"
        )

        return

    raise AssertionError(
        f"{name}: expected validation "
        "to fail, but it passed."
    )


def main():
    expanded_sources = (
        build_expanded_sources()
    )

    valid_draft = (
        build_valid_draft()
    )

    # -------------------------------------------------
    # Case 1:
    # Valid structure.
    # -------------------------------------------------

    expect_pass(
        name=(
            "Valid source-aware CV draft"
        ),
        draft=(
            valid_draft
        ),
        expanded_sources=(
            expanded_sources
        ),
    )

    # -------------------------------------------------
    # Case 2:
    # Research Experience must not be moved into
    # Selected Projects.
    # -------------------------------------------------

    misplaced_research = copy.deepcopy(
        valid_draft
    )

    research_entry = (
        misplaced_research[
            "research_experience"
        ].pop()
    )

    misplaced_research[
        "selected_projects"
    ].append(
        research_entry
    )

    expect_failure(
        name=(
            "Research experience must not appear "
            "in Selected Projects"
        ),
        draft=(
            misplaced_research
        ),
        expanded_sources=(
            expanded_sources
        ),
        expected_message_fragment=(
            "but was placed in "
            "selected_projects"
        ),
    )

    # -------------------------------------------------
    # Case 3:
    # A project bullet may not cite research evidence.
    # -------------------------------------------------

    cross_source_citation = copy.deepcopy(
        valid_draft
    )

    cross_source_citation[
        "selected_projects"
    ][0][
        "bullets"
    ][0][
        "evidence_ids"
    ] = [
        "research_project_alpha_evidence_2"
    ]

    expect_failure(
        name=(
            "Cross-source evidence citation "
            "must be rejected"
        ),
        draft=(
            cross_source_citation
        ),
        expanded_sources=(
            expanded_sources
        ),
        expected_message_fragment=(
            "Cross-source evidence citation"
        ),
    )

    # -------------------------------------------------
    # Case 4:
    # Unknown hallucinated evidence ID must fail.
    # -------------------------------------------------

    hallucinated_evidence = copy.deepcopy(
        valid_draft
    )

    hallucinated_evidence[
        "selected_projects"
    ][0][
        "bullets"
    ][0][
        "evidence_ids"
    ] = [
        "candidate_project_1_magic_result"
    ]

    expect_failure(
        name=(
            "Hallucinated evidence ID "
            "must be rejected"
        ),
        draft=(
            hallucinated_evidence
        ),
        expanded_sources=(
            expanded_sources
        ),
        expected_message_fragment=(
            "Unknown evidence ID"
        ),
    )

    print(
        "\n"
        + "#" * 100
    )

    print(
        "ALL CV STRUCTURAL "
        "VALIDATION TESTS PASSED"
    )

    print(
        "#" * 100
    )


if __name__ == "__main__":
    main()
