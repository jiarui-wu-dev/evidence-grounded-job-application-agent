import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

EXPERIENCES_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profile"
    / "experiences.json"
)


SECTION_ORDER = [
    "profile",
    "skills",
    "research_experience",
    "experience",
    "selected_projects",
    "education",
]


SOURCE_LIMITS = {
    "research_experience": {
        "max_bullets": 3,
        "max_words_per_bullet": 42,
    },

    "professional_experience": {
        "max_bullets": 2,
        "max_words_per_bullet": 36,
    },

    "project": {
        "max_bullets": 3,
        "max_words_per_bullet": 42,
    },
}


MAX_PROJECT_SOURCES = 2
MAX_RESEARCH_SOURCES = 1
MAX_PROFESSIONAL_SOURCES = 1


def _load_experiences() -> List[Dict[str, Any]]:

    with EXPERIENCES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):

        raise TypeError(
            "experiences.json must contain a list."
        )

    return data


def _add_limits(
    source: Dict[str, Any],
) -> Dict[str, Any]:

    source = deepcopy(source)

    entity_type = source.get(
        "entity_type",
        "project",
    )

    limits = SOURCE_LIMITS.get(
        entity_type,
        SOURCE_LIMITS["project"],
    )

    source[
        "max_bullets"
    ] = limits[
        "max_bullets"
    ]

    source[
        "max_words_per_bullet"
    ] = limits[
        "max_words_per_bullet"
    ]

    return source


def _select_core_sources(
    priority_sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Preserve strategy ranking while keeping the CV compact.

    Default one-page composition:
    - up to 2 technical projects
    - up to 1 research experience
    - up to 1 professional experience
    """

    selected = []

    project_count = 0
    research_count = 0
    professional_count = 0

    seen = set()

    for source in priority_sources:

        source_id = source.get(
            "source_id"
        )

        if (
            not source_id
            or source_id in seen
        ):
            continue

        entity_type = source.get(
            "entity_type",
            "project",
        )

        if entity_type == "project":

            if (
                project_count
                >= MAX_PROJECT_SOURCES
            ):
                continue

            project_count += 1

        elif (
            entity_type
            == "research_experience"
        ):

            if (
                research_count
                >= MAX_RESEARCH_SOURCES
            ):
                continue

            research_count += 1

        elif (
            entity_type
            == "professional_experience"
        ):

            if (
                professional_count
                >= MAX_PROFESSIONAL_SOURCES
            ):
                continue

            professional_count += 1

        normalized = deepcopy(
            source
        )

        normalized[
            "composition_role"
        ] = "core"

        selected.append(
            _add_limits(
                normalized
            )
        )

        seen.add(
            source_id
        )

    return selected


def _choose_professional_experience(
    already_selected: set,
) -> Dict[str, Any] | None:
    """
    Select one professional experience as commercial
    credibility without forcing it to compete with
    technical priority sources.

    Prefer verified evidence when available, but allow
    provisional experience when it is the best available
    professional record.
    """

    experiences = (
        _load_experiences()
    )

    candidates = [
        experience
        for experience in experiences
        if (
            experience.get(
                "entity_type"
            )
            == "professional_experience"
            and experience.get(
                "id"
            )
            not in already_selected
        )
    ]

    if not candidates:
        return None

    status_priority = {
        "verified": 0,
        "provisional": 1,
    }

    candidates.sort(
        key=lambda item: (
            status_priority.get(
                item.get(
                    "evidence_status",
                    "provisional",
                ),
                2,
            )
        )
    )

    experience = candidates[0]

    supplemental = {
        "source_id": experience[
            "id"
        ],

        "entity_type": (
            "professional_experience"
        ),

        "evidence_status": (
            experience.get(
                "evidence_status",
                "provisional",
            )
        ),

        "reason": (
            "Retained as professional-environment "
            "credibility without consuming a technical "
            "priority-source slot."
        ),

        "composition_role": (
            "supplemental_professional"
        ),
    }

    return _add_limits(
        supplemental
    )


def build_cv_composition(
    application_strategy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert technical source priorities into an actual
    one-page CV composition plan.

    Strategy decides relevance.
    Composition decides what appears in the resume.
    """

    priority_sources = (
        application_strategy.get(
            "priority_sources",
            [],
        )
    )

    core_sources = (
        _select_core_sources(
            priority_sources
        )
    )

    selected_ids = {
        source[
            "source_id"
        ]
        for source in core_sources
    }

    professional_already_present = any(
        source.get(
            "entity_type"
        )
        == "professional_experience"
        for source in core_sources
    )

    supplemental_sources = []

    if not professional_already_present:

        professional_source = (
            _choose_professional_experience(
                already_selected=(
                    selected_ids
                )
            )
        )

        if (
            professional_source
            is not None
        ):

            supplemental_sources.append(
                professional_source
            )

    all_sources = (
        core_sources
        + supplemental_sources
    )

    return {
        "section_order": (
            SECTION_ORDER
        ),

        "core_sources": (
            core_sources
        ),

        "supplemental_sources": (
            supplemental_sources
        ),

        "sources": (
            all_sources
        ),

        "limits": {
            "profile_max_words": 65,

            "max_project_sources": (
                MAX_PROJECT_SOURCES
            ),

            "max_research_sources": (
                MAX_RESEARCH_SOURCES
            ),

            "max_professional_sources": (
                MAX_PROFESSIONAL_SOURCES
            ),
        },
    }