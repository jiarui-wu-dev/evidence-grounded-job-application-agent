import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PROFILE_DIR = (
    PROJECT_ROOT
    / "data"
    / "candidate_profile"
)

PROJECTS_PATH = (
    PROFILE_DIR
    / "projects.json"
)

EXPERIENCES_PATH = (
    PROFILE_DIR
    / "experiences.json"
)


def _load_json(
    path: Path,
) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            f"{path.name} must contain "
            "a JSON list."
        )

    return data


def _build_project_documents(
    projects: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:
    documents = []

    for project in projects:
        project_id = project["id"]
        title = project["title"]

        summary = project.get(
            "summary",
            "",
        ).strip()

        if summary:
            documents.append({
                "evidence_id": (
                    f"{project_id}_summary"
                ),
                "project_id": (
                    project_id
                ),
                "project_title": title,
                "entity_type": "project",
                "evidence_status": (
                    project.get(
                        "evidence_status",
                        "verified",
                    )
                ),
                "type": "summary",
                "text": (
                    f"{title}. "
                    f"{summary}"
                ),
            })

        for index, evidence in enumerate(
            project.get(
                "evidence",
                [],
            )
        ):
            evidence = (
                str(evidence).strip()
            )

            if not evidence:
                continue

            documents.append({
                "evidence_id": (
                    f"{project_id}_"
                    f"evidence_{index}"
                ),
                "project_id": (
                    project_id
                ),
                "project_title": title,
                "entity_type": "project",
                "evidence_status": (
                    project.get(
                        "evidence_status",
                        "verified",
                    )
                ),
                "type": "evidence",
                "text": (
                    f"{title}. "
                    f"{evidence}"
                ),
            })

        skills = project.get(
            "skills",
            [],
        )

        if skills:
            skill_text = ", ".join(
                str(skill)
                for skill in skills
            )

            documents.append({
                "evidence_id": (
                    f"{project_id}_skills"
                ),
                "project_id": (
                    project_id
                ),
                "project_title": title,
                "entity_type": "project",
                "evidence_status": (
                    project.get(
                        "evidence_status",
                        "verified",
                    )
                ),
                "type": "skills",
                "text": (
                    f"{title}. "
                    f"Skills: {skill_text}"
                ),
            })

    return documents


def _build_experience_documents(
    experiences: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:
    documents = []

    for experience in experiences:
        experience_id = experience[
            "id"
        ]

        title = experience[
            "title"
        ]

        entity_type = experience.get(
            "entity_type",
            "experience",
        )

        evidence_status = (
            experience.get(
                "evidence_status",
                "verified",
            )
        )

        organization = (
            experience.get(
                "organization",
                "",
            ).strip()
        )

        unit = experience.get(
            "unit",
            "",
        ).strip()

        context_parts = [
            title,
        ]

        if organization:
            context_parts.append(
                organization
            )

        if unit:
            context_parts.append(
                unit
            )

        context = ". ".join(
            context_parts
        )

        summary = experience.get(
            "summary",
            "",
        ).strip()

        if summary:
            documents.append({
                "evidence_id": (
                    f"{experience_id}_"
                    "summary"
                ),
                # Compatibility alias:
                # existing retrievers currently
                # expect project_id/title.
                "project_id": (
                    experience_id
                ),
                "project_title": title,
                "entity_type": entity_type,
                "evidence_status": (
                    evidence_status
                ),
                "type": (
                    f"{entity_type}_summary"
                ),
                "text": (
                    f"{context}. "
                    f"{summary}"
                ),
            })

        for index, evidence in enumerate(
            experience.get(
                "evidence",
                [],
            )
        ):
            evidence = (
                str(evidence).strip()
            )

            if not evidence:
                continue

            documents.append({
                "evidence_id": (
                    f"{experience_id}_"
                    f"evidence_{index}"
                ),
                "project_id": (
                    experience_id
                ),
                "project_title": title,
                "entity_type": entity_type,
                "evidence_status": (
                    evidence_status
                ),
                "type": (
                    f"{entity_type}_evidence"
                ),
                "text": (
                    f"{context}. "
                    f"{evidence}"
                ),
            })

        skills = experience.get(
            "skills",
            [],
        )

        if skills:
            skill_text = ", ".join(
                str(skill)
                for skill in skills
            )

            documents.append({
                "evidence_id": (
                    f"{experience_id}_skills"
                ),
                "project_id": (
                    experience_id
                ),
                "project_title": title,
                "entity_type": entity_type,
                "evidence_status": (
                    evidence_status
                ),
                "type": (
                    f"{entity_type}_skills"
                ),
                "text": (
                    f"{context}. "
                    f"Skills: {skill_text}"
                ),
            })

    return documents


def load_profile_documents(
) -> List[Dict[str, Any]]:
    """
    Load candidate evidence from projects and
    experience records into one retrieval corpus.
    """

    projects = _load_json(
        PROJECTS_PATH
    )

    experiences = _load_json(
        EXPERIENCES_PATH
    )

    documents = []

    documents.extend(
        _build_project_documents(
            projects
        )
    )

    documents.extend(
        _build_experience_documents(
            experiences
        )
    )

    return documents


if __name__ == "__main__":
    documents = (
        load_profile_documents()
    )

    print(
        f"Loaded "
        f"{len(documents)} "
        f"candidate evidence units.\n"
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"{document['evidence_id']}"
        )

        print(
            f"    entity_type="
            f"{document['entity_type']}"
        )

        print(
            f"    evidence_status="
            f"{document['evidence_status']}"
        )

        print(
            f"    {document['text']}"
        )

        print()