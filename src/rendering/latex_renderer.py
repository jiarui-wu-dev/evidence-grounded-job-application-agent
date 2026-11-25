import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

STATIC_PROFILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profile"
    / "static_profile.json"
)


# ---------------------------------------------------------
# LaTeX escaping
# ---------------------------------------------------------

def latex_escape(
    value: Any,
) -> str:

    if value is None:
        return ""

    text = str(value)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    return text


# ---------------------------------------------------------
# Input loading
# ---------------------------------------------------------

def load_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def extract_cv(
    raw: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Accept either:

    1. a full LangGraph state containing `final_cv`
    2. a standalone final_cv object
    """

    if isinstance(raw, dict) and "final_status" in raw and raw["final_status"] != "grounded":
        raise ValueError("Only a grounded run can be rendered.")

    if (
        isinstance(raw, dict)
        and isinstance(
            raw.get("final_cv"),
            dict,
        )
    ):
        return raw["final_cv"]

    return raw


# ---------------------------------------------------------
# Basic rendering helpers
# ---------------------------------------------------------

def section_header(
    title: str,
) -> str:

    return (
        "\n"
        rf"\section*{{{latex_escape(title)}}}"
        "\n"
    )


def render_bullets(
    bullets: List[Dict[str, Any]],
) -> str:

    if not bullets:
        return ""

    lines = [
        r"\begin{itemize}",
    ]

    for bullet in bullets:

        text = bullet.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        lines.append(
            rf"\item {latex_escape(text)}"
        )

    lines.append(
        r"\end{itemize}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

def render_header(
    profile: Dict[str, Any],
) -> str:

    name = latex_escape(
        profile.get(
            "name",
            "",
        )
    )

    contact_parts = []

    for key in [
        "location",
        "email",
        "phone",
        "linkedin",
        "github",
    ]:

        value = profile.get(
            key,
            "",
        )

        if value:
            contact_parts.append(
                latex_escape(value)
            )

    contact_line = (
        r" \textbar{} ".join(
            contact_parts
        )
    )

    lines = [
        r"\begin{center}",
        rf"{{\LARGE \textbf{{{name}}}}}",
    ]

    if contact_line:

        lines.extend([
            r"\vspace{3pt}",
            contact_line,
        ])

    lines.append(
        r"\end{center}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------
# Profile
# ---------------------------------------------------------

def render_profile_section(
    cv: Dict[str, Any],
) -> str:

    profile = cv.get(
        "profile",
        {},
    )

    text = profile.get(
        "text",
        "",
    ).strip()

    if not text:
        return ""

    return (
        section_header(
            "Profile"
        )
        + latex_escape(text)
        + "\n"
    )


# ---------------------------------------------------------
# Skills
# ---------------------------------------------------------

def render_skills_section(
    cv: Dict[str, Any],
    static_profile: Dict[str, Any],
) -> str:

    skills = cv.get(
        "skills",
        [],
    )

    lines = []

    for group in skills:

        category = group.get(
            "category",
            "",
        )

        items = group.get(
            "items",
            [],
        )

        names = [
            item.get(
                "name",
                "",
            )
            for item in items
            if item.get(
                "name"
            )
        ]

        if not category or not names:
            continue

        lines.append(
            rf"\textbf{{{latex_escape(category)}:}} "
            + ", ".join(
                latex_escape(name)
                for name in names
            )
            + r"\\"
        )

    languages = static_profile.get(
        "languages",
        [],
    )

    if languages:

        formatted_languages = []

        for language in languages:

            name = language.get(
                "name",
                "",
            )

            level = language.get(
                "level",
                "",
            )

            if not name:
                continue

            if level:

                formatted_languages.append(
                    f"{name} ({level})"
                )

            else:

                formatted_languages.append(
                    name
                )

        if formatted_languages:

            lines.append(
                r"\textbf{Languages:} "
                + ", ".join(
                    latex_escape(item)
                    for item
                    in formatted_languages
                )
                + r"\\"
            )

    if not lines:
        return ""

    return (
        section_header(
            "Skills"
        )
        + "\n".join(lines)
        + "\n"
    )


# ---------------------------------------------------------
# Experience / Research Experience
# ---------------------------------------------------------

def render_experience_entry(
    entry: Dict[str, Any],
    static_profile: Dict[str, Any],
) -> str:

    source_id = entry.get(
        "source_id",
        "",
    )

    source_metadata = (
        static_profile
        .get(
            "source_metadata",
            {},
        )
        .get(
            source_id,
            {},
        )
    )

    title = latex_escape(
        entry.get(
            "title",
            "",
        )
    )

    organization = latex_escape(
        entry.get(
            "organization",
            "",
        )
    )

    unit = latex_escape(
        entry.get(
            "unit",
            "",
        )
    )

    period = latex_escape(
        source_metadata.get(
            "period",
            "",
        )
    )

    location = latex_escape(
        source_metadata.get(
            "location",
            "",
        )
    )

    lines = []

    if period:

        lines.append(
            rf"\textbf{{{title}}}"
            rf"\hfill {period}"
        )

    else:

        lines.append(
            rf"\textbf{{{title}}}"
        )

    organization_parts = [
        value
        for value in [
            organization,
            unit,
            location,
        ]
        if value
    ]

    if organization_parts:

        lines.append(
            r"\textit{"
            + r" \textbar{} ".join(
                organization_parts
            )
            + "}"
        )

    bullets = entry.get(
        "bullets",
        [],
    )

    bullet_text = render_bullets(
        bullets
    )

    if bullet_text:

        lines.append(
            bullet_text
        )

    return "\n".join(lines)


def render_experience_section(
    cv: Dict[str, Any],
    static_profile: Dict[str, Any],
    key: str,
    title: str,
) -> str:

    entries = cv.get(
        key,
        [],
    )

    if not entries:
        return ""

    rendered = [
        section_header(
            title
        )
    ]

    for index, entry in enumerate(
        entries
    ):

        if index > 0:
            rendered.append(
                r"\vspace{4pt}"
            )

        rendered.append(
            render_experience_entry(
                entry=entry,
                static_profile=(
                    static_profile
                ),
            )
        )

    return "\n".join(
        rendered
    )


# ---------------------------------------------------------
# Projects
# ---------------------------------------------------------

def render_project_entry(
    project: Dict[str, Any],
) -> str:

    title = latex_escape(
        project.get(
            "title",
            "",
        )
    )

    lines = [
        rf"\textbf{{{title}}}"
    ]

    bullets = project.get(
        "bullets",
        [],
    )

    bullet_text = render_bullets(
        bullets
    )

    if bullet_text:
        lines.append(
            bullet_text
        )

    return "\n".join(lines)


def render_projects_section(
    cv: Dict[str, Any],
) -> str:

    projects = cv.get(
        "selected_projects",
        [],
    )

    if not projects:
        return ""

    rendered = [
        section_header(
            "Selected Projects"
        )
    ]

    for index, project in enumerate(
        projects
    ):

        if index > 0:

            rendered.append(
                r"\vspace{4pt}"
            )

        rendered.append(
            render_project_entry(
                project
            )
        )

    return "\n".join(
        rendered
    )


# ---------------------------------------------------------
# Education
# ---------------------------------------------------------

def render_education_section(
    static_profile: Dict[str, Any],
) -> str:

    education = (
        static_profile.get(
            "education",
            [],
        )
    )

    if not education:
        return ""

    lines = [
        section_header(
            "Education"
        )
    ]

    for index, item in enumerate(
        education
    ):

        institution = latex_escape(
            item.get(
                "institution",
                "",
            )
        )

        location = latex_escape(
            item.get(
                "location",
                "",
            )
        )

        degree = latex_escape(
            item.get(
                "degree",
                "",
            )
        )

        period = latex_escape(
            item.get(
                "period",
                "",
            )
        )

        details = item.get(
            "details",
            [],
        )

        if index > 0:

            lines.append(
                r"\vspace{4pt}"
            )

        if period:

            lines.append(
                rf"\textbf{{{institution}}}"
                rf"\hfill {period}"
            )

        else:

            lines.append(
                rf"\textbf{{{institution}}}"
            )

        second_line_parts = [
            value
            for value in [
                degree,
                location,
            ]
            if value
        ]

        if second_line_parts:

            lines.append(
                r"\textit{"
                + r" \textbar{} ".join(
                    second_line_parts
                )
                + "}"
            )

        if details:

            lines.append(
                latex_escape(
                    " | ".join(
                        details
                    )
                )
            )

    return "\n".join(
        lines
    )


# ---------------------------------------------------------
# Document
# ---------------------------------------------------------

def render_document(
    cv: Dict[str, Any],
    static_profile: Dict[str, Any],
) -> str:

    body_parts = [
        render_header(
            static_profile
        ),

        render_profile_section(
            cv
        ),

        render_skills_section(
            cv,
            static_profile,
        ),

        render_experience_section(
            cv=cv,
            static_profile=static_profile,
            key="research_experience",
            title="Research Experience",
        ),

        render_experience_section(
            cv=cv,
            static_profile=static_profile,
            key="experience",
            title="Experience",
        ),

        render_projects_section(
            cv
        ),

        render_education_section(
            static_profile
        ),
    ]

    body = "\n".join(
        part
        for part in body_parts
        if part
    )

    template = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[
    margin=1.45cm
]{geometry}

\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{lmodern}
\usepackage[hidelinks]{hyperref}

\pagestyle{empty}

\setlength{\parindent}{0pt}
\setlength{\parskip}{2pt}

\setlist[itemize]{
    leftmargin=1.25em,
    itemsep=1.5pt,
    topsep=2pt,
    parsep=0pt
}

\titleformat{\section}
    {\large\bfseries}
    {}
    {0pt}
    {}
    [\titlerule]

\titlespacing*{\section}
    {0pt}
    {7pt}
    {3pt}

\begin{document}

__BODY__

\end{document}
"""

    return template.replace(
        "__BODY__",
        body,
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Render an evidence-grounded "
            "CV JSON object to ATS-friendly LaTeX."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to either a LangGraph "
            "full-state JSON file or a "
            "standalone final_cv JSON file."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Destination .tex file."
        ),
    )

    parser.add_argument(
        "--profile",
        default=str(
            STATIC_PROFILE_PATH
        ),
        help=(
            "Path to static_profile.json."
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    profile_path = Path(
        args.profile
    )

    raw = load_json(
        input_path
    )

    cv = extract_cv(
        raw
    )

    static_profile = load_json(
        profile_path
    )

    latex = render_document(
        cv=cv,
        static_profile=static_profile,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        latex,
        encoding="utf-8",
    )

    print("Rendered CV successfully.")

    print(
        "\nSections:"
    )

    print(
        "  Profile:",
        bool(
            cv.get("profile")
        ),
    )

    print(
        "  Skills:",
        len(
            cv.get(
                "skills",
                [],
            )
        ),
    )

    print(
        "  Research Experience:",
        len(
            cv.get(
                "research_experience",
                [],
            )
        ),
    )

    print(
        "  Experience:",
        len(
            cv.get(
                "experience",
                [],
            )
        ),
    )

    print(
        "  Selected Projects:",
        len(
            cv.get(
                "selected_projects",
                [],
            )
        ),
    )

    print(
        "  Education:",
        len(
            static_profile.get(
                "education",
                [],
            )
        ),
    )


if __name__ == "__main__":
    main()
