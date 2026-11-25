import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from src.llm.client import get_llm


# ============================================================
# Prompts
# ============================================================

PROFILE_PROMPT = """
You are an evidence-grounded CV tailoring system.

Generate ONLY:

1. a concise professional Profile;
2. a shortlist of relevant technical skills selected from
   candidate_skill_catalog.

Do NOT generate detailed skill objects.
Do NOT generate CV experience or project bullets.

GENERAL RULES

- Use only the supplied candidate evidence.
- Do not invent experience, technologies, outcomes, metrics,
  responsibilities, or achievements.
- Do not claim GAP skills as existing experience.
- Do not select anything listed in keywords_not_to_claim.
- Prefer directly SUPPORTED capabilities.
- Evidence marked provisional must not be the sole basis for
  a definitive capability claim.
- Do not claim unsupported years of experience.
- Do not claim leadership or ownership unless explicitly
  supported.

ACTIVITY EVIDENCE RULE

Skill-list evidence may support the name of a skill or
technology, but it does not by itself support claims that the
candidate built, designed, developed, implemented, applied,
conducted, or evaluated a workflow, pipeline, system, or model.

For any Profile claim describing a concrete activity,
workflow, project contribution, or result, cite at least one
detailed summary or evidence item that directly describes that
activity.

Examples:

- "experience with LoRA" may be supported by skill evidence;
- "applied LoRA-based fine-tuning" requires evidence that
  explicitly states the fine-tuning activity;
- "developed an ASR evaluation pipeline" requires evidence
  explicitly describing that pipeline.

Do not rely only on *_skills evidence for concrete activity
claims.


PROFILE

Write 2-3 concise sentences.

Position the candidate as a junior technical contributor with:

- supervised research experience;
- hands-on engineering experience;
- relevant professional experience when supported.

Do not frame the candidate merely as a student with projects.

Do not say the candidate develops or trains large language
models unless the evidence explicitly supports that.

Distinguish between:

- model development;
- fine-tuning;
- evaluation;
- workflow development;
- validation;
- failure analysis.

SKILL SELECTION

Select at most 12 skills.

Every selected skill MUST exactly match one string from
candidate_skill_catalog.

Do not rename skills.
Do not invent new skills.
Do not select a skill merely because it appears in the JD.

Return valid JSON only.
Do not use Markdown.
Do not add explanatory text.

Required format:

{
  "profile": {
    "text": "Professional profile.",
    "evidence_ids": [
      "evidence_id"
    ]
  },
  "selected_skill_names": [
    "Python",
    "Machine Learning"
  ]
}
"""


SOURCE_SECTION_PROMPT = """
You are an evidence-grounded CV bullet writer.

Generate CV content for ONE candidate source only.

The source may be:

- research_experience
- professional_experience
- project

Use only evidence supplied for this source.

COMPOSITION POLICY

The input may contain a "composition" object with:

- role
- max_bullets
- max_words_per_bullet
- evidence_statuses

You MUST obey these limits.

Generate no more than max_bullets bullets.

Each bullet MUST contain no more than
max_words_per_bullet words.

Prefer multiple concise bullets over combining many
independent engineering activities into one long bullet.

For a supplemental_professional source, emphasize concise
evidence of real professional or commercial software work.
Do not let it dominate the technical sources.

EVIDENCE STATUS POLICY

If the source contains provisional evidence:

Use only activities explicitly described by that evidence.

Do NOT add:
- metrics;
- business impact;
- technical outcomes;
- tools;
- responsibilities;
- scope;
- causal relationships;

unless they are explicitly present in the supplied evidence.

Do not mention internal labels such as:
- provisional
- evidence status
- legacy CV
- Candidate KB

in the generated CV text.

Write normal professional CV language while preserving the
actual meaning and confidence of the supplied evidence.

GENERAL RULES

- Generate concise bullets within the supplied composition limits.
- If no composition limits are supplied, generate 2-3 bullets,
  maximum 3.
- Every bullet must cite evidence_ids from this source.
- Never invent evidence IDs.
- Never use evidence from another source.
- Do not invent technologies, outcomes, metrics, deployment,
  customer usage, leadership, ownership, or business impact.
- Do not add dates.
- Preserve technical meaning.
- Prefer concise recruiter-readable engineering language.

BULLET STYLE

Use Problem / Goal -> Action / Method -> Result / Impact only
as an internal reasoning pattern.

Do NOT literally output labels such as:

- "Problem/Goal:"
- "Action/Method:"
- "Result/Impact:"

Do NOT output arrows such as "->".

Write each bullet as a natural professional CV sentence.

Include a Result / Impact only when the supplied evidence
explicitly supports that result or impact.

If the evidence supports only an action or method, stop after
the action or method. Do not invent an impact merely to
complete the pattern.

When multiple cited evidence items describe separate facts,
do not create a causal relationship between them unless the
evidence explicitly states that relationship.

Avoid causal connectors such as:

- resulting in
- leading to
- enabling
- thereby
- therefore
- through
- reducing
- improving
- ensuring
- achieving

when they would imply a causal relationship that is not
explicitly supported.

When a measured result is supported by a separate evidence item,
prefer neutral coordination such as "and achieved" rather than
participial constructions such as ", achieving ...".


when they would imply a causal relationship that is not
explicitly supported.

Separate independently supported facts using neutral
coordination such as "and" or separate clauses.

SKILL EVIDENCE RULE

Skill-list evidence may support that a technology, method, or
capability belongs to the source, but it does not by itself
prove that the skill was used for a specific activity,
experiment, result, or scenario.

Do not combine detailed activity evidence with skill-list
evidence using wording such as:

- applying
- using
- through
- with
- powered by

when that wording would imply that the listed skills were
specifically used to produce the cited activity or result.

For example:

Evidence A:
"Evaluated CWE-1243 and CWE-1302."

Evidence B:
"Skills: Python, LLM Evaluation, Failure Analysis."

Supported:
"Evaluated CWE-1243 and CWE-1302 scenarios."

Not automatically supported:
"Evaluated CWE-1243 and CWE-1302 using Python and LLM
evaluation."

Prefer leaving technologies in the Skills section when the
detailed activity evidence does not explicitly connect them
to the activity.

SOURCE TITLE RULE

A source or project title may provide general context about
the project, but it does not by itself prove the purpose,
cause, method, or outcome of a specific activity.

Do not infer relationships such as:

- "to adapt..."
- "to improve..."
- "for the purpose of..."
- "in order to..."

from the source title alone.

If the detailed evidence only states that an activity was
performed, describe the activity directly without adding an
inferred purpose.


Prefer:

- measured outcomes when explicitly supported;
- validated cases;
- rejected failure cases;
- evaluation results;
- engineering outcomes;
- failure analysis;
- automation;
- technical effects.

Avoid bullets that only list technologies.

RESEARCH EXPERIENCE

Do not rewrite:

"developed an LLM-assisted workflow"

as:

"developed or trained large language models".

For LLM-assisted research, emphasize when supported:

- LLM-generated code validation;
- automated evaluation;
- failure analysis;
- baseline-aware validation;
- behavioral correctness;
- security-semantic correctness;
- rejection of ineffective or invalid outputs.

ORGANIZATION METADATA

For organization and unit:

- copy them only when explicitly supported by evidence;
- otherwise return an empty string.

Return valid JSON only.
Do not use Markdown.
Do not add explanatory text.

Required format:

{
  "organization": "",
  "unit": "",
  "bullets": [
    {
      "text": "Evidence-grounded CV bullet.",
      "evidence_ids": [
        "evidence_id"
      ]
    }
  ]
}
"""


# ============================================================
# Skill categories
# ============================================================

SKILL_CATEGORIES = {
    "Programming": {
        "Python",
        "Java",
        "C",
        "C++",
        "SQL",
        "JavaScript",
    },

    "AI / ML": {
        "Machine Learning",
        "PyTorch",
        "Whisper",
        "Hugging Face",
        "PEFT",
        "LoRA",
        "ASR",
        "Fine-Tuning",
        "Model Evaluation",
        "Large Language Models",
        "LLM Evaluation",
        "Failure Analysis",
    },

    "Backend / Engineering": {
        "Spring Boot",
        "Spring Data JPA",
        "Spring Security",
        "REST APIs",
        "PostgreSQL",
        "Flyway",
        "Maven",
        "Docker",
        "Docker Compose",
        "Postman",
        "Swagger",
        "RBAC",
    },

    "Security / Hardware": {
        "Verilog",
        "Yosys",
        "Slang",
        "CWE",
        "Hardware Security",
        "Automated Validation",
    },
}


# ============================================================
# JSON parsing
# ============================================================

def parse_json_response(
    content: str,
) -> Dict[str, Any]:

    content = content.strip()

    if content.startswith("```json"):
        content = content[
            len("```json"):
        ]

    elif content.startswith("```"):
        content = content[
            len("```"):
        ]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    attempts = [
        content,
    ]

    first_brace = content.find("{")
    last_brace = content.rfind("}")

    if (
        first_brace != -1
        and last_brace != -1
        and last_brace > first_brace
    ):
        extracted = content[
            first_brace:
            last_brace + 1
        ]

        if extracted != content:
            attempts.append(
                extracted
            )

    last_error = None

    for attempt in attempts:

        try:
            result = json.loads(
                attempt
            )

            if isinstance(
                result,
                str,
            ):
                result = json.loads(
                    result
                )

            if not isinstance(
                result,
                dict,
            ):
                raise ValueError(
                    "LLM response must be "
                    "a JSON object."
                )

            return result

        except (
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error

        try:
            result = json.loads(
                attempt,
                strict=False,
            )

            if isinstance(
                result,
                str,
            ):
                result = json.loads(
                    result,
                    strict=False,
                )

            if not isinstance(
                result,
                dict,
            ):
                raise ValueError(
                    "LLM response must be "
                    "a JSON object."
                )

            return result

        except (
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error

    raise ValueError(
        "Failed to parse LLM response "
        f"as JSON: {type(last_error).__name__}"
    )


# ============================================================
# Evidence helpers
# ============================================================

def build_evidence_map(
    expanded_sources: List[
        Dict[str, Any]
    ],
) -> Dict[str, Dict[str, Any]]:

    evidence_map = {}

    for source in expanded_sources:

        for evidence in source[
            "evidence"
        ]:

            evidence_id = evidence[
                "evidence_id"
            ]

            if evidence_id in evidence_map:
                raise ValueError(
                    "Duplicate evidence ID: "
                    f"{evidence_id}"
                )

            evidence_map[
                evidence_id
            ] = {
                **evidence,

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
            }

    return evidence_map


def build_profile_evidence(
    expanded_sources: List[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:

    result = []

    for source in expanded_sources:

        for evidence in source[
            "evidence"
        ]:

            result.append({
                "evidence_id": (
                    evidence[
                        "evidence_id"
                    ]
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

                "evidence_status": (
                    evidence.get(
                        "evidence_status",
                        "verified",
                    )
                ),

                "text": (
                    evidence[
                        "text"
                    ]
                ),
            })

    return result


# ============================================================
# Deterministic skill catalogue
# ============================================================

def build_skill_catalog(
    expanded_sources: List[
        Dict[str, Any]
    ],
) -> Dict[str, Dict[str, Any]]:
    """
    Extract structured skills from Candidate KB evidence
    whose text contains 'Skills:'.

    Each skill records the evidence IDs that explicitly
    support it.
    """

    catalog = {}

    for source in expanded_sources:

        for evidence in source[
            "evidence"
        ]:

            text = evidence.get(
                "text",
                "",
            )

            if "Skills:" not in text:
                continue

            skill_text = text.split(
                "Skills:",
                1,
            )[1]

            raw_skills = skill_text.split(
                ","
            )

            for raw_skill in raw_skills:

                skill = raw_skill.strip()

                if not skill:
                    continue

                if skill not in catalog:

                    catalog[
                        skill
                    ] = {
                        "name": skill,
                        "evidence_ids": [],
                        "statuses": [],
                    }

                catalog[
                    skill
                ][
                    "evidence_ids"
                ].append(
                    evidence[
                        "evidence_id"
                    ]
                )

                catalog[
                    skill
                ][
                    "statuses"
                ].append(
                    evidence.get(
                        "evidence_status",
                        "verified",
                    )
                )

    return catalog


def _normalize(
    value: str,
) -> str:

    return (
        value.lower()
        .strip()
        .replace("-", " ")
        .replace("_", " ")
    )


def _skill_conflicts_with_blocked(
    skill: str,
    blocked: List[str],
) -> bool:

    normalized_skill = _normalize(
        skill
    )

    for item in blocked:

        normalized_item = _normalize(
            item
        )

        if not normalized_item:
            continue

        if (
            normalized_skill
            == normalized_item
        ):
            return True

        if (
            normalized_skill
            in normalized_item
        ):
            return True

        if (
            normalized_item
            in normalized_skill
        ):
            return True

    return False


def get_skill_category(
    skill_name: str,
) -> str:

    for (
        category,
        skills,
    ) in SKILL_CATEGORIES.items():

        if skill_name in skills:
            return category

    return "Technical"


def build_deterministic_skills(
    selected_skill_names: List[str],
    skill_catalog: Dict[
        str,
        Dict[str, Any]
    ],
    blocked_keywords: List[str],
) -> List[Dict[str, Any]]:
    """
    Convert the LLM's short skill-name selection into
    fully evidence-cited Skills sections.

    Unknown, blocked, or provisional-only skills are
    dropped deterministically.
    """

    grouped = {}

    seen = set()

    # Case-insensitive lookup while preserving the exact
    # Candidate KB spelling.
    catalog_lookup = {
        name.lower(): name
        for name in skill_catalog
    }

    for requested_name in (
        selected_skill_names
    ):

        if not isinstance(
            requested_name,
            str,
        ):
            continue

        requested_name = (
            requested_name.strip()
        )

        canonical_name = (
            catalog_lookup.get(
                requested_name.lower()
            )
        )

        # LLM invented or renamed the skill.
        if canonical_name is None:
            continue

        if canonical_name in seen:
            continue

        skill_info = skill_catalog[
            canonical_name
        ]

        # A definitive Skills section entry requires at
        # least one verified evidence item.
        has_verified = any(
            status == "verified"
            for status
            in skill_info[
                "statuses"
            ]
        )

        if not has_verified:
            continue

        if _skill_conflicts_with_blocked(
            canonical_name,
            blocked_keywords,
        ):
            continue

        category = get_skill_category(
            canonical_name
        )

        grouped.setdefault(
            category,
            [],
        )

        grouped[
            category
        ].append({
            "name": (
                canonical_name
            ),

            "evidence_ids": list(
                dict.fromkeys(
                    skill_info[
                        "evidence_ids"
                    ]
                )
            ),
        })

        seen.add(
            canonical_name
        )

    category_order = [
        "Programming",
        "AI / ML",
        "Backend / Engineering",
        "Security / Hardware",
        "Technical",
    ]

    result = []

    for category in category_order:

        items = grouped.get(
            category,
            [],
        )

        if not items:
            continue

        result.append({
            "category": category,
            "items": items,
        })

    return result


# ============================================================
# Evidence ID validation
# ============================================================

def _validate_evidence_ids(
    evidence_ids: List[str],
    evidence_map: Dict[
        str,
        Dict[str, Any]
    ],
    context: str,
    required_source_id: Optional[str] = None,
) -> None:

    for evidence_id in evidence_ids:

        if evidence_id not in evidence_map:
            raise ValueError(
                f"Unknown evidence ID "
                f"in {context}: "
                f"{evidence_id}"
            )

        if required_source_id is not None:

            actual_source_id = (
                evidence_map[
                    evidence_id
                ][
                    "source_id"
                ]
            )

            if (
                actual_source_id
                != required_source_id
            ):
                raise ValueError(
                    "Cross-source evidence "
                    f"citation in {context}: "
                    f"{evidence_id} belongs "
                    f"to {actual_source_id}, "
                    f"not {required_source_id}."
                )


# ============================================================
# Structural validation
# ============================================================

def validate_draft_structure(
    draft: Dict[str, Any],
    expanded_sources: List[
        Dict[str, Any]
    ],
) -> None:

    evidence_map = build_evidence_map(
        expanded_sources
    )

    source_map = {
        source[
            "source_id"
        ]: source
        for source
        in expanded_sources
    }

    required_sections = {
        "profile",
        "skills",
        "research_experience",
        "experience",
        "selected_projects",
    }

    missing_sections = (
        required_sections
        - set(
            draft.keys()
        )
    )

    if missing_sections:
        raise ValueError(
            "CV draft is missing sections: "
            + ", ".join(
                sorted(
                    missing_sections
                )
            )
        )

    # --------------------------------------------------------
    # Profile
    # --------------------------------------------------------

    profile = draft[
        "profile"
    ]

    if not isinstance(
        profile,
        dict,
    ):
        raise ValueError(
            "profile must be an object."
        )

    profile_text = profile.get(
        "text"
    )

    if (
        not isinstance(
            profile_text,
            str,
        )
        or not profile_text.strip()
    ):
        raise ValueError(
            "profile.text must be "
            "a non-empty string."
        )

    profile_evidence_ids = (
        profile.get(
            "evidence_ids",
            [],
        )
    )

    if not profile_evidence_ids:
        raise ValueError(
            "Profile must cite at least "
            "one evidence ID."
        )

    _validate_evidence_ids(
        profile_evidence_ids,
        evidence_map,
        "profile",
    )

    # --------------------------------------------------------
    # Skills
    # --------------------------------------------------------

    skills = draft[
        "skills"
    ]

    if not isinstance(
        skills,
        list,
    ):
        raise ValueError(
            "skills must be a list."
        )

    for category in skills:

        if not isinstance(
            category,
            dict,
        ):
            raise ValueError(
                "Each skills category "
                "must be an object."
            )

        if not isinstance(
            category.get(
                "category"
            ),
            str,
        ):
            raise ValueError(
                "Skill category must "
                "be a string."
            )

        items = category.get(
            "items",
            [],
        )

        if not isinstance(
            items,
            list,
        ):
            raise ValueError(
                "Skill items must "
                "be a list."
            )

        for skill in items:

            skill_name = skill.get(
                "name"
            )

            if not isinstance(
                skill_name,
                str,
            ):
                raise ValueError(
                    "Skill name must "
                    "be a string."
                )

            evidence_ids = (
                skill.get(
                    "evidence_ids",
                    [],
                )
            )

            if not evidence_ids:
                raise ValueError(
                    f"Skill {skill_name} "
                    "must cite evidence."
                )

            _validate_evidence_ids(
                evidence_ids,
                evidence_map,
                f"skill {skill_name}",
            )

    # --------------------------------------------------------
    # Source sections
    # --------------------------------------------------------

    section_entity_types = {
        "research_experience": (
            "research_experience"
        ),

        "experience": (
            "professional_experience"
        ),

        "selected_projects": (
            "project"
        ),
    }

    used_source_ids = set()

    for (
        section_name,
        expected_entity_type,
    ) in section_entity_types.items():

        entries = draft[
            section_name
        ]

        if not isinstance(
            entries,
            list,
        ):
            raise ValueError(
                f"{section_name} must "
                "be a list."
            )

        for entry in entries:

            if not isinstance(
                entry,
                dict,
            ):
                raise ValueError(
                    f"Each {section_name} "
                    "entry must be an object."
                )

            source_id = entry.get(
                "source_id"
            )

            if source_id not in source_map:
                raise ValueError(
                    "Unknown source ID "
                    f"in {section_name}: "
                    f"{source_id}"
                )

            actual_entity_type = (
                source_map[
                    source_id
                ][
                    "entity_type"
                ]
            )

            if (
                actual_entity_type
                != expected_entity_type
            ):
                raise ValueError(
                    f"Source {source_id} "
                    f"has entity_type "
                    f"{actual_entity_type} "
                    f"but was placed in "
                    f"{section_name}."
                )

            if source_id in used_source_ids:
                raise ValueError(
                    "Candidate source appears "
                    "more than once: "
                    f"{source_id}"
                )

            used_source_ids.add(
                source_id
            )

            bullets = entry.get(
                "bullets",
                [],
            )

            if (
                not isinstance(
                    bullets,
                    list,
                )
                or not bullets
            ):
                raise ValueError(
                    f"Source {source_id} must "
                    "contain at least one bullet."
                )

            for bullet in bullets:

                text = bullet.get(
                    "text"
                )

                if (
                    not isinstance(
                        text,
                        str,
                    )
                    or not text.strip()
                ):
                    raise ValueError(
                        f"Invalid bullet text "
                        f"for {source_id}."
                    )

                evidence_ids = (
                    bullet.get(
                        "evidence_ids",
                        [],
                    )
                )

                if not evidence_ids:
                    raise ValueError(
                        "Every CV bullet must "
                        "cite evidence: "
                        f"{source_id}"
                    )

                _validate_evidence_ids(
                    evidence_ids,
                    evidence_map,
                    (
                        f"{section_name} "
                        f"{source_id}"
                    ),
                    required_source_id=(
                        source_id
                    ),
                )


def validate_source_composition_result(
    result: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    """
    Enforce deterministic CV composition limits
    after source generation.
    """

    bullets = result.get(
        "bullets"
    )

    if (
        not isinstance(
            bullets,
            list,
        )
        or not bullets
    ):
        raise ValueError(
            "Source result must contain "
            "a non-empty bullets list."
        )

    composition = payload.get(
        "composition",
        {},
    )

    max_bullets = composition.get(
        "max_bullets"
    )

    max_words = composition.get(
        "max_words_per_bullet"
    )

    if (
        isinstance(
            max_bullets,
            int,
        )
        and max_bullets > 0
        and len(bullets) > max_bullets
    ):
        raise ValueError(
            f"Generated {len(bullets)} bullets, "
            f"but maximum is {max_bullets}."
        )

    for index, bullet in enumerate(
        bullets,
        start=1,
    ):

        if not isinstance(
            bullet,
            dict,
        ):
            raise ValueError(
                f"Bullet {index} must "
                "be an object."
            )

        text = bullet.get(
            "text",
            "",
        )

        if (
            not isinstance(
                text,
                str,
            )
            or not text.strip()
        ):
            raise ValueError(
                f"Bullet {index} has "
                "no valid text."
            )

        if (
            isinstance(
                max_words,
                int,
            )
            and max_words > 0
        ):

            word_count = len(
                text.split()
            )

            if (
                word_count
                > max_words
            ):
                raise ValueError(
                    f"Bullet {index} has "
                    f"{word_count} words; "
                    f"maximum is {max_words}."
                )


# ============================================================
# Small LLM call
# ============================================================

def invoke_json_with_retry(
    llm,
    system_prompt: str,
    payload: Dict[str, Any],
    label: str,
    max_attempts: int = 2,
    result_validator=None,
) -> Dict[str, Any]:

    last_error = None

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        response = llm.invoke([
            SystemMessage(
                content=system_prompt
            ),

            HumanMessage(
                content=json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                )
            ),
        ])

        content = response.content

        if not isinstance(
            content,
            str,
        ):
            content = str(
                content
            )

        content = content.strip()

        if not content:

            last_error = ValueError(
                "LLM returned empty content."
            )

            continue

        try:

            result = parse_json_response(
                content
            )

            if (
                result_validator
                is not None
            ):
                result_validator(
                    result,
                    payload,
                )

            return result

        except ValueError as error:

            last_error = error

    raise ValueError(
        f"{label} failed after "
        f"{max_attempts} attempts. "
        f"Last error: {type(last_error).__name__}"
    )


# ============================================================
# CV generation
# ============================================================

def build_cv_draft(
    jd_analysis: Dict[str, Any],
    match_gap_result: Dict[str, Any],
    application_strategy: Dict[str, Any],
    expanded_sources: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:

    llm = get_llm(
        temperature=0.0
    )

    # ========================================================
    # Step 1:
    # Build deterministic skill catalogue
    # ========================================================

    skill_catalog = build_skill_catalog(
        expanded_sources
    )

    candidate_skill_names = list(
        skill_catalog.keys()
    )

    # ========================================================
    # Step 2:
    # Generate ONLY Profile + short skill-name selection
    # ========================================================

    profile_payload = {
        "job_title": (
            jd_analysis.get(
                "job_title"
            )
        ),

        "requirements": (
            match_gap_result.get(
                "requirements",
                [],
            )
        ),

        "positioning_hint": (
            application_strategy.get(
                "positioning",
                "",
            )
        ),

        "cv_keywords_to_emphasize": (
            application_strategy.get(
                "cv_keywords_to_emphasize",
                [],
            )
        ),

        "keywords_to_frame_carefully": (
            application_strategy.get(
                "keywords_to_frame_carefully",
                [],
            )
        ),

        "keywords_not_to_claim": (
            application_strategy.get(
                "keywords_not_to_claim",
                [],
            )
        ),

        "candidate_skill_catalog": (
            candidate_skill_names
        ),

        "available_evidence": (
            build_profile_evidence(
                expanded_sources
            )
        ),
    }

    profile_result = (
        invoke_json_with_retry(
            llm=llm,
            system_prompt=(
                PROFILE_PROMPT
            ),
            payload=(
                profile_payload
            ),
            label="profile",
            max_attempts=2,
        )
    )

    if (
        "profile"
        not in profile_result
    ):
        raise ValueError(
            "Profile generation did not "
            "return 'profile'."
        )

    selected_skill_names = (
        profile_result.get(
            "selected_skill_names",
            [],
        )
    )

    if not isinstance(
        selected_skill_names,
        list,
    ):
        raise ValueError(
            "selected_skill_names must "
            "be a list."
        )

    skills = build_deterministic_skills(
        selected_skill_names=(
            selected_skill_names
        ),
        skill_catalog=(
            skill_catalog
        ),
        blocked_keywords=(
            application_strategy.get(
                "keywords_not_to_claim",
                [],
            )
        ),
    )

    draft = {
        "profile": (
            profile_result[
                "profile"
            ]
        ),

        "skills": skills,

        "research_experience": [],

        "experience": [],

        "selected_projects": [],
    }

    # ========================================================
    # Step 3:
    # Generate each source independently
    # ========================================================

    for source in expanded_sources:

        source_id = source[
            "source_id"
        ]

        entity_type = source[
            "entity_type"
        ]

        source_payload = {
            "job_title": (
                jd_analysis.get(
                    "job_title"
                )
            ),

            "source_id": (
                source_id
            ),

            "source_title": (
                source[
                    "source_title"
                ]
            ),

            "entity_type": (
                entity_type
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

        generated_source = (
            invoke_json_with_retry(
                llm=llm,
                system_prompt=(
                    SOURCE_SECTION_PROMPT
                ),
                payload=(
                    source_payload
                ),
                label=(
                    f"source {source_id}"
                ),
                max_attempts=2,
                result_validator=(
                    validate_source_composition_result
                ),
            )
        )

        bullets = generated_source.get(
            "bullets"
        )

        if (
            not isinstance(
                bullets,
                list,
            )
            or not bullets
        ):
            raise ValueError(
                f"Source {source_id} did "
                "not return valid bullets."
            )

        entry = {
            "source_id": (
                source_id
            ),

            "title": (
                source[
                    "source_title"
                ]
            ),

            "bullets": (
                bullets
            ),
        }

        # ----------------------------------------------------
        # Deterministic section routing
        # ----------------------------------------------------

        if (
            entity_type
            == "research_experience"
        ):

            entry[
                "organization"
            ] = generated_source.get(
                "organization",
                "",
            )

            entry[
                "unit"
            ] = generated_source.get(
                "unit",
                "",
            )

            draft[
                "research_experience"
            ].append(
                entry
            )

        elif (
            entity_type
            == "professional_experience"
        ):

            entry[
                "organization"
            ] = generated_source.get(
                "organization",
                "",
            )

            draft[
                "experience"
            ].append(
                entry
            )

        elif (
            entity_type
            == "project"
        ):

            draft[
                "selected_projects"
            ].append(
                entry
            )

        else:
            raise ValueError(
                "Unknown entity_type "
                f"for source {source_id}: "
                f"{entity_type}"
            )

    # ========================================================
    # Step 4:
    # Structural validation
    # ========================================================

    validate_draft_structure(
        draft=draft,
        expanded_sources=(
            expanded_sources
        ),
    )

    draft[
        "job_title"
    ] = jd_analysis.get(
        "job_title"
    )

    return draft