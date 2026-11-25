import json
import re
from typing import Any, Dict

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from src.llm.client import get_llm


SYSTEM_PROMPT = """
You are an evidence-grounded job application strategist.

Your task is to create an application strategy using:

1. structured job requirements,
2. candidate evidence,
3. match/gap analysis.

You must NOT invent candidate experience.

A retrieved item is not automatically valid evidence.
Use the match/gap analysis as the primary grounding signal.

Your goal is to help position the candidate as a junior
technical contributor with defensible hands-on experience,
not as a collection of unrelated student projects.

Candidate evidence may come from different source types:

- research_experience
- professional_experience
- project

Preserve these distinctions when selecting important CV
content.

For the strategy:

- prioritize directly supported requirements;
- identify the strongest candidate evidence;
- distinguish real strengths from transferable experience;
- never recommend claiming GAP skills as existing experience;
- PARTIAL skills may be framed as adjacent or transferable,
  but never as fully demonstrated;
- GAP skills may be listed as learning priorities, but must
  not be added to the CV as existing skills;

- for GAP requirements, actions must be future-facing;
- recommend learning, building, or practicing the missing
  skill before or during the application process;
- never recommend highlighting, showcasing, mentioning, or
  claiming existing coursework, projects, tutorials,
  certifications, experiments, or experience for a GAP
  unless such evidence is explicitly provided;

- prefer candidate sources that support multiple important
  job requirements;
- prefer concrete outcomes and technical evidence over
  generic skill lists;

- for priority_strengths and transferable_strengths,
  evidence_ids must be selected ONLY from the evidence_ids
  already assigned to that same requirement in the
  match/gap analysis;
- do not cite other retrieved evidence even if it appears
  semantically related;

- never use leadership or ownership language such as
  "led", "managed", "supervised", "owned", or
  "spearheaded" unless the provided evidence explicitly
  supports that claim;
- describe individual technical contribution using verbs
  actually supported by the evidence, such as developed,
  implemented, designed, evaluated, validated, analyzed,
  or fine-tuned;

- do not describe building, training, or developing a model
  when the evidence only supports building a workflow around
  that model;
- distinguish model development from model evaluation,
  adaptation, validation, orchestration, and application
  development.

Return JSON only.

Required format:

{
  "positioning": "Short description of how the candidate should be positioned.",

  "priority_strengths": [
    {
      "requirement_id": "must_1",
      "reason": "Why this should be emphasized.",
      "evidence_ids": ["evidence_id"]
    }
  ],

  "priority_sources": [
    {
      "source_id": "source_id",
      "entity_type": "research_experience",
      "reason": "Why this candidate source should receive CV space."
    }
  ],

  "transferable_strengths": [
    {
      "requirement_id": "must_2",
      "reason": "What related experience exists and what is still missing.",
      "evidence_ids": ["evidence_id"]
    }
  ],

  "gaps": [
    {
      "requirement_id": "must_3",
      "severity": "HIGH",
      "action": "How the gap should be handled honestly."
    }
  ],

  "cv_keywords_to_emphasize": [
    "keyword"
  ]
}
"""


HIGH_RISK_CLAIM_TERMS = {
    "led",
    "managed",
    "supervised",
    "owned",
    "spearheaded",
}


def parse_json_response(
    content: str,
) -> Dict[str, Any]:
    """
    Parse an LLM JSON response with lightweight
    recovery for common formatting problems.
    """

    content = content.strip()

    # Remove Markdown code fences.
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

    # Sometimes the model adds explanatory text
    # before or after the JSON object.
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

        # First try normal strict JSON.
        try:
            result = json.loads(
                attempt
            )

            # Some models return a JSON string
            # containing another JSON object.
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
                    "LLM JSON response must "
                    "be an object."
                )

            return result

        except (
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error

        # Fallback for raw control characters
        # occasionally emitted by LLMs.
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
                    "LLM JSON response must "
                    "be an object."
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


def sanitize_strategy_evidence(
    strategy: Dict[str, Any],
    match_gap_result: Dict[str, Any],
) -> None:
    """
    Remove evidence citations that were not validated
    for the corresponding requirement by Match/Gap.

    If a strategy item has no valid evidence remaining,
    remove that strategy item instead of crashing the
    entire pipeline.

    Unknown requirement IDs are deliberately preserved
    so validate_strategy() can reject them as structural
    errors.
    """

    requirement_map = {
        item["requirement_id"]: item
        for item in match_gap_result[
            "requirements"
        ]
    }

    sections = (
        "priority_strengths",
        "transferable_strengths",
    )

    for section in sections:
        cleaned_items = []

        for item in strategy.get(
            section,
            [],
        ):
            requirement_id = item.get(
                "requirement_id"
            )

            if (
                requirement_id
                not in requirement_map
            ):
                cleaned_items.append(
                    item
                )
                continue

            validated_evidence_ids = set(
                requirement_map[
                    requirement_id
                ].get(
                    "evidence_ids",
                    [],
                )
            )

            original_evidence_ids = (
                item.get(
                    "evidence_ids",
                    [],
                )
            )

            filtered_evidence_ids = [
                evidence_id
                for evidence_id
                in original_evidence_ids
                if evidence_id
                in validated_evidence_ids
            ]

            # Keep only strategy items that still
            # have at least one validated citation.
            if filtered_evidence_ids:
                item[
                    "evidence_ids"
                ] = filtered_evidence_ids

                cleaned_items.append(
                    item
                )

        strategy[
            section
        ] = cleaned_items


def validate_strategy(
    strategy: Dict[str, Any],
    match_gap_result: Dict[str, Any],
    evidence_package: Dict[str, Any],
) -> None:
    """
    Validate requirement references, evidence citations,
    and selected candidate source metadata.
    """

    requirement_map = {
        item["requirement_id"]: item
        for item in match_gap_result[
            "requirements"
        ]
    }

    allowed_evidence_ids = {
        item["evidence_id"]
        for item in evidence_package[
            "evidence"
        ]
    }

    # project_id is currently the compatibility field
    # used for every candidate source ID, including:
    #
    # project
    # research_experience
    # professional_experience
    source_map = {}

    for item in evidence_package[
        "evidence"
    ]:
        source_id = item[
            "project_id"
        ]

        entity_type = item.get(
            "entity_type",
            "project",
        )

        if (
            source_id in source_map
            and source_map[
                source_id
            ] != entity_type
        ):
            raise ValueError(
                "Inconsistent entity_type "
                f"for source {source_id}."
            )

        source_map[
            source_id
        ] = entity_type

    # -------------------------------------------------
    # Validate priority strengths
    # -------------------------------------------------

    for strength in strategy.get(
        "priority_strengths",
        [],
    ):
        requirement_id = strength[
            "requirement_id"
        ]

        if requirement_id not in requirement_map:
            raise ValueError(
                f"Unknown requirement ID: "
                f"{requirement_id}"
            )

        requirement = requirement_map[
            requirement_id
        ]

        if (
            requirement["status"]
            != "SUPPORTED"
        ):
            raise ValueError(
                "Only SUPPORTED requirements "
                "may appear in priority_strengths: "
                f"{requirement_id}"
            )

        requirement_evidence_ids = set(
            requirement[
                "evidence_ids"
            ]
        )

        strength_evidence_ids = (
            strength.get(
                "evidence_ids",
                [],
            )
        )

        if not strength_evidence_ids:
            raise ValueError(
                "A priority strength must "
                "reference at least one "
                "validated evidence ID."
            )

        for evidence_id in (
            strength_evidence_ids
        ):
            if (
                evidence_id
                not in allowed_evidence_ids
            ):
                raise ValueError(
                    f"Unknown evidence ID: "
                    f"{evidence_id}"
                )

            if (
                evidence_id
                not in requirement_evidence_ids
            ):
                raise ValueError(
                    "Strategy cited evidence "
                    "that was not validated "
                    f"for {requirement_id}: "
                    f"{evidence_id}"
                )

    # -------------------------------------------------
    # Validate transferable strengths
    # -------------------------------------------------

    for item in strategy.get(
        "transferable_strengths",
        [],
    ):
        requirement_id = item[
            "requirement_id"
        ]

        if requirement_id not in requirement_map:
            raise ValueError(
                f"Unknown requirement ID: "
                f"{requirement_id}"
            )

        requirement = requirement_map[
            requirement_id
        ]

        if (
            requirement["status"]
            != "PARTIAL"
        ):
            raise ValueError(
                "Only PARTIAL requirements "
                "may appear in "
                "transferable_strengths: "
                f"{requirement_id}"
            )

        requirement_evidence_ids = set(
            requirement[
                "evidence_ids"
            ]
        )

        transferable_evidence_ids = (
            item.get(
                "evidence_ids",
                [],
            )
        )

        if not transferable_evidence_ids:
            raise ValueError(
                "A transferable strength must "
                "reference at least one "
                "validated evidence ID."
            )

        for evidence_id in (
            transferable_evidence_ids
        ):
            if (
                evidence_id
                not in allowed_evidence_ids
            ):
                raise ValueError(
                    f"Unknown evidence ID: "
                    f"{evidence_id}"
                )

            if (
                evidence_id
                not in requirement_evidence_ids
            ):
                raise ValueError(
                    "Strategy cited evidence "
                    "that was not validated "
                    f"for {requirement_id}: "
                    f"{evidence_id}"
                )

    # -------------------------------------------------
    # Validate gaps
    # -------------------------------------------------

    for gap in strategy.get(
        "gaps",
        [],
    ):
        requirement_id = gap[
            "requirement_id"
        ]

        if requirement_id not in requirement_map:
            raise ValueError(
                f"Unknown requirement ID: "
                f"{requirement_id}"
            )

        if (
            requirement_map[
                requirement_id
            ]["status"]
            != "GAP"
        ):
            raise ValueError(
                "Only GAP requirements "
                "may appear in gaps: "
                f"{requirement_id}"
            )

    # -------------------------------------------------
    # Validate priority sources
    # -------------------------------------------------

    for source in strategy.get(
        "priority_sources",
        [],
    ):
        source_id = source.get(
            "source_id"
        )

        entity_type = source.get(
            "entity_type"
        )

        if source_id not in source_map:
            raise ValueError(
                f"Unknown source ID: "
                f"{source_id}"
            )

        expected_entity_type = (
            source_map[
                source_id
            ]
        )

        if (
            entity_type
            != expected_entity_type
        ):
            raise ValueError(
                "Incorrect entity_type for "
                f"{source_id}: expected "
                f"{expected_entity_type}, "
                f"got {entity_type}"
            )


def _contains_term(
    text: str,
    term: str,
) -> bool:
    """
    Match a high-risk term as a whole word.
    """

    return bool(
        re.search(
            rf"\b{re.escape(term)}\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _validate_claim_text(
    text: str,
    evidence_text: str,
    context: str,
) -> None:
    """
    Reject high-risk language unless it also
    appears in the evidence supporting that claim.
    """

    for term in HIGH_RISK_CLAIM_TERMS:
        if (
            _contains_term(
                text,
                term,
            )
            and not _contains_term(
                evidence_text,
                term,
            )
        ):
            raise ValueError(
                "Unsupported high-risk claim "
                f"detected in {context}: "
                f"'{term}'"
            )


def validate_high_risk_claims(
    strategy: Dict[str, Any],
    evidence_package: Dict[str, Any],
) -> None:
    """
    Reject unsupported leadership or ownership
    language in generated strategy text.

    Evidence is checked at the most specific
    available level:

    - positioning -> all retrieved evidence
    - priority strength -> cited evidence IDs
    - transferable strength -> cited evidence IDs
    - priority source -> evidence belonging to source
    """

    evidence_map = {
        item["evidence_id"]: item[
            "text"
        ]
        for item in evidence_package[
            "evidence"
        ]
    }

    source_evidence_map = {}

    for item in evidence_package[
        "evidence"
    ]:
        source_id = item[
            "project_id"
        ]

        source_evidence_map.setdefault(
            source_id,
            [],
        )

        source_evidence_map[
            source_id
        ].append(
            item["text"]
        )

    all_evidence_text = " ".join(
        evidence_map.values()
    )

    # -------------------------------------------------
    # Positioning
    # -------------------------------------------------

    positioning = strategy.get(
        "positioning"
    )

    if isinstance(
        positioning,
        str,
    ):
        _validate_claim_text(
            text=positioning,
            evidence_text=(
                all_evidence_text
            ),
            context="positioning",
        )

    # -------------------------------------------------
    # Priority strengths
    # -------------------------------------------------

    for strength in strategy.get(
        "priority_strengths",
        [],
    ):
        reason = strength.get(
            "reason"
        )

        if not isinstance(
            reason,
            str,
        ):
            continue

        cited_evidence_text = " ".join(
            evidence_map[
                evidence_id
            ]
            for evidence_id in strength.get(
                "evidence_ids",
                [],
            )
            if evidence_id in evidence_map
        )

        _validate_claim_text(
            text=reason,
            evidence_text=(
                cited_evidence_text
            ),
            context=(
                "priority strength "
                f"{strength.get('requirement_id')}"
            ),
        )

    # -------------------------------------------------
    # Transferable strengths
    # -------------------------------------------------

    for item in strategy.get(
        "transferable_strengths",
        [],
    ):
        reason = item.get(
            "reason"
        )

        if not isinstance(
            reason,
            str,
        ):
            continue

        cited_evidence_text = " ".join(
            evidence_map[
                evidence_id
            ]
            for evidence_id in item.get(
                "evidence_ids",
                [],
            )
            if evidence_id in evidence_map
        )

        _validate_claim_text(
            text=reason,
            evidence_text=(
                cited_evidence_text
            ),
            context=(
                "transferable strength "
                f"{item.get('requirement_id')}"
            ),
        )

    # -------------------------------------------------
    # Priority sources
    # -------------------------------------------------

    for source in strategy.get(
        "priority_sources",
        [],
    ):
        reason = source.get(
            "reason"
        )

        if not isinstance(
            reason,
            str,
        ):
            continue

        source_id = source.get(
            "source_id"
        )

        source_evidence_text = " ".join(
            source_evidence_map.get(
                source_id,
                [],
            )
        )

        _validate_claim_text(
            text=reason,
            evidence_text=(
                source_evidence_text
            ),
            context=(
                "priority source "
                f"{source_id}"
            ),
        )


def _normalize_keyword(
    value: str,
) -> str:
    """
    Normalize keywords for deterministic
    policy comparison.
    """

    return (
        value.lower()
        .strip()
        .replace("-", " ")
        .replace("_", " ")
    )


def _keyword_conflicts_with_gap(
    keyword: str,
    blocked_requirements,
) -> bool:
    """
    Return True when a CV keyword directly overlaps
    with a GAP requirement.
    """

    keyword_normalized = (
        _normalize_keyword(
            keyword
        )
    )

    if not keyword_normalized:
        return False

    for requirement in (
        blocked_requirements
    ):
        blocked_normalized = (
            _normalize_keyword(
                requirement
            )
        )

        if not blocked_normalized:
            continue

        # Exact match.
        if (
            keyword_normalized
            == blocked_normalized
        ):
            return True

        # Keyword occurs inside a longer
        # blocked requirement.
        if (
            keyword_normalized
            in blocked_normalized
        ):
            return True

        # Blocked requirement occurs inside
        # a longer keyword.
        if (
            blocked_normalized
            in keyword_normalized
        ):
            return True

    return False


def add_keyword_policy(
    strategy: Dict[str, Any],
    match_gap_result: Dict[str, Any],
) -> None:
    """
    Build deterministic keyword policies from
    Match/Gap statuses.

    SUPPORTED:
        may be emphasized.

    PARTIAL:
        may be mentioned with careful framing.

    GAP:
        must not be claimed as existing experience.

    GAP requirements override any keyword
    recommendations produced by the LLM.
    """

    careful_keywords = []
    blocked_keywords = []

    for item in match_gap_result[
        "requirements"
    ]:
        requirement = item[
            "requirement"
        ]

        status = item[
            "status"
        ]

        if status == "PARTIAL":
            careful_keywords.append(
                requirement
            )

        elif status == "GAP":
            blocked_keywords.append(
                requirement
            )

    careful_keywords = list(
        dict.fromkeys(
            careful_keywords
        )
    )

    blocked_keywords = list(
        dict.fromkeys(
            blocked_keywords
        )
    )

    # -------------------------------------------------
    # Sanitize LLM-generated emphasize keywords.
    # -------------------------------------------------

    original_emphasize_keywords = (
        strategy.get(
            "cv_keywords_to_emphasize",
            [],
        )
    )

    cleaned_emphasize_keywords = []

    for keyword in (
        original_emphasize_keywords
    ):
        if not isinstance(
            keyword,
            str,
        ):
            continue

        keyword = keyword.strip()

        if not keyword:
            continue

        if (
            _keyword_conflicts_with_gap(
                keyword=keyword,
                blocked_requirements=(
                    blocked_keywords
                ),
            )
        ):
            continue

        cleaned_emphasize_keywords.append(
            keyword
        )

    strategy[
        "cv_keywords_to_emphasize"
    ] = list(
        dict.fromkeys(
            cleaned_emphasize_keywords
        )
    )

    strategy[
        "keywords_to_frame_carefully"
    ] = careful_keywords

    strategy[
        "keywords_not_to_claim"
    ] = blocked_keywords


def generate_strategy_with_retry(
    llm,
    payload: Dict[str, Any],
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    Call the strategy LLM with retries for
    empty or malformed JSON responses.

    Transport-level retries are handled by
    ChatOpenAI. This function handles successful
    API responses whose content is unusable.
    """

    last_error = None

    for _ in range(
        1,
        max_attempts + 1,
    ):

        response = llm.invoke([
            SystemMessage(
                content=SYSTEM_PROMPT
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

            print(
                "[STRATEGY] Empty response. "
                "Retrying..."
            )

            continue

        try:
            strategy = parse_json_response(
                content
            )

            print(
                "[STRATEGY] Valid JSON received."
            )

            return strategy

        except ValueError as error:
            last_error = error

            print(
                "[STRATEGY] Invalid JSON. "
                "Retrying..."
            )

    raise ValueError(
        "Application Strategy failed after "
        f"{max_attempts} LLM attempts. "
        f"Last error: {type(last_error).__name__}"
    )

def build_application_strategy(
    jd_analysis: Dict[str, Any],
    evidence_package: Dict[str, Any],
    match_gap_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate and validate an evidence-grounded
    application strategy.
    """

    payload = {
        "job_title": (
            jd_analysis.get(
                "job_title"
            )
        ),

        "jd_analysis": (
            jd_analysis
        ),

        "candidate_evidence": [
            {
                "evidence_id": item[
                    "evidence_id"
                ],

                # Compatibility field currently used
                # as the unique source identifier.
                "source_id": item[
                    "project_id"
                ],

                "source_title": item[
                    "project_title"
                ],

                "type": item[
                    "type"
                ],

                "entity_type": item.get(
                    "entity_type",
                    "project",
                ),

                "evidence_status": (
                    item.get(
                        "evidence_status",
                        "verified",
                    )
                ),

                "text": item[
                    "text"
                ],
            }
            for item in evidence_package[
                "evidence"
            ]
        ],

        "match_gap_analysis": (
            match_gap_result
        ),
    }

    llm = get_llm(
        temperature=0.0
    )

    strategy = generate_strategy_with_retry(
    llm=llm,
    payload=payload,
    max_attempts=3,
)
    # -------------------------------------------------
    # Step 1:
    # Filter citations not validated by Match/Gap.
    # -------------------------------------------------

    sanitize_strategy_evidence(
        strategy=strategy,
        match_gap_result=(
            match_gap_result
        ),
    )

    # -------------------------------------------------
    # Step 2:
    # Strict structural / evidence / source validation.
    # -------------------------------------------------

    validate_strategy(
        strategy=strategy,
        match_gap_result=(
            match_gap_result
        ),
        evidence_package=(
            evidence_package
        ),
    )

    # -------------------------------------------------
    # Step 3:
    # Reject unsupported leadership / ownership claims.
    # -------------------------------------------------

    validate_high_risk_claims(
        strategy=strategy,
        evidence_package=(
            evidence_package
        ),
    )

    # -------------------------------------------------
    # Step 4:
    # Build PARTIAL / GAP keyword policy
    # deterministically.
    # -------------------------------------------------

    add_keyword_policy(
        strategy=strategy,
        match_gap_result=(
            match_gap_result
        ),
    )

    strategy["job_title"] = (
        jd_analysis.get(
            "job_title"
        )
    )

    return strategy