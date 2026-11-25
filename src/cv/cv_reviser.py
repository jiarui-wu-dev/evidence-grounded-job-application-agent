import copy
import json
from typing import Any, Dict, List

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from src.llm.client import get_llm


PROBLEM_STATUSES = {
    "OVERSTATED",
    "UNSUPPORTED",
}


SYSTEM_PROMPT = """
You are an evidence-grounded CV claim reviser.

Your task is to revise CV claims that were flagged by an
Evidence Validator.

You will receive:

1. the original CV claim;
2. validator feedback;
3. unsupported phrases identified by the validator;
4. the exact candidate evidence cited by the claim.

IMPORTANT:

The validator feedback is useful but is NOT automatically
correct.

You must independently inspect ALL cited evidence before
revising the claim.

If the validator flags a phrase that is actually directly
supported by any cited evidence, preserve that phrase.

Your goal is:

- preserve as much useful supported information as possible;
- remove or soften only unsupported implications;
- keep quantitative results when directly supported;
- preserve important technical keywords when supported;
- avoid adding new factual claims.


GROUNDING RULES

Do not infer:

- causality;
- business impact;
- production deployment;
- customer usage;
- improved reliability;
- improved robustness;
- improved coverage;
- ensured consistency;
- auditability;
- efficiency improvement;

unless explicitly supported by the cited evidence.

Do not convert:

- workflow development
  into model development;

- model evaluation
  into model training;

- use of a technology
  into an inferred impact of that technology.

Do not use stronger capability language such as:

- expert;
- expertise;
- extensive experience;

unless explicitly supported.

CAUSALITY AND RELATIONSHIP REVISION RULES

When revising a flagged claim, do not replace one unsupported
relationship with another unsupported relationship.

If cited evidence separately supports fact A and fact B, but
does not explicitly state that A caused B, describe them as
independent facts.

Prefer neutral coordination such as:

- "A and B"
- "A; additionally, B"
- separate clauses or separate sentences

Do NOT use causal or result wording such as:

- ->
- resulting in
- leading to
- enabling
- thereby
- therefore
- through
- by
- reducing
- improving
- ensuring
- allowing
- achieving

unless the cited evidence explicitly establishes that causal
relationship.

In particular, avoid participial constructions such as:

- ", reducing ..."
- ", improving ..."
- ", enabling ..."
- ", ensuring ..."

when they would imply that the preceding activity caused the
following result.

If the evidence supports an activity and a measured result
separately, preserve both as neutral independent facts rather
than inferring causality.


PURPOSE RULE

Do not infer the purpose of an activity from:

- the project title;
- the source title;
- a skill list;
- the general project domain.

If evidence states only that an activity was performed, state
the activity directly.

Do not add purpose phrases such as:

- "to adapt..."
- "to improve..."
- "to enable..."
- "in order to..."
- "for the purpose of..."

unless that purpose is explicitly supported by cited evidence.

ORIGINAL CLAIM SAFETY RULE

The original CV claim is not evidence.

Do not preserve or rephrase a factual statement merely because
it appeared in the original claim or because the validator did
not explicitly list it in unsupported_phrases.

Every factual part of the revised claim must be independently
supported by the cited evidence.

Validator feedback may identify only some problematic phrases.
You are responsible for checking the entire original claim
against the evidence before preserving any wording.

When an unsupported Result / Impact or Problem / Goal is
removed, do not convert that unsupported content into a new
activity, purpose, or method statement.

For example:

Original:
"Evaluate and adapt ASR -> built an evaluation workflow ->
enabled systematic assessment."

Evidence:
"Built an end-to-end evaluation and domain-adaptation
workflow."

Safe revision:
"Built an end-to-end evaluation and domain-adaptation
workflow."

Unsafe revision:
"Evaluated and adapted ASR by building an end-to-end
workflow."

The original claim must never be treated as a source of truth.


FORMAT RULE

Never output:

- "Problem/Goal:"
- "Action/Method:"
- "Result/Impact:"
- "->"

Write natural professional CV sentences only.


REVISION STYLE

Keep CV language concise and recruiter-readable.

Prefer:

Problem / Goal
-> Action / Method
-> directly supported Result / Impact

If no supported impact exists, stop after the Action / Method.

Do not invent a result just to make the bullet sound stronger.

For profile claims, prefer language such as:

- hands-on experience;
- experience with;
- worked on;
- conducted;
- developed workflows for;
- evaluated;

when these accurately reflect the evidence.

ORIGINAL CLAIM SAFETY RULE

The original CV claim is not evidence.

Do not preserve or rephrase a factual statement merely because
it appeared in the original claim or because the validator did
not explicitly list it in unsupported_phrases.

Every factual part of the revised claim must be independently
supported by the cited evidence.

Validator feedback may identify only some problematic phrases.
You are responsible for checking the entire original claim
against the evidence before preserving any wording.

When an unsupported Result / Impact or Problem / Goal is
removed, do not convert that unsupported content into a new
activity, purpose, or method statement.

For example:

Original:
"Evaluate and adapt ASR -> built an evaluation workflow ->
enabled systematic assessment."

Evidence:
"Built an end-to-end evaluation and domain-adaptation
workflow."

Safe revision:
"Built an end-to-end evaluation and domain-adaptation
workflow."

Unsafe revision:
"Evaluated and adapted ASR by building an end-to-end
workflow."

The original claim must never be treated as a source of truth.


EVIDENCE IDS

Do NOT change evidence IDs.

You are revising wording only.

Do not request additional evidence.

Do not introduce facts that require evidence not already
cited by the claim.


OUTPUT

Return valid JSON only.

Return exactly one result for every input claim.

Required format:

{
  "results": [
    {
      "claim_id": "claim_id",
      "revised_text": "Revised evidence-grounded wording."
    }
  ]
}

Do not include explanations.
Do not include evidence_ids.
Do not use Markdown.
"""


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
                    "Revision response must "
                    "be a JSON object."
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
                    "Revision response must "
                    "be a JSON object."
                )

            return result

        except (
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error

    raise ValueError(
        "Failed to parse revision response "
        f"as JSON: {type(last_error).__name__}"
    )


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


def build_claim_map(
    cv_draft: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Build deterministic claim locations inside the CV.

    This lets us replace text without changing evidence IDs
    or regenerating the complete CV.
    """

    claim_map = {
        "profile": {
            "text": (
                cv_draft[
                    "profile"
                ][
                    "text"
                ]
            ),

            "evidence_ids": (
                cv_draft[
                    "profile"
                ][
                    "evidence_ids"
                ]
            ),

            "section": "profile",

            "source_id": None,

            "bullet_index": None,
        }
    }

    section_names = [
        "research_experience",
        "experience",
        "selected_projects",
    ]

    for section_name in section_names:

        for entry in cv_draft.get(
            section_name,
            [],
        ):

            source_id = entry[
                "source_id"
            ]

            for index, bullet in enumerate(
                entry.get(
                    "bullets",
                    [],
                ),
                start=1,
            ):

                claim_id = (
                    f"{section_name}:"
                    f"{source_id}:"
                    f"bullet_{index}"
                )

                claim_map[
                    claim_id
                ] = {
                    "text": (
                        bullet[
                            "text"
                        ]
                    ),

                    "evidence_ids": (
                        bullet[
                            "evidence_ids"
                        ]
                    ),

                    "section": (
                        section_name
                    ),

                    "source_id": (
                        source_id
                    ),

                    "bullet_index": (
                        index - 1
                    ),
                }

    return claim_map


def build_revision_batches(
    cv_draft: Dict[str, Any],
    validation_result: Dict[str, Any],
    expanded_sources: List[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    """
    Group problematic claims by CV source.

    The profile forms its own batch.

    Each Research / Experience / Project source forms
    another batch.
    """

    claim_map = build_claim_map(
        cv_draft
    )

    evidence_map = build_evidence_map(
        expanded_sources
    )

    grouped = {}

    for validation in validation_result.get(
        "results",
        [],
    ):

        status = validation.get(
            "status"
        )

        if status not in PROBLEM_STATUSES:
            continue

        claim_id = validation[
            "claim_id"
        ]

        if claim_id not in claim_map:
            raise ValueError(
                "Validation references "
                "unknown CV claim: "
                f"{claim_id}"
            )

        claim = claim_map[
            claim_id
        ]

        if claim_id == "profile":
            batch_key = "profile"

        else:
            batch_key = (
                f"{claim['section']}:"
                f"{claim['source_id']}"
            )

        evidence = []

        for evidence_id in claim[
            "evidence_ids"
        ]:

            if evidence_id not in evidence_map:
                raise ValueError(
                    f"Unknown evidence ID "
                    f"for {claim_id}: "
                    f"{evidence_id}"
                )

            item = evidence_map[
                evidence_id
            ]

            evidence.append({
                "evidence_id": (
                    evidence_id
                ),

                "evidence_status": (
                    item.get(
                        "evidence_status",
                        "verified",
                    )
                ),

                "text": (
                    item[
                        "text"
                    ]
                ),
            })

        revision_item = {
            "claim_id": (
                claim_id
            ),

            "original_text": (
                claim[
                    "text"
                ]
            ),

            "validator_status": (
                status
            ),

            "validator_reason": (
                validation.get(
                    "reason",
                    "",
                )
            ),

            "unsupported_phrases": (
                validation.get(
                    "unsupported_phrases",
                    [],
                )
            ),

            "cited_evidence": (
                evidence
            ),
        }

        grouped.setdefault(
            batch_key,
            [],
        )

        grouped[
            batch_key
        ].append(
            revision_item
        )

    return [
        {
            "label": label,
            "claims": claims,
        }
        for (
            label,
            claims
        ) in grouped.items()
    ]


def validate_revision_result(
    result: Dict[str, Any],
    claims: List[Dict[str, Any]],
) -> None:

    results = result.get(
        "results"
    )

    if not isinstance(
        results,
        list,
    ):
        raise ValueError(
            "Revision output must contain "
            "a results list."
        )

    expected_ids = {
        item[
            "claim_id"
        ]
        for item in claims
    }

    returned_ids = set()

    for item in results:

        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Each revision result "
                "must be an object."
            )

        claim_id = item.get(
            "claim_id"
        )

        revised_text = item.get(
            "revised_text"
        )

        if claim_id not in expected_ids:
            raise ValueError(
                "Unknown revision claim ID: "
                f"{claim_id}"
            )

        if claim_id in returned_ids:
            raise ValueError(
                "Duplicate revision claim ID: "
                f"{claim_id}"
            )

        returned_ids.add(
            claim_id
        )

        if (
            not isinstance(
                revised_text,
                str,
            )
            or not revised_text.strip()
        ):
            raise ValueError(
                "revised_text must be "
                "a non-empty string for "
                f"{claim_id}."
            )

    missing_ids = (
        expected_ids
        - returned_ids
    )

    if missing_ids:
        raise ValueError(
            "Reviser did not return "
            "all claims: "
            f"{sorted(missing_ids)}"
        )


def invoke_revision_batch(
    llm,
    claims: List[Dict[str, Any]],
    label: str,
    max_attempts: int = 2,
) -> Dict[str, Any]:

    payload = {
        "claims": claims,
    }

    last_error = None

    for attempt in range(
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

            continue

        try:

            result = parse_json_response(
                content
            )

            validate_revision_result(
                result=result,
                claims=claims,
            )

            return result

        except ValueError as error:

            last_error = error

    raise ValueError(
        f"Revision batch {label} "
        f"failed after {max_attempts} attempts. "
        f"Last error: {type(last_error).__name__}"
    )


def apply_revision(
    revised_draft: Dict[str, Any],
    claim_id: str,
    revised_text: str,
) -> None:
    """
    Replace claim text only.

    Evidence IDs remain unchanged.
    """

    if claim_id == "profile":

        revised_draft[
            "profile"
        ][
            "text"
        ] = revised_text

        return

    parts = claim_id.split(
        ":"
    )

    if len(parts) != 3:
        raise ValueError(
            "Invalid claim ID format: "
            f"{claim_id}"
        )

    section_name = parts[
        0
    ]

    source_id = parts[
        1
    ]

    bullet_token = parts[
        2
    ]

    if not bullet_token.startswith(
        "bullet_"
    ):
        raise ValueError(
            "Invalid bullet claim ID: "
            f"{claim_id}"
        )

    bullet_number = int(
        bullet_token.split(
            "_",
            1,
        )[1]
    )

    bullet_index = (
        bullet_number - 1
    )

    for entry in revised_draft.get(
        section_name,
        [],
    ):

        if (
            entry.get(
                "source_id"
            )
            == source_id
        ):

            bullets = entry[
                "bullets"
            ]

            if (
                bullet_index < 0
                or bullet_index >= len(
                    bullets
                )
            ):
                raise ValueError(
                    "Bullet index out of "
                    f"range for {claim_id}."
                )

            bullets[
                bullet_index
            ][
                "text"
            ] = revised_text

            return

    raise ValueError(
        "Could not locate claim "
        f"in CV draft: {claim_id}"
    )


def revise_cv_draft(
    cv_draft: Dict[str, Any],
    validation_result: Dict[str, Any],
    expanded_sources: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    """
    Revise only claims classified as OVERSTATED or
    UNSUPPORTED.

    SUPPORTED claims remain byte-for-byte unchanged.
    Evidence IDs remain unchanged.
    """

    revised_draft = copy.deepcopy(
        cv_draft
    )

    batches = build_revision_batches(
        cv_draft=cv_draft,
        validation_result=(
            validation_result
        ),
        expanded_sources=(
            expanded_sources
        ),
    )

    if not batches:

        print(
            "[CV REVISER] "
            "No problematic claims found."
        )

        return revised_draft

    llm = get_llm(
        temperature=0.0
    )

    for batch in batches:

        result = invoke_revision_batch(
            llm=llm,
            claims=(
                batch[
                    "claims"
                ]
            ),
            label=(
                batch[
                    "label"
                ]
            ),
            max_attempts=2,
        )

        for item in result[
            "results"
        ]:

            apply_revision(
                revised_draft=(
                    revised_draft
                ),
                claim_id=(
                    item[
                        "claim_id"
                    ]
                ),
                revised_text=(
                    item[
                        "revised_text"
                    ]
                ),
            )

    return revised_draft