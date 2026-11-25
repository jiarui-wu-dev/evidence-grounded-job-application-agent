import json
from typing import Any, Dict, List

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from src.llm.client import get_llm


VALID_STATUSES = {
    "SUPPORTED",
    "OVERSTATED",
    "UNSUPPORTED",
}


SYSTEM_PROMPT = """
You are a strict evidence-grounding validator for CV claims.

Your task is to determine whether each generated CV claim is
fully supported by the exact candidate evidence cited for it.

Use ONLY the supplied evidence.

Do not use:
- general technical knowledge,
- assumptions about what a technology usually implies,
- information from other candidate sources,
- information not explicitly contained in the cited evidence.


CLASSIFICATION

SUPPORTED

Use SUPPORTED only when every material factual assertion in
the CV claim is directly supported by the cited evidence or is
a conservative paraphrase of that evidence.

OVERSTATED

Use OVERSTATED when the core claim is supported, but the CV
wording adds one or more unsupported implications, stronger
claims, causal relationships, outcomes, scope, expertise, or
business impact.

Typical OVERSTATED patterns include:

- evidence says an activity occurred, but the CV claims it
  caused an improvement;
- evidence says Docker was used, but the CV claims this
  ensured consistency or automated provisioning;
- evidence says a workflow was built, but the CV claims
  production deployment or business impact;
- evidence demonstrates hands-on experience, but the CV says
  "expertise" or "expert";
- evidence lists several activities separately, but the CV
  asserts a causal relationship between them;
- evidence says an LLM-assisted workflow was developed, but
  the CV says the candidate developed or trained an LLM.

UNSUPPORTED

Use UNSUPPORTED when the central factual claim is not
supported by the cited evidence, or when the claim relies on
evidence that does not demonstrate the stated capability.


STRICT CAUSALITY RULE

Do not infer causality.

If evidence separately says:

- error analysis was performed;
- fine-tuning was performed;
- a metric improved;

this does NOT prove:

"the metric improved through error analysis and fine-tuning"

unless that causal relationship is explicitly supported.


IMPACT RULE

Do not infer:

- improved reliability,
- improved robustness,
- improved coverage,
- ensured consistency,
- enabled auditability,
- improved efficiency,
- business value,
- production readiness,

unless the cited evidence explicitly supports that impact.


EVIDENCE STATUS

Evidence marked "verified" may be used normally.

Evidence marked "provisional" must be treated conservatively.
A definitive claim must not rely only on provisional evidence.


OUTPUT

Return valid JSON only.

For every input claim, return exactly one result.

Required format:

{
  "results": [
    {
      "claim_id": "claim_1",
      "status": "SUPPORTED",
      "reason": "Short explanation based only on cited evidence.",
      "unsupported_phrases": []
    }
  ]
}

For OVERSTATED or UNSUPPORTED claims, unsupported_phrases
should contain the exact or near-exact phrases that are not
defended by the cited evidence.

Do not rewrite the CV claim yet.
Do not suggest improved wording yet.
"""


def parse_json_response(
    content: str,
) -> Dict[str, Any]:
    """
    Parse validator JSON with lightweight recovery for
    common LLM formatting problems.
    """

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
                    "Validator response must "
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
                    "Validator response must "
                    "be a JSON object."
                )

            return result

        except (
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error

    raise ValueError(
        "Failed to parse Evidence Validator "
        f"response as JSON: {type(last_error).__name__}"
    )


def build_evidence_map(
    expanded_sources: List[
        Dict[str, Any]
    ],
) -> Dict[str, Dict[str, Any]]:
    """
    Build evidence lookup from frozen expanded sources.
    """

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


def build_claim_payload(
    claim_id: str,
    text: str,
    evidence_ids: List[str],
    evidence_map: Dict[
        str,
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    """
    Build one claim with only its cited evidence.
    """

    evidence = []

    for evidence_id in evidence_ids:

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

    return {
        "claim_id": claim_id,
        "text": text,
        "evidence": evidence,
    }


def invoke_validation_batch(
    llm,
    claims: List[Dict[str, Any]],
    label: str,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    """
    Validate one small batch of related claims.
    """

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

            validate_batch_result(
                result=result,
                claims=claims,
            )

            return result

        except ValueError as error:

            last_error = error

    raise ValueError(
        f"Evidence validation batch "
        f"{label} failed after "
        f"{max_attempts} attempts. "
        f"Last error: {type(last_error).__name__}"
    )


def validate_batch_result(
    result: Dict[str, Any],
    claims: List[Dict[str, Any]],
) -> None:
    """
    Deterministically validate validator output.
    """

    results = result.get(
        "results"
    )

    if not isinstance(
        results,
        list,
    ):
        raise ValueError(
            "Validator output must contain "
            "a results list."
        )

    expected_ids = {
        claim[
            "claim_id"
        ]
        for claim in claims
    }

    returned_ids = set()

    for item in results:

        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Each validator result "
                "must be an object."
            )

        claim_id = item.get(
            "claim_id"
        )

        status = item.get(
            "status"
        )

        reason = item.get(
            "reason"
        )

        unsupported_phrases = item.get(
            "unsupported_phrases"
        )

        if not isinstance(claim_id, str) or claim_id not in expected_ids:
            raise ValueError(
                "Unknown claim ID returned: "
                f"{claim_id}"
            )

        if claim_id in returned_ids:
            raise ValueError(
                "Duplicate claim ID returned: "
                f"{claim_id}"
            )

        returned_ids.add(
            claim_id
        )

        if not isinstance(status, str) or status not in VALID_STATUSES:
            raise ValueError(
                "Invalid validation status "
                f"for {claim_id}: {status}"
            )

        if not isinstance(
            reason,
            str,
        ):
            raise ValueError(
                f"reason must be a string "
                f"for {claim_id}."
            )

        if not isinstance(
            unsupported_phrases,
            list,
        ):
            raise ValueError(
                "unsupported_phrases must "
                f"be a list for {claim_id}."
            )

        for phrase in unsupported_phrases:

            if not isinstance(
                phrase,
                str,
            ):
                raise ValueError(
                    "unsupported_phrases must "
                    "contain strings only."
                )

        if (
            status == "SUPPORTED"
            and unsupported_phrases
        ):
            raise ValueError(
                "SUPPORTED claim must not "
                "contain unsupported phrases: "
                f"{claim_id}"
            )

    missing_ids = (
        expected_ids
        - returned_ids
    )

    if missing_ids:
        raise ValueError(
            "Validator did not return "
            "all claims: "
            f"{sorted(missing_ids)}"
        )


def build_validation_batches(
    cv_draft: Dict[str, Any],
    expanded_sources: List[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    """
    Create small validation batches:

    - profile;
    - one batch per Research/Experience/Project source.

    Skills are not LLM-validated here because the current
    CV Tailor builds Skills deterministically from explicit
    Candidate KB skill evidence.
    """

    evidence_map = build_evidence_map(
        expanded_sources
    )

    batches = []

    # ========================================================
    # Profile
    # ========================================================

    profile = cv_draft[
        "profile"
    ]

    profile_claim = build_claim_payload(
        claim_id="profile",
        text=profile[
            "text"
        ],
        evidence_ids=profile[
            "evidence_ids"
        ],
        evidence_map=evidence_map,
    )

    batches.append({
        "label": "profile",
        "claims": [
            profile_claim
        ],
    })

    # ========================================================
    # Source-backed sections
    # ========================================================

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

            claims = []

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

                claims.append(
                    build_claim_payload(
                        claim_id=claim_id,
                        text=bullet[
                            "text"
                        ],
                        evidence_ids=bullet[
                            "evidence_ids"
                        ],
                        evidence_map=(
                            evidence_map
                        ),
                    )
                )

            if claims:

                batches.append({
                    "label": (
                        f"{section_name}:"
                        f"{source_id}"
                    ),

                    "claims": claims,
                })

    return batches


def find_changed_claim_ids(
    previous_draft: Dict[str, Any],
    revised_draft: Dict[str, Any],
    expanded_sources: List[Dict[str, Any]],
) -> List[str]:
    """Compare the exact validator inputs before and after a text-only revision."""
    previous_claims, revised_claims = (
        {
            claim["claim_id"]: claim
            for batch in build_validation_batches(draft, expanded_sources)
            for claim in batch["claims"]
        }
        for draft in (previous_draft, revised_draft)
    )
    if previous_claims.keys() != revised_claims.keys():
        raise ValueError("Revision changed the set of CV claim IDs.")

    return [
        claim_id
        for claim_id, claim in revised_claims.items()
        if claim != previous_claims[claim_id]
    ]


def add_summary(
    validation_result: Dict[str, Any],
) -> None:
    """
    Add deterministic aggregate counts.
    """

    summary = {
        "SUPPORTED": 0,
        "OVERSTATED": 0,
        "UNSUPPORTED": 0,
        "total": 0,
    }

    for item in validation_result[
        "results"
    ]:

        summary[
            item["status"]
        ] += 1

        summary[
            "total"
        ] += 1

    if summary["total"]:

        summary[
            "grounded_rate"
        ] = (
            summary["SUPPORTED"]
            / summary["total"]
        )

        summary[
            "problem_rate"
        ] = (
            (
                summary["OVERSTATED"]
                + summary["UNSUPPORTED"]
            )
            / summary["total"]
        )

    else:

        summary[
            "grounded_rate"
        ] = 0.0

        summary[
            "problem_rate"
        ] = 0.0

    validation_result[
        "summary"
    ] = summary


def validate_cv_evidence(
    cv_draft: Dict[str, Any],
    expanded_sources: List[
        Dict[str, Any]
    ],
    *,
    changed_claim_ids: List[str] | None = None,
    previous_validation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Validate Profile and CV bullets against their cited evidence.

    With changed claim IDs, replace only those verdicts in the previous result.
    An empty list preserves every verdict without invoking the model.
    """

    batches = build_validation_batches(
        cv_draft=cv_draft,
        expanded_sources=(
            expanded_sources
        ),
    )

    previous_rows = None
    if changed_claim_ids is not None:
        if previous_validation is None:
            raise ValueError("Targeted validation requires previous validation results.")

        claims = [claim for batch in batches for claim in batch["claims"]]
        validate_batch_result(previous_validation, claims)
        previous_rows = previous_validation["results"]

        requested_ids = set(changed_claim_ids)
        known_ids = {claim["claim_id"] for claim in claims}
        if len(known_ids) != len(claims):
            raise ValueError("Duplicate CV claim IDs.")
        if len(requested_ids) != len(changed_claim_ids):
            raise ValueError("Duplicate changed claim IDs.")
        if requested_ids - known_ids:
            raise ValueError("Unknown changed claim IDs.")

        selected_batches = []
        for batch in batches:
            selected_claims = [
                claim for claim in batch["claims"] if claim["claim_id"] in requested_ids
            ]
            if selected_claims:
                selected_batches.append({"label": batch["label"], "claims": selected_claims})
        batches = selected_batches

    all_results = []
    if batches:
        llm = get_llm(temperature=0.0)

    for batch in batches:

        batch_result = (
            invoke_validation_batch(
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
        )

        all_results.extend(
            batch_result[
                "results"
            ]
        )

    if previous_rows is not None:
        selected_claims = [claim for batch in batches for claim in batch["claims"]]
        validate_batch_result({"results": all_results}, selected_claims)
        replacements = {row["claim_id"]: row for row in all_results}
        all_results = [
            replacements.get(row["claim_id"], row) for row in previous_rows
        ]

    validation_result = {
        "results": (
            all_results
        )
    }

    add_summary(
        validation_result
    )

    return validation_result
