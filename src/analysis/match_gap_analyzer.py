import json
from typing import Any, Dict, List

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from src.analysis.evidence_retriever import (
    EvidenceRetriever,
)
from src.llm.client import get_llm


SYSTEM_PROMPT = """
You are an evidence-grounded job requirement matcher.

Your task is to compare job requirements against
candidate evidence.

You MUST use only the candidate evidence provided
in the input.

Do not infer skills, technologies, experience, or
achievements that are not directly supported by
the evidence.

A retrieved evidence item is NOT automatically proof
that the requirement is satisfied.

Classify every requirement as exactly one of:

SUPPORTED:
There is direct, defensible evidence that the candidate
has demonstrated the requirement.

PARTIAL:
The candidate has related or transferable experience,
but the exact requirement is not directly demonstrated.

GAP:
There is no defensible evidence supporting the requirement.

Important rules:

1. Do not treat semantic similarity alone as evidence.

2. If a specific technology is required, related
technologies are not sufficient for SUPPORTED.

Example:
React experience does not prove NLP experience.
Docker experience does not prove RAG experience.

3. If the evidence only demonstrates adjacent or
transferable experience, use PARTIAL.

4. GAP requirements must use an empty evidence_ids list.

5. evidence_ids may contain only IDs from the evidence
provided in the input.

6. Never invent evidence IDs.

7. Be conservative. The result should be defendable
in a real technical interview.

Evidence confidence rules:

8. evidence_status="verified" represents candidate
   information that may be used normally when deciding
   SUPPORTED or PARTIAL.

9. evidence_status="provisional" represents candidate
   information that has not yet been fully verified.

10. Provisional evidence may support a PARTIAL
    classification, but must never be the sole basis
    for SUPPORTED.

11. A SUPPORTED classification must cite at least one
    directly relevant evidence item with
    evidence_status="verified".

12. If a requirement is supported only by provisional
    evidence, classify it as PARTIAL rather than
    SUPPORTED.


Return JSON only.

Required format:

{
  "requirements": [
    {
      "requirement_id": "must_1",
      "status": "SUPPORTED",
      "evidence_ids": [
        "example_evidence_id"
      ],
      "explanation": "Short evidence-grounded explanation."
    }
  ]
}
"""


def build_requirements(
    jd_analysis: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Convert structured JD analysis into a stable
    requirement list with deterministic IDs.
    """

    requirements = []

    for index, item in enumerate(
        jd_analysis.get(
            "must_have_skills",
            [],
        ),
        start=1,
    ):
        requirements.append({
            "requirement_id": (
                f"must_{index}"
            ),
            "category": (
                "must_have_skill"
            ),
            "requirement": item,
        })

    for index, item in enumerate(
        jd_analysis.get(
            "responsibilities",
            [],
        ),
        start=1,
    ):
        requirements.append({
            "requirement_id": (
                f"resp_{index}"
            ),
            "category": (
                "responsibility"
            ),
            "requirement": item,
        })

    for index, item in enumerate(
        jd_analysis.get(
            "experience_requirements",
            [],
        ),
        start=1,
    ):
        requirements.append({
            "requirement_id": (
                f"exp_{index}"
            ),
            "category": (
                "experience"
            ),
            "requirement": item,
        })

    for index, item in enumerate(
        jd_analysis.get(
            "nice_to_have_skills",
            [],
        ),
        start=1,
    ):
        requirements.append({
            "requirement_id": (
                f"nice_{index}"
            ),
            "category": (
                "nice_to_have_skill"
            ),
            "requirement": item,
        })

    return requirements


def build_evidence_context(
    evidence_package: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Reduce the retrieval package to the fields
    the LLM is allowed to reason over.

    Evidence confidence metadata is preserved so
    the model and deterministic validation layer
    can distinguish verified from provisional
    candidate information.
    """

    return [
        {
            "evidence_id": item[
                "evidence_id"
            ],

            "project_title": item[
                "project_title"
            ],

            "type": item[
                "type"
            ],

            "entity_type": item.get(
                "entity_type",
                "project",
            ),

            "evidence_status": item.get(
                "evidence_status",
                "verified",
            ),

            "text": item[
                "text"
            ],
        }
        for item
        in evidence_package["evidence"]
    ]

def parse_json_response(
    content: str,
) -> Dict[str, Any]:
    """
    Parse JSON while tolerating accidental
    Markdown code fences.
    """

    content = content.strip()

    if content.startswith(
        "```json"
    ):
        content = content[
            len("```json"):
        ]

    elif content.startswith(
        "```"
    ):
        content = content[
            len("```"):
        ]

    if content.endswith(
        "```"
    ):
        content = content[:-3]

    return json.loads(
        content.strip()
    )


def enforce_evidence_confidence(
    result: Dict[str, Any],
    evidence: List[Dict[str, Any]],
) -> None:
    """
    Enforce candidate evidence confidence
    deterministically.

    A SUPPORTED classification requires at least
    one cited VERIFIED evidence item.

    If all cited evidence is provisional, downgrade
    SUPPORTED -> PARTIAL instead of crashing the
    entire pipeline.
    """

    evidence_map = {
        item["evidence_id"]: item
        for item in evidence
    }

    for requirement in result.get(
        "requirements",
        [],
    ):
        if (
            requirement.get("status")
            != "SUPPORTED"
        ):
            continue

        evidence_ids = requirement.get(
            "evidence_ids",
            [],
        )

        cited_evidence = [
            evidence_map[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_map
        ]

        has_verified_evidence = any(
            item.get(
                "evidence_status",
                "verified",
            )
            == "verified"
            for item in cited_evidence
        )

        if not has_verified_evidence:
            requirement[
                "status"
            ] = "PARTIAL"

            requirement[
                "confidence_adjustment"
            ] = (
                "Downgraded from SUPPORTED "
                "to PARTIAL because the cited "
                "candidate evidence is only "
                "provisional."
            )


def enforce_status_evidence_consistency(
    result: Dict[str, Any],
) -> None:
    """
    Enforce the invariant that SUPPORTED and PARTIAL
    requirements must cite candidate evidence.

    If the LLM returns SUPPORTED or PARTIAL without
    evidence IDs, downgrade the requirement to GAP
    instead of crashing the pipeline.

    This is intentionally conservative:
    no evidence means no claimable support.
    """

    for requirement in result.get(
        "requirements",
        [],
    ):
        status = requirement.get(
            "status"
        )

        evidence_ids = requirement.get(
            "evidence_ids",
            [],
        )

        if (
            status
            in {
                "SUPPORTED",
                "PARTIAL",
            }
            and not evidence_ids
        ):
            original_status = status

            requirement[
                "status"
            ] = "GAP"

            requirement[
                "evidence_ids"
            ] = []

            requirement[
                "consistency_adjustment"
            ] = (
                f"Downgraded from "
                f"{original_status} to GAP "
                "because the model did not "
                "cite any supporting candidate "
                "evidence."
            )


def validate_result(
    result: Dict[str, Any],
    requirements: List[
        Dict[str, str]
    ],
    evidence: List[
        Dict[str, str]
    ],
) -> None:
    """
    Validate that the LLM output only references
    known requirements and evidence IDs.
    """

    allowed_requirement_ids = {
        item["requirement_id"]
        for item in requirements
    }

    allowed_evidence_ids = {
        item["evidence_id"]
        for item in evidence
    }

    allowed_statuses = {
        "SUPPORTED",
        "PARTIAL",
        "GAP",
    }

    returned_requirements = result.get(
        "requirements"
    )

    if not isinstance(
        returned_requirements,
        list,
    ):
        raise ValueError(
            "LLM response must contain "
            "a 'requirements' list."
        )

    returned_ids = set()

    for item in returned_requirements:
        requirement_id = item.get(
            "requirement_id"
        )

        status = item.get(
            "status"
        )

        evidence_ids = item.get(
            "evidence_ids"
        )

        if requirement_id not in (
            allowed_requirement_ids
        ):
            raise ValueError(
                "Unknown requirement ID "
                f"returned by LLM: "
                f"{requirement_id}"
            )

        if requirement_id in returned_ids:
            raise ValueError(
                "Duplicate requirement ID "
                f"returned by LLM: "
                f"{requirement_id}"
            )

        returned_ids.add(
            requirement_id
        )

        if status not in allowed_statuses:
            raise ValueError(
                f"Invalid status: {status}"
            )

        if not isinstance(
            evidence_ids,
            list,
        ):
            raise ValueError(
                "evidence_ids must be "
                "a list."
            )

        for evidence_id in evidence_ids:
            if evidence_id not in (
                allowed_evidence_ids
            ):
                raise ValueError(
                    "Unknown evidence ID "
                    f"returned by LLM: "
                    f"{evidence_id}"
                )

        if (
            status == "GAP"
            and evidence_ids
        ):
            raise ValueError(
                "GAP requirements must "
                "have no evidence IDs."
            )

        if (
        status in {
            "SUPPORTED",
            "PARTIAL",
        }
        and not evidence_ids
    ):
            raise ValueError(
                f"{status} requirements must "
                "reference at least one "
                "evidence ID."
        )


    missing_ids = (
        allowed_requirement_ids
        - returned_ids
    )

    if missing_ids:
        raise ValueError(
            "LLM did not classify all "
            "requirements: "
            f"{sorted(missing_ids)}"
        )


def add_requirement_metadata(
    result: Dict[str, Any],
    requirements: List[
        Dict[str, str]
    ],
) -> None:
    """
    Add deterministic requirement text and
    category back into the LLM result.
    """

    requirement_map = {
        item["requirement_id"]: item
        for item in requirements
    }

    for item in result[
        "requirements"
    ]:
        metadata = requirement_map[
            item["requirement_id"]
        ]

        item["category"] = metadata[
            "category"
        ]

        item["requirement"] = metadata[
            "requirement"
        ]


def add_summary(
    result: Dict[str, Any],
) -> None:
    """
    Calculate summary counts deterministically
    instead of asking the LLM to count them.
    """

    summary = {
        "SUPPORTED": 0,
        "PARTIAL": 0,
        "GAP": 0,
    }

    for item in result[
        "requirements"
    ]:
        summary[
            item["status"]
        ] += 1

    total = len(
        result["requirements"]
    )

    summary["total"] = total

    if total:
        summary[
            "support_rate"
        ] = (
            summary["SUPPORTED"]
            / total
        )

    else:
        summary[
            "support_rate"
        ] = 0.0

    result["summary"] = summary


def analyze_match_gap(
    jd_analysis: Dict[str, Any],
    evidence_package: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform evidence-grounded job requirement
    classification.
    """

    requirements = build_requirements(
        jd_analysis
    )

    evidence = build_evidence_context(
        evidence_package
    )

    payload = {
        "job_title": jd_analysis.get(
            "job_title"
        ),
        "requirements": requirements,
        "candidate_evidence": evidence,
    }

    llm = get_llm(
        temperature=0.0
    )

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

    result = parse_json_response(
    response.content
)

    # -------------------------------------------------
    # Step 1:
    # Enforce candidate evidence confidence.
    #
    # SUPPORTED based only on provisional evidence
    # becomes PARTIAL.
    # -------------------------------------------------

    enforce_evidence_confidence(
        result=result,
        evidence=evidence,
    )

    # -------------------------------------------------
    # Step 2:
    # Enforce status / evidence consistency.
    #
    # SUPPORTED or PARTIAL without any cited evidence
    # becomes GAP instead of crashing the pipeline.
    # -------------------------------------------------

    enforce_status_evidence_consistency(
        result=result,
    )

    # -------------------------------------------------
    # Step 3:
    # Strict structural validation.
    #
    # Unknown requirement IDs, unknown evidence IDs,
    # malformed statuses, GAP with evidence, etc.
    # still raise errors.
    # -------------------------------------------------

    validate_result(
        result=result,
        requirements=requirements,
        evidence=evidence,
    )

    add_requirement_metadata(
        result=result,
        requirements=requirements,
    )

    add_summary(
        result
    )

    result[
        "job_title"
    ] = jd_analysis.get(
        "job_title"
    )

    return result


if __name__ == "__main__":
    example_analysis = {
        "job_title": (
            "Working Student AI Engineer"
        ),
        "must_have_skills": [
            "Python",
            "machine learning",
            "NLP",
            "RAG",
        ],
        "nice_to_have_skills": [
            "Docker",
            "LangGraph",
        ],
        "responsibilities": [
            (
                "Develop and evaluate "
                "LLM-based applications"
            ),
            (
                "Build retrieval-augmented "
                "generation workflows"
            ),
            (
                "Analyze model failures and "
                "improve system reliability"
            ),
        ],
        "experience_requirements": [
            (
                "hands-on experience with "
                "machine learning projects"
            )
        ],
        "education_requirements": [],
        "languages": [],
        "keywords": [
            "Python",
            "LLM",
            "RAG",
            "Docker",
            "evaluation",
        ],
    }

    evidence_retriever = (
        EvidenceRetriever()
    )

    try:
        evidence_package = (
            evidence_retriever.retrieve(
                jd_analysis=example_analysis,
                max_queries=8,
                per_query_top_k=5,
                candidate_k=10,
                final_top_k=10,
            )
        )

        result = analyze_match_gap(
            jd_analysis=example_analysis,
            evidence_package=(
                evidence_package
            ),
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

    finally:
        evidence_retriever.close()