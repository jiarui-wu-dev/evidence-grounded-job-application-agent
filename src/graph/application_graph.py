from typing import Any, Dict, List, TypedDict

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from src.analysis.jd_analyzer import (
    analyze_jd,
)

from src.analysis.query_builder import (
    build_retrieval_queries,
)

from src.analysis.evidence_retriever import (
    EvidenceRetriever,
)

from src.analysis.match_gap_analyzer import (
    analyze_match_gap,
)

from src.analysis.application_strategy import (
    build_application_strategy,
)

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


# ============================================================
# State
# ============================================================

class ApplicationState(
    TypedDict,
    total=False,
):
    """
    Shared state for the complete job-application workflow.

    Raw JD
    -> JD analysis
    -> query building
    -> evidence retrieval
    -> match/gap analysis
    -> application strategy
    -> source expansion
    -> CV generation
    -> evidence validation
    -> revision loop
    -> final CV
    """

    # Raw input
    jd_text: str

    # Upstream analysis
    jd_analysis: Dict[str, Any]
    queries: List[str]
    evidence_package: Dict[str, Any]
    match_gap: Dict[str, Any]
    application_strategy: Dict[str, Any]
    cv_composition: Dict[str, Any]
    expanded_sources: List[Dict[str, Any]]

    # CV generation / validation
    cv_draft: Dict[str, Any]
    cv_validation: Dict[str, Any]

    # Revision control
    revision_round: int
    max_revision_rounds: int
    pending_revalidation_claim_ids: List[str] | None

    # Final output
    final_cv: Dict[str, Any]
    final_status: str
    warnings: List[str]


# ============================================================
# Helpers
# ============================================================

def count_problematic_claims(
    validation: Dict[str, Any],
) -> int:
    """
    Count claims that still require revision.
    """

    summary = validation.get(
        "summary",
        {},
    )

    if (
        "OVERSTATED" in summary
        and "UNSUPPORTED" in summary
    ):
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

    return sum(
        1
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
    )


def collect_problematic_claim_ids(
    validation: Dict[str, Any],
) -> List[str]:

    return [
        result[
            "claim_id"
        ]
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


# ============================================================
# Upstream Nodes
# ============================================================

def analyze_jd_node(
    state: ApplicationState,
) -> Dict[str, Any]:
    """
    Convert raw JD text into structured requirements.
    """

    print(
        "\n[LANGGRAPH] NODE: analyze_jd"
    )

    jd_text = state.get(
        "jd_text",
        "",
    )

    if not isinstance(
        jd_text,
        str,
    ) or not jd_text.strip():

        raise ValueError(
            "analyze_jd requires "
            "non-empty jd_text."
        )

    result = analyze_jd(
        jd_text
    )

    print(
        "[LANGGRAPH] JD analyzed."
    )

    return {
        "jd_analysis": result,
    }


def build_queries_node(
    state: ApplicationState,
) -> Dict[str, Any]:
    """
    Build retrieval queries from structured JD analysis.
    """

    print(
        "\n[LANGGRAPH] NODE: build_queries"
    )

    if "jd_analysis" not in state:
        raise ValueError(
            "build_queries requires "
            "jd_analysis."
        )

    queries = build_retrieval_queries(
        state[
            "jd_analysis"
        ]
    )

    if not queries:
        raise ValueError(
            "Query Builder returned "
            "no retrieval queries."
        )

    return {
        "queries": queries,
    }


def retrieve_evidence_node(
    state: ApplicationState,
) -> Dict[str, Any]:
    """
    Retrieve candidate evidence using the existing
    confidence-aware hybrid retrieval pipeline.
    """

    print(
        "\n[LANGGRAPH] NODE: retrieve_evidence"
    )

    if "jd_analysis" not in state:
        raise ValueError(
            "retrieve_evidence requires "
            "jd_analysis."
        )

    if "queries" not in state:
        raise ValueError(
            "retrieve_evidence requires "
            "queries."
        )

    queries = state[
        "queries"
    ]

    retriever = EvidenceRetriever()

    try:

        evidence_package = retriever.retrieve(
            jd_analysis=(
                state[
                    "jd_analysis"
                ]
            ),
            max_queries=(
                len(
                    queries
                )
            ),
            per_query_top_k=5,
            candidate_k=10,
            final_top_k=10,
        )

    finally:

        retriever.close()

    print(
        "[LANGGRAPH] Retrieval complete."
    )

    return {
        "evidence_package": (
            evidence_package
        ),
    }


def analyze_match_gap_node(
    state: ApplicationState,
) -> Dict[str, Any]:
    """
    Classify each JD requirement as:

    SUPPORTED / PARTIAL / GAP.
    """

    print(
        "\n[LANGGRAPH] NODE: analyze_match_gap"
    )

    required_keys = [
        "jd_analysis",
        "evidence_package",
    ]

    for key in required_keys:

        if key not in state:
            raise ValueError(
                "analyze_match_gap missing "
                f"state key: {key}"
            )

    result = analyze_match_gap(
        jd_analysis=(
            state[
                "jd_analysis"
            ]
        ),
        evidence_package=(
            state[
                "evidence_package"
            ]
        ),
    )

    print(
        "[LANGGRAPH] Match/Gap complete."
    )

    return {
        "match_gap": result,
    }


def build_strategy_node(
    state: ApplicationState,
) -> Dict[str, Any]:
    """
    Build an evidence-grounded application strategy.
    """

    print(
        "\n[LANGGRAPH] NODE: build_strategy"
    )

    required_keys = [
        "jd_analysis",
        "evidence_package",
        "match_gap",
    ]

    for key in required_keys:

        if key not in state:
            raise ValueError(
                "build_strategy missing "
                f"state key: {key}"
            )

    strategy = build_application_strategy(
        jd_analysis=(
            state[
                "jd_analysis"
            ]
        ),
        evidence_package=(
            state[
                "evidence_package"
            ]
        ),
        match_gap_result=(
            state[
                "match_gap"
            ]
        ),
    )

    print(
        "[LANGGRAPH] Strategy generated."
    )

    return {
        "application_strategy": (
            strategy
        ),
    }


def expand_sources_node(
    state: ApplicationState,
) -> Dict[str, Any]:
    """
    Build the final CV composition plan and expand
    every selected source into complete evidence.
    """

    print(
        "\n[LANGGRAPH] NODE: expand_sources"
    )

    if (
        "application_strategy"
        not in state
    ):
        raise ValueError(
            "expand_sources requires "
            "application_strategy."
        )

    # -------------------------------------------------
    # Strategy answers:
    # What is most relevant to this JD?
    #
    # Composition answers:
    # What should actually appear in the CV?
    # -------------------------------------------------

    composition = (
        build_cv_composition(
            application_strategy=(
                state[
                    "application_strategy"
                ]
            )
        )
    )

    expanded_sources = (
        expand_priority_sources(
            composition
        )
    )

    print(
        "[LANGGRAPH] CV composition built."
    )

    return {
        "cv_composition": (
            composition
        ),

        "expanded_sources": (
            expanded_sources
        ),
    }

# ============================================================
# CV Nodes
# ============================================================

def tailor_cv_node(
    state: ApplicationState,
) -> Dict[str, Any]:
    """
    Generate the initial CV draft.
    """

    print(
        "\n[LANGGRAPH] NODE: tailor_cv"
    )

    required_keys = [
        "jd_analysis",
        "match_gap",
        "application_strategy",
        "expanded_sources",
    ]

    for key in required_keys:

        if key not in state:
            raise ValueError(
                "tailor_cv missing "
                f"state key: {key}"
            )

    cv_draft = build_cv_draft(
        jd_analysis=(
            state[
                "jd_analysis"
            ]
        ),
        match_gap_result=(
            state[
                "match_gap"
            ]
        ),
        application_strategy=(
            state[
                "application_strategy"
            ]
        ),
        expanded_sources=(
            state[
                "expanded_sources"
            ]
        ),
    )

    print(
        "[LANGGRAPH] Initial CV "
        "draft generated."
    )

    return {
        "cv_draft": cv_draft,
        "revision_round": 0,
        "pending_revalidation_claim_ids": None,
        "warnings": [],
    }


def validate_cv_node(
    state: ApplicationState,
) -> Dict[str, Any]:
    """
    Semantically validate generated CV claims against
    cited candidate evidence.
    """

    print(
        "\n[LANGGRAPH] NODE: validate_cv"
    )

    if "cv_draft" not in state:
        raise ValueError(
            "validate_cv requires "
            "cv_draft."
        )

    if "expanded_sources" not in state:
        raise ValueError(
            "validate_cv requires "
            "expanded_sources."
        )

    validation = validate_cv_evidence(
        cv_draft=(
            state[
                "cv_draft"
            ]
        ),
        expanded_sources=(
            state[
                "expanded_sources"
            ]
        ),
        # Older checkpoints lack this field and keep the full-validation path.
        changed_claim_ids=state.get("pending_revalidation_claim_ids"),
        previous_validation=state.get("cv_validation"),
    )

    print(
        "[LANGGRAPH] Validation complete."
    )

    return {
        "cv_validation": (
            validation
        ),
        "pending_revalidation_claim_ids": None,
    }


def revise_cv_node(
    state: ApplicationState,
) -> Dict[str, Any]:
    """
    Revise only problematic CV claims.
    """

    current_round = state.get(
        "revision_round",
        0,
    )

    next_round = (
        current_round + 1
    )

    print(
        "\n[LANGGRAPH] NODE: revise_cv"
    )

    if "cv_validation" not in state:
        raise ValueError(
            "revise_cv requires "
            "cv_validation."
        )

    revised_draft = revise_cv_draft(
        cv_draft=(
            state[
                "cv_draft"
            ]
        ),
        validation_result=(
            state[
                "cv_validation"
            ]
        ),
        expanded_sources=(
            state[
                "expanded_sources"
            ]
        ),
    )

    validate_draft_structure(
        draft=(
            revised_draft
        ),
        expanded_sources=(
            state[
                "expanded_sources"
            ]
        ),
    )

    print(
        "[LANGGRAPH] Revised draft "
        "passed structural validation."
    )

    return {
        "cv_draft": (
            revised_draft
        ),
        "pending_revalidation_claim_ids": find_changed_claim_ids(
            previous_draft=state["cv_draft"],
            revised_draft=revised_draft,
            expanded_sources=state["expanded_sources"],
        ),
        "revision_round": (
            next_round
        ),
    }


# ============================================================
# Terminal Nodes
# ============================================================

def finalize_node(
    state: ApplicationState,
) -> Dict[str, Any]:
    """
    Finalize a fully grounded CV.
    """

    print(
        "\n[LANGGRAPH] NODE: finalize"
    )

    print(
        "[LANGGRAPH] All validated "
        "claims are grounded."
    )

    return {
        "final_cv": (
            state[
                "cv_draft"
            ]
        ),
        "final_status": (
            "grounded"
        ),
    }


def finalize_with_warning_node(
    state: ApplicationState,
) -> Dict[str, Any]:
    """
    Stop automatic revision after the configured limit
    and require human review.
    """

    print(
        "\n[LANGGRAPH] NODE: "
        "finalize_with_warning"
    )

    validation = state[
        "cv_validation"
    ]

    claim_ids = (
        collect_problematic_claim_ids(
            validation
        )
    )

    warning = (
        "Maximum revision rounds reached. "
        "The following claims still require "
        "human review: "
        + ", ".join(
            claim_ids
        )
    )

    print(
        "[LANGGRAPH] WARNING:"
    )

    warnings = list(
        state.get(
            "warnings",
            [],
        )
    )

    warnings.append(
        warning
    )

    return {
        "final_cv": (
            state[
                "cv_draft"
            ]
        ),
        "final_status": (
            "needs_human_review"
        ),
        "warnings": warnings,
    }


# ============================================================
# Conditional Routing
# ============================================================

def route_after_validation(
    state: ApplicationState,
) -> str:
    """
    Decide whether to:

    - finalize;
    - revise;
    - stop and require human review.
    """

    validation = state.get(
        "cv_validation"
    )

    if validation is None:
        raise ValueError(
            "Routing requires "
            "cv_validation."
        )

    problem_count = (
        count_problematic_claims(
            validation
        )
    )

    if problem_count == 0:

        print(
            "[LANGGRAPH] ROUTE: "
            "validate_cv -> finalize"
        )

        return "finalize"

    revision_round = state.get(
        "revision_round",
        0,
    )

    max_revision_rounds = (
        state.get(
            "max_revision_rounds",
            2,
        )
    )

    if (
        revision_round
        >= max_revision_rounds
    ):

        print(
            "[LANGGRAPH] ROUTE: "
            "validate_cv -> "
            "finalize_with_warning"
        )

        return (
            "finalize_with_warning"
        )

    print(
        "[LANGGRAPH] ROUTE: "
        "validate_cv -> revise_cv"
    )

    return "revise_cv"


# ============================================================
# Full Application Graph
# ============================================================

def build_application_graph(
    checkpointer=None,
):
    """
    Complete application workflow.

    START
      -> analyze_jd
      -> build_queries
      -> retrieve_evidence
      -> analyze_match_gap
      -> build_strategy
      -> expand_sources
      -> tailor_cv
      -> validate_cv
      -> conditional revision loop
      -> finalize / human review
      -> END
    """

    graph = StateGraph(
        ApplicationState
    )

    graph.add_node(
        "analyze_jd",
        analyze_jd_node,
    )

    graph.add_node(
        "build_queries",
        build_queries_node,
    )

    graph.add_node(
        "retrieve_evidence",
        retrieve_evidence_node,
    )

    graph.add_node(
        "analyze_match_gap",
        analyze_match_gap_node,
    )

    graph.add_node(
        "build_strategy",
        build_strategy_node,
    )

    graph.add_node(
        "expand_sources",
        expand_sources_node,
    )

    graph.add_node(
        "tailor_cv",
        tailor_cv_node,
    )

    graph.add_node(
        "validate_cv",
        validate_cv_node,
    )

    graph.add_node(
        "revise_cv",
        revise_cv_node,
    )

    graph.add_node(
        "finalize",
        finalize_node,
    )

    graph.add_node(
        "finalize_with_warning",
        finalize_with_warning_node,
    )

    graph.add_edge(
        START,
        "analyze_jd",
    )

    graph.add_edge(
        "analyze_jd",
        "build_queries",
    )

    graph.add_edge(
        "build_queries",
        "retrieve_evidence",
    )

    graph.add_edge(
        "retrieve_evidence",
        "analyze_match_gap",
    )

    graph.add_edge(
        "analyze_match_gap",
        "build_strategy",
    )

    graph.add_edge(
        "build_strategy",
        "expand_sources",
    )

    graph.add_edge(
        "expand_sources",
        "tailor_cv",
    )

    graph.add_edge(
        "tailor_cv",
        "validate_cv",
    )

    graph.add_conditional_edges(
        "validate_cv",
        route_after_validation,
        {
            "finalize": (
                "finalize"
            ),
            "revise_cv": (
                "revise_cv"
            ),
            "finalize_with_warning": (
                "finalize_with_warning"
            ),
        },
    )

    graph.add_edge(
        "revise_cv",
        "validate_cv",
    )

    graph.add_edge(
        "finalize",
        END,
    )

    graph.add_edge(
        "finalize_with_warning",
        END,
    )

    return graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# Regression / Resume Graph
# ============================================================

def build_revision_resume_graph(
    checkpointer=None,
):
    """
    Fast regression graph for an already generated and
    validated CV draft.

    START
      -> revise_cv
      -> validate_cv
      -> conditional revision loop
      -> finalize / human review
      -> END
    """

    graph = StateGraph(
        ApplicationState
    )

    graph.add_node(
        "validate_cv",
        validate_cv_node,
    )

    graph.add_node(
        "revise_cv",
        revise_cv_node,
    )

    graph.add_node(
        "finalize",
        finalize_node,
    )

    graph.add_node(
        "finalize_with_warning",
        finalize_with_warning_node,
    )

    graph.add_edge(
        START,
        "revise_cv",
    )

    graph.add_edge(
        "revise_cv",
        "validate_cv",
    )

    graph.add_conditional_edges(
        "validate_cv",
        route_after_validation,
        {
            "finalize": (
                "finalize"
            ),
            "revise_cv": (
                "revise_cv"
            ),
            "finalize_with_warning": (
                "finalize_with_warning"
            ),
        },
    )

    graph.add_edge(
        "finalize",
        END,
    )

    graph.add_edge(
        "finalize_with_warning",
        END,
    )

    return graph.compile(
        checkpointer=checkpointer
)
