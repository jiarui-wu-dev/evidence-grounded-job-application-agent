from typing import Any, Dict, List

from src.analysis.query_builder import (
    build_retrieval_queries,
)
from src.retrieval.reranked_retriever import (
    RerankedRetriever,
)


PROVISIONAL_WEIGHT = 0.35

MAX_PROVISIONAL_PER_QUERY = 2
MAX_PROVISIONAL_FINAL = 2


def _confidence_weight(
    evidence_status: str,
) -> float:
    """
    Weight evidence by confidence level.

    Verified evidence receives full weight.

    Provisional evidence remains retrievable,
    but contributes less to cross-query
    aggregation.
    """

    if evidence_status == "provisional":
        return PROVISIONAL_WEIGHT

    return 1.0


def _select_per_query_results(
    results: List[Dict[str, Any]],
    top_k: int,
    max_provisional: int,
) -> List[Dict[str, Any]]:
    """
    Preserve reranker order while preventing
    provisional evidence from dominating a
    single retrieval query.

    We fetch a larger reranked pool first,
    then keep at most a limited number of
    provisional evidence items.
    """

    selected = []
    provisional_count = 0

    for result in results:
        evidence_status = result.get(
            "evidence_status",
            "verified",
        )

        if (
            evidence_status
            == "provisional"
        ):
            if (
                provisional_count
                >= max_provisional
            ):
                continue

            provisional_count += 1

        selected.append(
            result
        )

        if len(selected) >= top_k:
            break

    return selected


def _select_final_evidence(
    evidence: List[Dict[str, Any]],
    top_k: int,
    max_provisional: int,
) -> List[Dict[str, Any]]:
    """
    Build the final evidence package while
    limiting provisional evidence.

    Provisional professional experience may
    still provide useful context, but cannot
    crowd verified technical evidence out of
    the downstream Match/Gap context.
    """

    selected = []
    provisional_count = 0

    for item in evidence:
        evidence_status = item.get(
            "evidence_status",
            "verified",
        )

        if (
            evidence_status
            == "provisional"
        ):
            if (
                provisional_count
                >= max_provisional
            ):
                continue

            provisional_count += 1

        selected.append(
            item
        )

        if len(selected) >= top_k:
            break

    return selected


class EvidenceRetriever:
    """
    Retrieve and aggregate candidate evidence for
    a structured job-description analysis.

    Pipeline:
        JD analysis
        -> focused retrieval queries
        -> larger reranked candidate pool
        -> confidence-aware per-query selection
        -> cross-query evidence aggregation
        -> confidence-aware final evidence package
    """

    def __init__(self):
        self.retriever = (
            RerankedRetriever()
        )

    def retrieve(
        self,
        jd_analysis: Dict[str, Any],
        max_queries: int = 8,
        per_query_top_k: int = 5,
        candidate_k: int = 10,
        final_top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Build focused queries from the JD analysis,
        retrieve evidence for each query, and aggregate
        duplicate evidence across queries.
        """

        queries = (
            build_retrieval_queries(
                jd_analysis=jd_analysis,
                max_queries=max_queries,
            )
        )

        aggregated = {}
        query_results = []

        for query in queries:

            # Retrieve a larger reranked pool first.
            #
            # We need extra candidates because some
            # high-ranked provisional items may be
            # filtered before aggregation.
            rerank_pool_k = max(
                per_query_top_k * 2,
                per_query_top_k + 3,
            )

            raw_results = (
                self.retriever.search(
                    query=query,
                    top_k=rerank_pool_k,
                    candidate_k=(
                        candidate_k
                    ),
                )
            )

            results = (
                _select_per_query_results(
                    results=raw_results,
                    top_k=(
                        per_query_top_k
                    ),
                    max_provisional=(
                        MAX_PROVISIONAL_PER_QUERY
                    ),
                )
            )

            query_results.append({
                "query": query,
                "results": results,
            })

            for result in results:
                evidence_id = result[
                    "evidence_id"
                ]

                rank = result[
                    "rank"
                ]

                evidence_status = (
                    result.get(
                        "evidence_status",
                        "verified",
                    )
                )

                confidence_weight = (
                    _confidence_weight(
                        evidence_status
                    )
                )

                # Reciprocal-rank contribution:
                #
                # rank 1 -> 1.0
                # rank 2 -> 0.5
                # rank 3 -> 0.333...
                rank_score = (
                    1.0 / rank
                )

                if (
                    evidence_id
                    not in aggregated
                ):
                    aggregated[
                        evidence_id
                    ] = {
                        "evidence_id": (
                            evidence_id
                        ),

                        "project_id": (
                            result[
                                "project_id"
                            ]
                        ),

                        "project_title": (
                            result[
                                "project_title"
                            ]
                        ),

                        "type": result[
                            "type"
                        ],

                        "entity_type": (
                            result.get(
                                "entity_type",
                                "project",
                            )
                        ),

                        "evidence_status": (
                            evidence_status
                        ),

                        "text": result[
                            "text"
                        ],

                        "matched_queries": [],

                        "support_count": 0,

                        "aggregate_rank_score": (
                            0.0
                        ),

                        "weighted_support_score": (
                            0.0
                        ),

                        "weighted_rank_score": (
                            0.0
                        ),

                        "best_rank": rank,

                        "best_reranker_score": (
                            result[
                                "reranker_score"
                            ]
                        ),
                    }

                item = aggregated[
                    evidence_id
                ]

                if (
                    query
                    not in item[
                        "matched_queries"
                    ]
                ):
                    item[
                        "matched_queries"
                    ].append(
                        query
                    )

                    item[
                        "support_count"
                    ] += 1

                    item[
                        "weighted_support_score"
                    ] += confidence_weight

                item[
                    "aggregate_rank_score"
                ] += rank_score

                item[
                    "weighted_rank_score"
                ] += (
                    rank_score
                    * confidence_weight
                )

                item[
                    "best_rank"
                ] = min(
                    item[
                        "best_rank"
                    ],
                    rank,
                )

                item[
                    "best_reranker_score"
                ] = max(
                    item[
                        "best_reranker_score"
                    ],
                    result[
                        "reranker_score"
                    ],
                )

        evidence = list(
            aggregated.values()
        )

        # -------------------------------------------------
        # Confidence-aware aggregation.
        #
        # Raw support_count is retained for diagnostics,
        # but final ranking uses confidence-weighted
        # support first.
        # -------------------------------------------------

        evidence.sort(
            key=lambda item: (
                item[
                    "weighted_support_score"
                ],
                item[
                    "weighted_rank_score"
                ],
                item[
                    "support_count"
                ],
                -item[
                    "best_rank"
                ],
            ),
            reverse=True,
        )

        evidence = (
            _select_final_evidence(
                evidence=evidence,
                top_k=final_top_k,
                max_provisional=(
                    MAX_PROVISIONAL_FINAL
                ),
            )
        )

        return {
            "job_title": (
                jd_analysis.get(
                    "job_title"
                )
            ),

            "queries": queries,

            "query_count": len(
                queries
            ),

            "evidence_count": len(
                evidence
            ),

            "evidence": evidence,

            "query_results": (
                query_results
            ),
        }

    def close(self) -> None:
        self.retriever.close()


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

        "education_requirements": [
            (
                "Currently enrolled in a "
                "technical degree"
            )
        ],

        "languages": [
            "English"
        ],

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
        package = (
            evidence_retriever.retrieve(
                jd_analysis=example_analysis,
                max_queries=8,
                per_query_top_k=5,
                candidate_k=10,
                final_top_k=10,
            )
        )

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"JOB: "
            f"{package['job_title']}"
        )

        print(
            "\nRETRIEVAL QUERIES"
        )

        for index, query in enumerate(
            package["queries"],
            start=1,
        ):
            print(
                f"{index}. {query}"
            )

        print(
            "\n"
            + "=" * 100
        )

        print(
            "AGGREGATED EVIDENCE"
        )

        for rank, item in enumerate(
            package[
                "evidence"
            ],
            start=1,
        ):
            print(
                "\n"
                + "-" * 100
            )

            print(
                f"Evidence Rank {rank}"
            )

            print(
                f"ID: "
                f"{item['evidence_id']}"
            )

            print(
                f"Source: "
                f"{item['project_title']}"
            )

            print(
                f"Entity type: "
                f"{item['entity_type']}"
            )

            print(
                f"Evidence status: "
                f"{item['evidence_status']}"
            )

            print(
                f"Support count: "
                f"{item['support_count']}"
            )

            print(
                f"Weighted support score: "
                f"{item['weighted_support_score']:.3f}"
            )

            print(
                f"Aggregate rank score: "
                f"{item['aggregate_rank_score']:.3f}"
            )

            print(
                f"Weighted rank score: "
                f"{item['weighted_rank_score']:.3f}"
            )

            print(
                f"Best retrieval rank: "
                f"{item['best_rank']}"
            )

            print(
                "Matched queries:"
            )

            for query in item[
                "matched_queries"
            ]:
                print(
                    f"  - {query}"
                )

            print(
                "Evidence:"
            )

            print(
                item["text"]
            )

    finally:
        evidence_retriever.close()