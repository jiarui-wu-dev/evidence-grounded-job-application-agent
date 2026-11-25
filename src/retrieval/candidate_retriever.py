from typing import Any, Dict, List

from src.retrieval.bm25_retriever import (
    BM25Retriever,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
)


class CandidateRetriever:
    """
    Retrieve a high-recall candidate pool by combining
    BM25 lexical retrieval and dense semantic retrieval.

    This class only generates candidates.
    Final ranking is handled by the reranker.

    Candidate metadata such as entity_type and
    evidence_status is preserved for downstream
    evidence reasoning.
    """

    def __init__(self):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()

    def search(
        self,
        query: str,
        candidate_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        bm25_results = self.bm25.search(
            query=query,
            top_k=candidate_k,
        )

        dense_results = self.dense.search(
            query=query,
            top_k=candidate_k,
        )

        candidates = {}

        # -------------------------------------------------
        # Add BM25 candidates.
        # -------------------------------------------------

        for result in bm25_results:
            evidence_id = result[
                "evidence_id"
            ]

            candidates[evidence_id] = {
                "evidence_id": evidence_id,

                "project_id": result[
                    "project_id"
                ],

                "project_title": result[
                    "project_title"
                ],

                "type": result[
                    "type"
                ],

                "entity_type": result.get(
                    "entity_type",
                    "project",
                ),

                "evidence_status": (
                    result.get(
                        "evidence_status",
                        "verified",
                    )
                ),

                "text": result[
                    "text"
                ],

                "bm25_rank": result[
                    "rank"
                ],

                "bm25_score": result[
                    "score"
                ],

                "dense_rank": None,

                "dense_score": None,
            }

        # -------------------------------------------------
        # Merge Dense candidates.
        # -------------------------------------------------

        for result in dense_results:
            evidence_id = result[
                "evidence_id"
            ]

            if (
                evidence_id
                not in candidates
            ):
                candidates[
                    evidence_id
                ] = {
                    "evidence_id": (
                        evidence_id
                    ),

                    "project_id": result[
                        "project_id"
                    ],

                    "project_title": result[
                        "project_title"
                    ],

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
                        result.get(
                            "evidence_status",
                            "verified",
                        )
                    ),

                    "text": result[
                        "text"
                    ],

                    "bm25_rank": None,

                    "bm25_score": None,

                    "dense_rank": result[
                        "rank"
                    ],

                    "dense_score": result[
                        "score"
                    ],
                }

            else:
                candidates[
                    evidence_id
                ][
                    "dense_rank"
                ] = result[
                    "rank"
                ]

                candidates[
                    evidence_id
                ][
                    "dense_score"
                ] = result[
                    "score"
                ]

        candidate_list = list(
            candidates.values()
        )

        def best_source_rank(
            candidate: Dict[str, Any],
        ) -> int:
            ranks = [
                rank
                for rank in [
                    candidate[
                        "bm25_rank"
                    ],
                    candidate[
                        "dense_rank"
                    ],
                ]
                if rank is not None
            ]

            return min(ranks)

        candidate_list.sort(
            key=best_source_rank
        )

        return candidate_list

    def close(self) -> None:
        self.dense.close()


if __name__ == "__main__":
    retriever = (
        CandidateRetriever()
    )

    try:
        test_queries = [
            (
                "automated validation "
                "of generated code"
            ),
            (
                "commercial software "
                "testing authentication"
            ),
        ]

        for query in test_queries:
            print(
                "\n"
                + "=" * 80
            )

            print(
                f"QUERY: {query}"
            )

            results = (
                retriever.search(
                    query=query,
                    candidate_k=10,
                )
            )

            for rank, result in enumerate(
                results,
                start=1,
            ):
                print(
                    "-" * 80
                )

                print(
                    f"Candidate {rank}"
                )

                print(
                    f"{result['project_title']} "
                    f"[{result['entity_type']}] "
                    f"[{result['evidence_status']}]"
                )

                print(
                    f"BM25 rank: "
                    f"{result['bm25_rank']} | "
                    f"Dense rank: "
                    f"{result['dense_rank']}"
                )

                print(
                    result["text"]
                )

    finally:
        retriever.close()