from typing import Any, Dict, List

from src.retrieval.candidate_retriever import (
    CandidateRetriever,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
)


class RerankedRetriever:
    """
    Two-stage retrieval pipeline:

    1. BM25 + dense candidate generation
    2. Cross-encoder relevance reranking
    """

    def __init__(self):
        self.candidate_retriever = (
            CandidateRetriever()
        )

        self.reranker = (
            CrossEncoderReranker()
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> List[Dict[str, Any]]:
        candidates = (
            self.candidate_retriever.search(
                query=query,
                candidate_k=candidate_k,
            )
        )

        return self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    def close(self) -> None:
        self.candidate_retriever.close()


if __name__ == "__main__":
    retriever = RerankedRetriever()

    try:
        test_queries = [
            (
                "Maintain durable backend data while "
                "safely evolving the database structure "
                "between application versions."
            ),
            (
                "Use model-guided experiment selection "
                "to explore an expensive simulation "
                "space more efficiently."
            ),
            (
                "systematically investigate where a "
                "trained model fails and why"
            ),
        ]

        for query in test_queries:
            print("\n" + "=" * 100)
            print(
                f"QUERY: {query}"
            )

            results = retriever.search(
                query=query,
                top_k=5,
                candidate_k=10,
            )

            for result in results:
                print("-" * 100)

                print(
                    f"Rank {result['rank']} | "
                    f"Reranker="
                    f"{result['reranker_score']:.4f}"
                )

                print(
                    f"BM25 rank="
                    f"{result['bm25_rank']} | "
                    f"Dense rank="
                    f"{result['dense_rank']}"
                )

                print(
                    result["evidence_id"]
                )

                print(
                    result["text"]
                )

    finally:
        retriever.close()