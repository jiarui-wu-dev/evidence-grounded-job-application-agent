from typing import Any, Dict, List

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    """
    Hybrid candidate-evidence retriever.

    Combines:
    - BM25 lexical retrieval
    - Dense semantic retrieval

    using weighted Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 2.0,
    ):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()

        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evidence using both BM25 and dense retrieval,
        then fuse the two ranked lists with weighted RRF.

        Parameters
        ----------
        query:
            Search query.

        top_k:
            Number of final hybrid results to return.

        candidate_k:
            Number of candidates retrieved from each
            individual retriever before fusion.
        """

        if not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        # 1. Retrieve independently
        bm25_results = self.bm25.search(
            query=query,
            top_k=candidate_k,
        )

        dense_results = self.dense.search(
            query=query,
            top_k=candidate_k,
        )

        # 2. Merge by evidence_id
        fused = {}

        for result in bm25_results:
            evidence_id = result["evidence_id"]

            if evidence_id not in fused:
                fused[evidence_id] = {
                    "evidence_id": evidence_id,
                    "project_id": result["project_id"],
                    "project_title": result["project_title"],
                    "type": result["type"],
                    "text": result["text"],
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "dense_rank": None,
                    "dense_score": None,
                }

            rank = result["rank"]

            fused[evidence_id]["rrf_score"] += (
                self.bm25_weight
                / (self.rrf_k + rank)
            )

            fused[evidence_id]["bm25_rank"] = rank
            fused[evidence_id]["bm25_score"] = result["score"]

        for result in dense_results:
            evidence_id = result["evidence_id"]

            if evidence_id not in fused:
                fused[evidence_id] = {
                    "evidence_id": evidence_id,
                    "project_id": result["project_id"],
                    "project_title": result["project_title"],
                    "type": result["type"],
                    "text": result["text"],
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "dense_rank": None,
                    "dense_score": None,
                }

            rank = result["rank"]

            fused[evidence_id]["rrf_score"] += (
                self.dense_weight
                / (self.rrf_k + rank)
            )

            fused[evidence_id]["dense_rank"] = rank
            fused[evidence_id]["dense_score"] = result["score"]

        # 3. Sort by fused weighted-RRF score
        ranked_results = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        # 4. Add final hybrid rank
        final_results = []

        for rank, result in enumerate(
            ranked_results[:top_k],
            start=1,
        ):
            final_results.append({
                **result,
                "rank": rank,
                "retriever": "hybrid",
            })

        return final_results

    def close(self) -> None:
        """
        Release local Qdrant resources.
        """
        self.dense.close()


if __name__ == "__main__":
    retriever = HybridRetriever()

    try:
        test_queries = [
            "Spring Boot REST API Docker",
            "parameter-efficient adaptation of speech models",
            "automated testing and validation of generated code",
        ]

        for query in test_queries:
            print("\n" + "=" * 100)
            print(f"QUERY: {query}")

            results = retriever.search(
                query=query,
                top_k=5,
                candidate_k=10,
            )

            for result in results:
                print("-" * 100)

                print(
                    f"Hybrid Rank {result['rank']} | "
                    f"RRF={result['rrf_score']:.6f}"
                )

                print(
                    f"BM25 rank={result['bm25_rank']} | "
                    f"BM25 score={result['bm25_score']}"
                )

                print(
                    f"Dense rank={result['dense_rank']} | "
                    f"Dense score={result['dense_score']}"
                )

                print(
                    f"{result['project_title']} "
                    f"[{result['type']}]"
                )

                print(result["text"])

    finally:
        retriever.close()