from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


DEFAULT_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


class CrossEncoderReranker:
    """
    Rerank retrieved evidence candidates using a
    cross-encoder relevance model.

    Unlike first-stage dense retrieval, the cross-encoder
    jointly evaluates the query and evidence text.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.model_name = model_name

        self.model = CrossEncoder(
            model_name,
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not candidates:
            return []

        pairs = [
            [
                query,
                candidate["text"],
            ]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            reranked.append({
                **candidate,
                "reranker_score": float(score),
            })

        reranked.sort(
            key=lambda item: item[
                "reranker_score"
            ],
            reverse=True,
        )

        final_results = []

        for rank, result in enumerate(
            reranked[:top_k],
            start=1,
        ):
            final_results.append({
                **result,
                "rank": rank,
                "retriever": (
                    "candidate_union_cross_encoder"
                ),
            })

        return final_results