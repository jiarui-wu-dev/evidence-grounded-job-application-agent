from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import (
    QdrantClient,
    models,
)


MODEL_NAME = (
    "BAAI/bge-small-en-v1.5"
)

COLLECTION_NAME = (
    "candidate_profile"
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

QDRANT_PATH = (
    PROJECT_ROOT
    / "qdrant_data"
)


class DenseRetriever:
    """
    Semantic retriever over candidate evidence
    units stored in Qdrant.
    """

    def __init__(self):
        self.client = (
            QdrantClient(
                path=str(
                    QDRANT_PATH
                )
            )
        )

        if not self.client.collection_exists(
            COLLECTION_NAME
        ):
            raise RuntimeError(
                f"Qdrant collection "
                f"'{COLLECTION_NAME}' "
                "does not exist. "
                "Run build_index first."
            )

        count = self.client.count(
            collection_name=(
                COLLECTION_NAME
            ),
            exact=True,
        ).count

        if count == 0:
            raise RuntimeError(
                f"Qdrant collection "
                f"'{COLLECTION_NAME}' "
                "is empty. "
                "Run build_index first."
            )

    def close(self) -> None:
        """
        Explicitly close the Qdrant client and
        release local resources.
        """

        self.client.close()

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        response = (
            self.client.query_points(
                collection_name=(
                    COLLECTION_NAME
                ),
                query=models.Document(
                    text=query,
                    model=MODEL_NAME,
                ),
                limit=top_k,
                with_payload=True,
            )
        )

        results = []

        for rank, point in enumerate(
            response.points,
            start=1,
        ):
            payload = (
                point.payload
                or {}
            )

            results.append({
                "evidence_id": payload.get(
                    "evidence_id"
                ),

                "project_id": payload.get(
                    "project_id"
                ),

                "project_title": (
                    payload.get(
                        "project_title"
                    )
                ),

                "type": payload.get(
                    "type"
                ),

                "entity_type": payload.get(
                    "entity_type",
                    "project",
                ),

                "evidence_status": (
                    payload.get(
                        "evidence_status",
                        "verified",
                    )
                ),

                "text": payload.get(
                    "text"
                ),

                "rank": rank,

                "score": float(
                    point.score
                ),

                "retriever": "dense",
            })

        return results


if __name__ == "__main__":
    retriever = DenseRetriever()

    try:
        test_queries = [
            (
                "backend API development "
                "and containerization"
            ),
            (
                "parameter-efficient "
                "adaptation of speech models"
            ),
            (
                "automated testing and "
                "validation of generated code"
            ),
            (
                "commercial software testing "
                "and authentication"
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

            results = retriever.search(
                query=query,
                top_k=5,
            )

            for result in results:
                print(
                    "-" * 80
                )

                print(
                    f"Rank "
                    f"{result['rank']} | "
                    f"Dense="
                    f"{result['score']:.4f}"
                )

                print(
                    f"{result['project_title']} "
                    f"[{result['entity_type']}] "
                    f"[{result['evidence_status']}] "
                    f"[{result['type']}]"
                )

                print(
                    result["text"]
                )

    finally:
        retriever.close()