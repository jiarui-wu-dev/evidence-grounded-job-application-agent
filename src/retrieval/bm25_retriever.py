import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from src.retrieval.profile_documents import (
    load_profile_documents,
)


STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
}


def tokenize(
    text: str,
) -> List[str]:
    """
    Tokenize technical job-search text while
    removing common English stopwords that add
    lexical noise.
    """

    tokens = re.findall(
        r"[a-zA-Z0-9][a-zA-Z0-9_+#./-]*",
        text.lower(),
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


class BM25Retriever:
    """
    Keyword-based retriever over the unified
    candidate evidence corpus.
    """

    def __init__(self):
        self.documents = (
            load_profile_documents()
        )

        if not self.documents:
            raise RuntimeError(
                "No profile documents "
                "available for BM25 retrieval."
            )

        self.tokenized_corpus = [
            tokenize(
                document["text"]
            )
            for document
            in self.documents
        ]

        self.index = BM25Okapi(
            self.tokenized_corpus
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        query_tokens = tokenize(
            query
        )

        scores = self.index.get_scores(
            query_tokens
        )

        ranked_indices = [
            index
            for index in sorted(
                range(
                    len(scores)
                ),
                key=lambda index: (
                    scores[index]
                ),
                reverse=True,
            )
            if scores[index] > 0
        ][:top_k]

        results = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            document = self.documents[
                index
            ]

            results.append({
                "evidence_id": document[
                    "evidence_id"
                ],

                "project_id": document[
                    "project_id"
                ],

                "project_title": document[
                    "project_title"
                ],

                "type": document[
                    "type"
                ],

                "entity_type": document.get(
                    "entity_type",
                    "project",
                ),

                "evidence_status": (
                    document.get(
                        "evidence_status",
                        "verified",
                    )
                ),

                "text": document[
                    "text"
                ],

                "rank": rank,

                "score": float(
                    scores[index]
                ),

                "retriever": "bm25",
            })

        return results


if __name__ == "__main__":
    retriever = BM25Retriever()

    test_queries = [
        "Spring Boot REST API Docker",
        "LoRA Whisper domain adaptation",
        "automated validation testing",
        "commercial software testing authentication",
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
                f"Rank {result['rank']} | "
                f"BM25={result['score']:.4f}"
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