import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_retrieval import (
    average_metrics,
    evaluate_results,
    print_summary,
    resolve_relevance_labels,
)
from src.retrieval.bm25_retriever import (
    load_profile_documents,
)
from src.retrieval.reranked_retriever import (
    RerankedRetriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = (
    PROJECT_ROOT
    / "evaluation"
)

RESULTS_DIR = (
    EVALUATION_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate-union + "
            "cross-encoder reranked retrieval."
        )
    )

    parser.add_argument(
        "--eval-file",
        default="retrieval_challenge.json",
    )

    parser.add_argument(
        "--output",
        default="reranked_v1.json",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
    )

    return parser.parse_args()


def load_json(
    path: Path,
) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    args = parse_args()

    eval_path = (
        EVALUATION_DIR
        / args.eval_file
    )

    output_path = (
        RESULTS_DIR
        / args.output
    )

    corpus = load_profile_documents()

    eval_cases = load_json(
        eval_path
    )

    print(
        f"Evaluation file: {eval_path.name}"
    )

    print(
        f"Loaded {len(corpus)} evidence units."
    )

    print(
        f"Loaded {len(eval_cases)} evaluation queries."
    )

    retriever = RerankedRetriever()

    all_metrics = []
    case_results = []

    try:
        for case in eval_cases:
            relevance = resolve_relevance_labels(
                case,
                corpus,
            )

            query = case["query"]

            results = retriever.search(
                query=query,
                top_k=10,
                candidate_k=args.candidate_k,
            )

            metrics = evaluate_results(
                results,
                relevance,
            )

            all_metrics.append(
                metrics
            )

            case_results.append({
                "id": case["id"],
                "category": case["category"],
                "query": query,
                "metrics": metrics,
                "top_results": [
                    {
                        "rank": result["rank"],
                        "evidence_id": (
                            result["evidence_id"]
                        ),
                        "project_id": (
                            result["project_id"]
                        ),
                        "type": result["type"],
                        "reranker_score": (
                            result[
                                "reranker_score"
                            ]
                        ),
                        "bm25_rank": (
                            result["bm25_rank"]
                        ),
                        "dense_rank": (
                            result["dense_rank"]
                        ),
                        "text": result["text"],
                    }
                    for result in results[:5]
                ],
            })

            print(
                "\n"
                + "=" * 100
            )

            print(
                f"{case['id']} "
                f"[{case['category']}]"
            )

            print(
                f"Query: {query}"
            )

            print_summary(
                "Reranked",
                metrics,
            )

        overall = average_metrics(
            all_metrics
        )

        print(
            "\n"
            + "#" * 100
        )

        print(
            "OVERALL RESULTS"
        )

        print(
            "#" * 100
        )

        print_summary(
            "Reranked",
            overall,
        )

        output = {
            "evaluation_file": (
                eval_path.name
            ),
            "corpus_size": len(corpus),
            "query_count": len(eval_cases),
            "system": (
                "BM25 + Dense Candidate Union "
                "+ CrossEncoder Reranker"
            ),
            "reranker_model": (
                "cross-encoder/"
                "ms-marco-MiniLM-L-6-v2"
            ),
            "candidate_k": (
                args.candidate_k
            ),
            "overall": overall,
            "cases": case_results,
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"\nSaved reranked results to: "
            f"{output_path}"
        )

    finally:
        retriever.close()


if __name__ == "__main__":
    main()