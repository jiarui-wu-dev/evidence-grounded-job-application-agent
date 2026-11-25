import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.evaluate_retrieval import (
    resolve_relevance_labels,
)
from src.retrieval.bm25_retriever import (
    load_profile_documents,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze retrieval failure and improvement cases "
            "for BM25, dense and hybrid retrieval."
        )
    )

    parser.add_argument(
        "--eval-file",
        default="retrieval_challenge.json",
        help="Evaluation dataset filename.",
    )

    parser.add_argument(
        "--results-file",
        default="challenge_v1.json",
        help="Saved evaluation results filename.",
    )

    parser.add_argument(
        "--output",
        default="challenge_v1_error_analysis.json",
        help="Error-analysis output filename.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval results to inspect.",
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def format_result(
    result: Dict[str, Any],
    relevance: Dict[str, int],
) -> Dict[str, Any]:
    evidence_id = result["evidence_id"]

    return {
        "rank": result["rank"],
        "evidence_id": evidence_id,
        "project_id": result["project_id"],
        "project_title": result["project_title"],
        "type": result["type"],
        "text": result["text"],
        "relevance_grade": relevance.get(
            evidence_id,
            0,
        ),
    }


def classify_case(
    dense_metrics: Dict[str, float],
    hybrid_metrics: Dict[str, float],
) -> List[str]:
    labels = []

    if (
        hybrid_metrics["recall@5"]
        > dense_metrics["recall@5"]
    ):
        labels.append(
            "hybrid_recall_gain"
        )

    if (
        hybrid_metrics["recall@5"]
        < dense_metrics["recall@5"]
    ):
        labels.append(
            "hybrid_recall_loss"
        )

    if (
        hybrid_metrics["ndcg@5"]
        > dense_metrics["ndcg@5"]
    ):
        labels.append(
            "hybrid_ranking_gain"
        )

    if (
        hybrid_metrics["ndcg@5"]
        < dense_metrics["ndcg@5"]
    ):
        labels.append(
            "hybrid_ranking_loss"
        )

    if (
        hybrid_metrics["hit@1"]
        > dense_metrics["hit@1"]
    ):
        labels.append(
            "hybrid_top1_gain"
        )

    if (
        hybrid_metrics["hit@1"]
        < dense_metrics["hit@1"]
    ):
        labels.append(
            "hybrid_top1_loss"
        )

    if not labels:
        labels.append(
            "same_performance"
        )

    return labels


def print_ranked_results(
    name: str,
    results: List[Dict[str, Any]],
    relevance: Dict[str, int],
) -> None:
    print(
        f"\n{name} TOP RESULTS"
    )

    for result in results:
        evidence_id = result["evidence_id"]
        grade = relevance.get(
            evidence_id,
            0,
        )

        marker = (
            "RELEVANT"
            if grade > 0
            else "noise"
        )

        print(
            f"  Rank {result['rank']} | "
            f"grade={grade} | "
            f"{marker}"
        )

        print(
            f"    {evidence_id}"
        )

        print(
            f"    {result['text']}"
        )


def main() -> None:
    args = parse_args()

    eval_path = (
        EVALUATION_DIR
        / args.eval_file
    )

    results_path = (
        RESULTS_DIR
        / args.results_file
    )

    output_path = (
        RESULTS_DIR
        / args.output
    )

    eval_cases = load_json(
        eval_path
    )

    saved_results = load_json(
        results_path
    )

    corpus = load_profile_documents()

    saved_case_map = {
        case["id"]: case
        for case in saved_results["cases"]
    }

    hybrid = HybridRetriever()

    bm25 = hybrid.bm25
    dense = hybrid.dense

    analysis_cases = []

    try:
        for case in eval_cases:
            case_id = case["id"]

            saved_case = saved_case_map[
                case_id
            ]

            metrics = saved_case[
                "metrics"
            ]

            dense_metrics = metrics[
                "Dense"
            ]

            hybrid_metrics = metrics[
                "Hybrid"
            ]

            labels = classify_case(
                dense_metrics,
                hybrid_metrics,
            )

            # We mainly care about cases where
            # Dense and Hybrid behave differently.
            if labels == [
                "same_performance"
            ]:
                continue

            relevance = (
                resolve_relevance_labels(
                    case,
                    corpus,
                )
            )

            query = case["query"]

            bm25_results = bm25.search(
                query,
                top_k=args.top_k,
            )

            dense_results = dense.search(
                query,
                top_k=args.top_k,
            )

            hybrid_results = hybrid.search(
                query,
                top_k=args.top_k,
                candidate_k=10,
            )

            print(
                "\n"
                + "=" * 100
            )

            print(
                f"{case_id} "
                f"[{case['category']}]"
            )

            print(
                f"Query: {query}"
            )

            print(
                f"Labels: "
                f"{', '.join(labels)}"
            )

            print(
                "\nDense metrics:"
            )

            print(
                json.dumps(
                    dense_metrics,
                    indent=2,
                )
            )

            print(
                "\nHybrid metrics:"
            )

            print(
                json.dumps(
                    hybrid_metrics,
                    indent=2,
                )
            )

            print_ranked_results(
                "BM25",
                bm25_results,
                relevance,
            )

            print_ranked_results(
                "Dense",
                dense_results,
                relevance,
            )

            print_ranked_results(
                "Hybrid",
                hybrid_results,
                relevance,
            )

            analysis_cases.append({
                "id": case_id,
                "category": case[
                    "category"
                ],
                "query": query,
                "labels": labels,
                "metrics": {
                    "Dense": dense_metrics,
                    "Hybrid": hybrid_metrics,
                },
                "bm25_results": [
                    format_result(
                        result,
                        relevance,
                    )
                    for result
                    in bm25_results
                ],
                "dense_results": [
                    format_result(
                        result,
                        relevance,
                    )
                    for result
                    in dense_results
                ],
                "hybrid_results": [
                    format_result(
                        result,
                        relevance,
                    )
                    for result
                    in hybrid_results
                ],
            })

        output = {
            "evaluation_file": (
                args.eval_file
            ),
            "results_file": (
                args.results_file
            ),
            "analyzed_case_count": (
                len(analysis_cases)
            ),
            "cases": analysis_cases,
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
            "\n"
            + "#" * 100
        )

        print(
            f"Analyzed "
            f"{len(analysis_cases)} "
            f"cases."
        )

        print(
            f"Saved error analysis to: "
            f"{output_path}"
        )

    finally:
        hybrid.close()


if __name__ == "__main__":
    main()