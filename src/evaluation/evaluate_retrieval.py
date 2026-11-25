import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from src.retrieval.bm25_retriever import load_profile_documents
from src.retrieval.hybrid_retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = (
    PROJECT_ROOT
    / "evaluation"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_eval_cases(
    eval_path: Path,
) -> List[Dict[str, Any]]:
    """
    Load an evidence-level retrieval evaluation dataset.
    """
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {eval_path}"
        )

    with eval_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def resolve_relevance_labels(
    eval_case: Dict[str, Any],
    corpus: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Convert human-readable text labels into:

        evidence_id -> relevance grade

    Each 'contains' phrase must identify exactly one
    evidence unit.
    """
    resolved = {}

    for rule in eval_case["relevance"]:
        target = rule["contains"].lower()
        grade = int(rule["grade"])

        matches = [
            document
            for document in corpus
            if target in document["text"].lower()
        ]

        if not matches:
            raise ValueError(
                f"\nEvaluation label not found.\n"
                f"Case: {eval_case['id']}\n"
                f"Text: {rule['contains']}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"\nEvaluation label is ambiguous.\n"
                f"Case: {eval_case['id']}\n"
                f"Text: {rule['contains']}\n"
                f"Matched {len(matches)} documents."
            )

        evidence_id = matches[0]["evidence_id"]

        resolved[evidence_id] = grade

    return resolved


def hit_at_k(
    results: List[Dict[str, Any]],
    relevant_ids: set,
    k: int,
) -> float:
    retrieved = {
        result["evidence_id"]
        for result in results[:k]
    }

    return float(
        bool(retrieved & relevant_ids)
    )


def recall_at_k(
    results: List[Dict[str, Any]],
    relevant_ids: set,
    k: int,
) -> float:
    if not relevant_ids:
        return 0.0

    retrieved = {
        result["evidence_id"]
        for result in results[:k]
    }

    return (
        len(retrieved & relevant_ids)
        / len(relevant_ids)
    )


def reciprocal_rank(
    results: List[Dict[str, Any]],
    relevant_ids: set,
) -> float:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result["evidence_id"] in relevant_ids:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    results: List[Dict[str, Any]],
    relevance: Dict[str, int],
    k: int,
) -> float:
    dcg = 0.0

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):
        grade = relevance.get(
            result["evidence_id"],
            0,
        )

        gain = (2 ** grade) - 1
        discount = math.log2(rank + 1)

        dcg += gain / discount

    return dcg


def ndcg_at_k(
    results: List[Dict[str, Any]],
    relevance: Dict[str, int],
    k: int,
) -> float:
    actual_dcg = dcg_at_k(
        results,
        relevance,
        k,
    )

    ideal_grades = sorted(
        relevance.values(),
        reverse=True,
    )[:k]

    ideal_dcg = 0.0

    for rank, grade in enumerate(
        ideal_grades,
        start=1,
    ):
        gain = (2 ** grade) - 1
        discount = math.log2(rank + 1)

        ideal_dcg += gain / discount

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_results(
    results: List[Dict[str, Any]],
    relevance: Dict[str, int],
) -> Dict[str, float]:
    relevant_ids = set(
        relevance.keys()
    )

    return {
        "hit@1": hit_at_k(
            results,
            relevant_ids,
            1,
        ),
        "hit@3": hit_at_k(
            results,
            relevant_ids,
            3,
        ),
        "recall@5": recall_at_k(
            results,
            relevant_ids,
            5,
        ),
        "mrr": reciprocal_rank(
            results,
            relevant_ids,
        ),
        "ndcg@5": ndcg_at_k(
            results,
            relevance,
            5,
        ),
    }


def average_metrics(
    all_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    if not all_metrics:
        return {}

    keys = all_metrics[0].keys()

    return {
        key: sum(
            metric[key]
            for metric in all_metrics
        ) / len(all_metrics)
        for key in keys
    }


def print_summary(
    name: str,
    metrics: Dict[str, float],
) -> None:
    print(
        f"{name:<10} | "
        f"Hit@1={metrics['hit@1']:.3f} | "
        f"Hit@3={metrics['hit@3']:.3f} | "
        f"Recall@5={metrics['recall@5']:.3f} | "
        f"MRR={metrics['mrr']:.3f} | "
        f"nDCG@5={metrics['ndcg@5']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options so the same evaluator
    can run multiple retrieval benchmarks.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BM25, dense and hybrid "
            "candidate-evidence retrieval."
        )
    )

    parser.add_argument(
        "--eval-file",
        default="retrieval_eval.json",
        help=(
            "Evaluation JSON filename inside "
            "the evaluation directory."
        ),
    )

    parser.add_argument(
        "--output",
        default="baseline_v1.json",
        help=(
            "Output JSON filename inside "
            "evaluation/results."
        ),
    )

    return parser.parse_args()


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

    eval_cases = load_eval_cases(
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

    # HybridRetriever already owns one BM25 retriever
    # and one Dense retriever.
    # Reuse them to avoid opening Qdrant twice.
    hybrid = HybridRetriever()

    bm25 = hybrid.bm25
    dense = hybrid.dense

    system_metrics = {
        "BM25": [],
        "Dense": [],
        "Hybrid": [],
    }

    case_results = []

    try:
        for case in eval_cases:
            relevance = resolve_relevance_labels(
                case,
                corpus,
            )

            query = case["query"]

            bm25_results = bm25.search(
                query,
                top_k=10,
            )

            dense_results = dense.search(
                query,
                top_k=10,
            )

            hybrid_results = hybrid.search(
                query,
                top_k=10,
                candidate_k=10,
            )

            bm25_metrics = evaluate_results(
                bm25_results,
                relevance,
            )

            dense_metrics = evaluate_results(
                dense_results,
                relevance,
            )

            hybrid_metrics = evaluate_results(
                hybrid_results,
                relevance,
            )

            case_results.append({
                "id": case["id"],
                "category": case["category"],
                "query": query,
                "metrics": {
                    "BM25": bm25_metrics,
                    "Dense": dense_metrics,
                    "Hybrid": hybrid_metrics,
                },
            })

            system_metrics["BM25"].append(
                bm25_metrics
            )

            system_metrics["Dense"].append(
                dense_metrics
            )

            system_metrics["Hybrid"].append(
                hybrid_metrics
            )

            print("\n" + "=" * 100)

            print(
                f"{case['id']} "
                f"[{case['category']}]"
            )

            print(
                f"Query: {query}"
            )

            print_summary(
                "BM25",
                bm25_metrics,
            )

            print_summary(
                "Dense",
                dense_metrics,
            )

            print_summary(
                "Hybrid",
                hybrid_metrics,
            )

        print("\n" + "#" * 100)
        print("OVERALL RESULTS")
        print("#" * 100)

        overall_results = {}

        for name, metrics_list in system_metrics.items():
            summary = average_metrics(
                metrics_list
            )

            overall_results[name] = summary

            print_summary(
                name,
                summary,
            )

        output = {
            "evaluation_file": eval_path.name,
            "corpus_size": len(corpus),
            "query_count": len(eval_cases),

            "systems": {
                "BM25": "BM25Okapi",
                "Dense": (
                    "BAAI/bge-small-en-v1.5 + Qdrant"
                ),
                "Hybrid": (
                    "BM25 + Dense + Weighted Reciprocal Rank Fusion"
                ),
            },

            "hybrid_config": {
                "rrf_k": hybrid.rrf_k,
                "bm25_weight": hybrid.bm25_weight,
                "dense_weight": hybrid.dense_weight,
                "candidate_k": 10,
            },

            "overall": overall_results,
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
            f"\nSaved evaluation results to: "
            f"{output_path}"
        )

    finally:
        hybrid.close()


if __name__ == "__main__":
    main()