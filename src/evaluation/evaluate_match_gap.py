import json
from pathlib import Path
from typing import Any, Dict, List

from src.analysis.match_gap_analyzer import (
    analyze_match_gap,
)
from src.retrieval.bm25_retriever import (
    load_profile_documents,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "match_gap_eval.json"
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


def load_json(
    path: Path,
) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_corpus_map() -> Dict[str, Dict[str, Any]]:
    documents = load_profile_documents()

    return {
        document["evidence_id"]: document
        for document in documents
    }


def build_evidence_package(
    evidence_ids: List[str],
    corpus_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    evidence = []

    for evidence_id in evidence_ids:
        if evidence_id not in corpus_map:
            raise ValueError(
                f"Unknown benchmark evidence ID: "
                f"{evidence_id}"
            )

        document = corpus_map[
            evidence_id
        ]

        evidence.append({
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
            "text": document[
                "text"
            ],
        })

    return {
        "job_title": (
            "Match Gap Evaluation"
        ),
        "queries": [],
        "query_count": 0,
        "evidence_count": len(
            evidence
        ),
        "evidence": evidence,
        "query_results": [],
    }


def evaluate_attribution(
    predicted_status: str,
    predicted_evidence_ids: List[str],
    acceptable_evidence_ids: List[str],
) -> bool:
    """
    A GAP is correct only when it cites no evidence.

    For SUPPORTED / PARTIAL:
    - at least one evidence ID must be cited
    - every cited ID must belong to the manually
      accepted evidence set
    """

    if predicted_status == "GAP":
        return (
            len(predicted_evidence_ids)
            == 0
        )

    if not predicted_evidence_ids:
        return False

    acceptable = set(
        acceptable_evidence_ids
    )

    predicted = set(
        predicted_evidence_ids
    )

    return (
        bool(predicted & acceptable)
        and predicted.issubset(
            acceptable
        )
    )


def main() -> None:
    cases = load_json(
        EVAL_PATH
    )

    corpus_map = build_corpus_map()

    case_results = []

    correct_status_count = 0
    correct_attribution_count = 0
    fully_correct_count = 0

    confusion = {
        "SUPPORTED": {
            "SUPPORTED": 0,
            "PARTIAL": 0,
            "GAP": 0,
        },
        "PARTIAL": {
            "SUPPORTED": 0,
            "PARTIAL": 0,
            "GAP": 0,
        },
        "GAP": {
            "SUPPORTED": 0,
            "PARTIAL": 0,
            "GAP": 0,
        },
    }

    for case in cases:
        jd_analysis = {
            "job_title": (
                "Match Gap Evaluation"
            ),
            "must_have_skills": [
                case["requirement"]
            ],
            "nice_to_have_skills": [],
            "responsibilities": [],
            "experience_requirements": [],
            "education_requirements": [],
            "languages": [],
            "keywords": [],
        }

        evidence_package = (
            build_evidence_package(
                evidence_ids=case[
                    "evidence_ids"
                ],
                corpus_map=corpus_map,
            )
        )

        result = analyze_match_gap(
            jd_analysis=jd_analysis,
            evidence_package=(
                evidence_package
            ),
        )

        prediction = (
            result["requirements"][0]
        )

        predicted_status = prediction[
            "status"
        ]

        predicted_evidence_ids = (
            prediction[
                "evidence_ids"
            ]
        )

        expected_status = case[
            "expected_status"
        ]

        status_correct = (
            predicted_status
            == expected_status
        )

        attribution_correct = (
            evaluate_attribution(
                predicted_status=(
                    predicted_status
                ),
                predicted_evidence_ids=(
                    predicted_evidence_ids
                ),
                acceptable_evidence_ids=(
                    case[
                        "acceptable_evidence_ids"
                    ]
                ),
            )
        )

        fully_correct = (
            status_correct
            and attribution_correct
        )

        if status_correct:
            correct_status_count += 1

        if attribution_correct:
            correct_attribution_count += 1

        if fully_correct:
            fully_correct_count += 1

        confusion[
            expected_status
        ][
            predicted_status
        ] += 1

        case_result = {
            "id": case["id"],
            "requirement": case[
                "requirement"
            ],
            "expected_status": (
                expected_status
            ),
            "predicted_status": (
                predicted_status
            ),
            "predicted_evidence_ids": (
                predicted_evidence_ids
            ),
            "acceptable_evidence_ids": (
                case[
                    "acceptable_evidence_ids"
                ]
            ),
            "status_correct": (
                status_correct
            ),
            "attribution_correct": (
                attribution_correct
            ),
            "fully_correct": (
                fully_correct
            ),
            "explanation": prediction[
                "explanation"
            ],
        }

        case_results.append(
            case_result
        )

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"{case['id']} | "
            f"{case['requirement']}"
        )

        print(
            f"Expected: "
            f"{expected_status}"
        )

        print(
            f"Predicted: "
            f"{predicted_status}"
        )

        print(
            "Evidence IDs: "
            f"{predicted_evidence_ids}"
        )

        print(
            f"Status correct: "
            f"{status_correct}"
        )

        print(
            f"Attribution correct: "
            f"{attribution_correct}"
        )

        print(
            f"Explanation: "
            f"{prediction['explanation']}"
        )

    total = len(cases)

    status_accuracy = (
        correct_status_count / total
        if total
        else 0.0
    )

    attribution_accuracy = (
        correct_attribution_count / total
        if total
        else 0.0
    )

    full_accuracy = (
        fully_correct_count / total
        if total
        else 0.0
    )

    output = {
        "benchmark": (
            "match_gap_eval.json"
        ),
        "case_count": total,
        "metrics": {
            "status_accuracy": (
                status_accuracy
            ),
            "attribution_accuracy": (
                attribution_accuracy
            ),
            "fully_correct_rate": (
                full_accuracy
            ),
        },
        "confusion_matrix": (
            confusion
        ),
        "cases": case_results,
    }

    output_path = (
        RESULTS_DIR
        / "match_gap_v1.json"
    )

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
        "MATCH / GAP EVALUATION"
    )

    print(
        "#" * 100
    )

    print(
        f"Cases: {total}"
    )

    print(
        f"Status accuracy: "
        f"{status_accuracy:.3f}"
    )

    print(
        f"Evidence attribution accuracy: "
        f"{attribution_accuracy:.3f}"
    )

    print(
        f"Fully correct rate: "
        f"{full_accuracy:.3f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        json.dumps(
            confusion,
            indent=2,
        )
    )

    print(
        f"\nSaved results to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()