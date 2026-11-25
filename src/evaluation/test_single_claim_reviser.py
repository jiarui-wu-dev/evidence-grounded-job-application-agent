import argparse
import json
from pathlib import Path

from src.cv.cv_reviser import (
    build_revision_batches,
    invoke_revision_batch,
)

from src.llm.client import (
    get_llm,
)


INPUT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_langgraph_full_result.json"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--claim",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    print(
        "[1/3] Loading saved full-run state..."
    )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        state = json.load(
            file
        )

    cv_draft = state[
        "final_cv"
    ]

    validation = state[
        "cv_validation"
    ]

    expanded_sources = state[
        "expanded_sources"
    ]

    batches = build_revision_batches(
        cv_draft=cv_draft,
        validation_result=(
            validation
        ),
        expanded_sources=(
            expanded_sources
        ),
    )

    target_claim = None

    for batch in batches:

        for claim in batch[
            "claims"
        ]:

            if (
                claim["claim_id"]
                == args.claim
            ):
                target_claim = claim
                break

        if target_claim is not None:
            break

    if target_claim is None:

        raise ValueError(
            "Problematic claim not found: "
            f"{args.claim}"
        )

    print(
        "[1/3] Done."
    )

    print(
        "\nORIGINAL CLAIM:"
    )

    print(
        target_claim[
            "original_text"
        ]
    )

    print(
        "\nVALIDATOR FEEDBACK:"
    )

    print(
        target_claim[
            "validator_reason"
        ]
    )

    print(
        "\nUNSUPPORTED PHRASES:"
    )

    for phrase in target_claim.get(
        "unsupported_phrases",
        [],
    ):

        print(
            f"  - {phrase}"
        )

    print(
        "\n[2/3] Revising ONLY this claim..."
    )

    llm = get_llm(
        temperature=0.0
    )

    result = invoke_revision_batch(
        llm=llm,
        claims=[
            target_claim
        ],
        label=(
            f"single claim "
            f"{args.claim}"
        ),
        max_attempts=1,
    )

    print(
        "[2/3] Done."
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "REVISION RESULT"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "=" * 100
    )

    print(
        "[3/3] Done."
    )


if __name__ == "__main__":
    main()