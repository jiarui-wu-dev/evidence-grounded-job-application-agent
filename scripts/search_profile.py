import argparse
from pathlib import Path

from qdrant_client import QdrantClient, models


MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "candidate_experience"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = PROJECT_ROOT / "qdrant_data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        type=str,
        help="Job description or skill requirements",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    client = QdrantClient(path=str(QDRANT_PATH))

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(
            text=args.query,
            model=MODEL_NAME,
        ),
        limit=args.top_k,
    ).points

    print("\nQUERY")
    print(args.query)

    for rank, result in enumerate(results, start=1):
        payload = result.payload

        print(f"\n{'=' * 60}")
        print(f"RANK {rank}")
        print(f"Project: {payload['title']}")
        print(f"Score:   {result.score:.4f}")

        print("\nSkills:")
        print(", ".join(payload["skills"]))

        print("\nEvidence:")
        for item in payload["evidence"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()