from pathlib import Path

from qdrant_client import (
    QdrantClient,
    models,
)

from src.retrieval.profile_documents import (
    load_profile_documents,
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


def build_index() -> None:
    """
    Rebuild the Qdrant candidate-profile index
    from the unified candidate evidence corpus.

    The corpus may contain:
    - projects
    - research experience
    - professional experience

    Metadata such as entity_type and
    evidence_status is preserved in Qdrant.
    """

    documents = (
        load_profile_documents()
    )

    print(
        f"Loaded {len(documents)} "
        f"candidate evidence units."
    )

    if not documents:
        raise RuntimeError(
            "No evidence units were generated. "
            "Check candidate profile data and "
            "profile_documents.py."
        )

    client = QdrantClient(
        path=str(
            QDRANT_PATH
        )
    )

    try:
        # -------------------------------------------------
        # Rebuild collection from scratch.
        # -------------------------------------------------

        if client.collection_exists(
            COLLECTION_NAME
        ):
            print(
                "Deleting existing "
                f"'{COLLECTION_NAME}' collection..."
            )

            client.delete_collection(
                collection_name=(
                    COLLECTION_NAME
                )
            )

        embedding_size = (
            client.get_embedding_size(
                MODEL_NAME
            )
        )

        client.create_collection(
            collection_name=(
                COLLECTION_NAME
            ),
            vectors_config=(
                models.VectorParams(
                    size=embedding_size,
                    distance=(
                        models.Distance.COSINE
                    ),
                )
            ),
        )

        # -------------------------------------------------
        # Build vectors and payloads.
        # -------------------------------------------------

        texts = [
            document["text"]
            for document
            in documents
        ]

        payloads = [
            {
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
            }
            for document
            in documents
        ]

        ids = list(
            range(
                1,
                len(documents) + 1,
            )
        )

        # -------------------------------------------------
        # Embed and upload.
        # -------------------------------------------------

        print(
            "Embedding and uploading "
            "candidate evidence..."
        )

        client.upload_collection(
            collection_name=(
                COLLECTION_NAME
            ),
            vectors=[
                models.Document(
                    text=text,
                    model=MODEL_NAME,
                )
                for text in texts
            ],
            payload=payloads,
            ids=ids,
        )

        count = client.count(
            collection_name=(
                COLLECTION_NAME
            ),
            exact=True,
        ).count

        print(
            f"Indexed {count} evidence units "
            f"into Qdrant collection "
            f"'{COLLECTION_NAME}'."
        )

        # -------------------------------------------------
        # Show a few examples for sanity checking.
        # -------------------------------------------------

        print(
            "\nSample evidence units:"
        )

        for document in documents[:5]:
            print(
                "-" * 80
            )

            print(
                f"[{document['entity_type']}] "
                f"[{document['evidence_status']}] "
                f"[{document['type']}]"
            )

            print(
                document[
                    "project_title"
                ]
            )

            print(
                document["text"]
            )

        # Also explicitly show the experience records
        # so we can verify their metadata survived.
        print(
            "\nExperience evidence samples:"
        )

        experience_documents = [
            document
            for document
            in documents
            if document.get(
                "entity_type"
            )
            in {
                "research_experience",
                "professional_experience",
            }
        ]

        for document in (
            experience_documents[:4]
        ):
            print(
                "-" * 80
            )

            print(
                f"ID: "
                f"{document['evidence_id']}"
            )

            print(
                f"entity_type: "
                f"{document['entity_type']}"
            )

            print(
                f"evidence_status: "
                f"{document['evidence_status']}"
            )

            print(
                document["text"]
            )

    finally:
        client.close()


if __name__ == "__main__":
    build_index()