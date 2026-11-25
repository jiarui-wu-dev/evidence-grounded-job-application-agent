from qdrant_client import QdrantClient, models


MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "candidate_experience"


client = QdrantClient(":memory:")

documents = [
    "Built a Spring Boot document approval system with REST APIs, PostgreSQL, "
    "Spring Security, Docker, and role-based access control.",

    "Developed an LLM-assisted Verilog weakness injection workflow with "
    "automated validation using Python, Yosys, and Slang.",

    "Built a maritime speech recognition evaluation pipeline using Whisper, "
    "PyTorch, Hugging Face, LoRA, and WER-based error analysis.",

    "Applied Bayesian optimization and Gaussian Processes to improve "
    "particle accelerator simulation efficiency by approximately 40%.",
]

metadata = [
    {"project": "Document Review & Approval System"},
    {"project": "LLM-Assisted Hardware Weakness Injection"},
    {"project": "Maritime ASR"},
    {"project": "ARES Bayesian Optimization"},
]

# 1. Create vector collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=client.get_embedding_size(MODEL_NAME),
        distance=models.Distance.COSINE,
    ),
)

# 2. Store documents + embeddings
client.upload_collection(
    collection_name=COLLECTION_NAME,
    vectors=[
        models.Document(text=doc, model=MODEL_NAME)
        for doc in documents
    ],
    payload=[
        {
            "document": doc,
            **meta,
        }
        for doc, meta in zip(documents, metadata)
    ],
    ids=list(range(len(documents))),
)

# 3. Query
query = """
We are looking for a working student with experience in
Java, Spring Boot, REST APIs, Docker and backend development.
"""

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=models.Document(
        text=query,
        model=MODEL_NAME,
    ),
    limit=3,
).points

# 4. Print ranked results
for rank, result in enumerate(results, start=1):
    print(f"\n--- Rank {rank} ---")
    print(f"Project: {result.payload['project']}")
    print(f"Score:   {result.score:.4f}")
    print(f"Text:    {result.payload['document']}")