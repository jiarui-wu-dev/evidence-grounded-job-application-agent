# Architecture

```mermaid
flowchart TD
    UI["React workbench"] --> REST["FastAPI"]
    Host["Local MCP host"] --> MCP["MCP stdio server"]
    REST --> Service["Application service"]
    MCP --> Service
    Service --> Graph["LangGraph"]
    Service --> Store["SQLite runs, checkpoints, and decisions"]
    Graph --> Store
    Graph --> Index["Local Qdrant and candidate evidence"]
```

## Application workflow

```mermaid
flowchart TD
    JD["Analyze JD and build queries"] --> Search["Retrieve and rerank evidence"]
    Search --> Match["Map match and gap"]
    Match --> Strategy["Plan and expand CV sources"]
    Strategy --> Draft["Compose cited CV claims"]
    Draft --> Validate["Validate claims"]
    Validate -->|Supported| Grounded["Grounded final CV"]
    Validate -->|Revision budget available| Revise["Revise flagged claims"]
    Revise --> Recheck["Recheck changed claims and merge verdicts"]
    Recheck --> Validate
    Validate -->|Unresolved at limit| Review["Human review, export blocked"]
    Review -->|Review decisions completed| Target["Recheck affected claims"]
    Target -->|Supported| Grounded
    Target -->|Unresolved| Review
```

Queries combine lexical BM25 and dense Qdrant candidates. A CrossEncoder
reranks the pool, and confidence-aware aggregation combines results across
queries. Weighted reciprocal rank fusion is also available for retrieval
benchmark comparisons. Requirements
are displayed with their recorded status and cited IDs. The workbench can
trace each claim back to its evidence and show how validation changed after
revision.

After an automatic revision, the Validator receives changed claim IDs.
Previously evaluated rows remain in the result when their claims are
untouched. Human-review decisions use a run-scoped overlay and retain an
auditable original status. Rejection removes citations, and a claim that
loses its last citation stays unsupported. Both adapters use the same CV
access rule, and the MCP adapter exposes review queries only.
