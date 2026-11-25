import argparse
import json
import sqlite3
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)

from src.graph.application_graph import (
    build_application_graph,
    build_revision_resume_graph,
)


# ============================================================
# Paths
# ============================================================

FIXTURE_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_tailor_fixture.json"
)

DRAFT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_draft.json"
)

VALIDATION_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_cv_validation.json"
)

RESUME_OUTPUT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_langgraph_result.json"
)

FULL_OUTPUT_PATH = Path(
    "evaluation/fixtures/"
    "ai_engineer_langgraph_full_result.json"
)

CHECKPOINT_DIR = Path(
    "evaluation/checkpoints"
)

CHECKPOINT_DB_PATH = (
    CHECKPOINT_DIR
    / "application_graph.sqlite"
)

LATEST_THREAD_PATH = (
    CHECKPOINT_DIR
    / "latest_full_thread.txt"
)


# ============================================================
# Test JD
# ============================================================

DEFAULT_JD_TEXT = """
Working Student AI Engineer

Responsibilities:
- Develop and evaluate LLM-based applications.
- Build retrieval-augmented generation workflows.
- Analyze model failures and improve system reliability.

Requirements:
- Strong Python skills.
- Hands-on experience with machine learning projects.
- Experience with NLP.
- Experience with retrieval-augmented generation (RAG).

Nice to have:
- Docker.
- LangGraph.
"""


# ============================================================
# Basic helpers
# ============================================================

def load_json(
    path: Path,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def create_checkpointer():
    """
    Create the persistent SQLite checkpointer.

    The SQLite file survives Python process termination,
    API errors, machine restarts, etc.
    """

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        CHECKPOINT_DB_PATH,
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(
        connection
    )

    checkpointer.setup()

    return (
        connection,
        checkpointer,
    )


def build_config(
    thread_id: str,
):

    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def generate_thread_id():

    return (
        "job-"
        + uuid.uuid4().hex[:12]
    )


def save_latest_thread_id(
    thread_id: str,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LATEST_THREAD_PATH.write_text(
        thread_id,
        encoding="utf-8",
    )


def load_latest_thread_id():

    if not LATEST_THREAD_PATH.exists():

        raise FileNotFoundError(
            "No saved LangGraph thread found.\n"
            "Start one first with:\n"
            "python -m "
            "src.evaluation.test_application_graph "
            "--full"
        )

    thread_id = (
        LATEST_THREAD_PATH
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    if not thread_id:

        raise ValueError(
            "Saved thread ID file is empty."
        )

    return thread_id


# ============================================================
# Checkpoint diagnostics
# ============================================================

def print_checkpoint_state(
    graph,
    config,
):
    """
    Print where LangGraph can resume.
    """

    snapshot = graph.get_state(
        config
    )

    print(
        "\n"
        + "-" * 100
    )

    print(
        "LANGGRAPH CHECKPOINT"
    )

    if not snapshot.values:

        print(
            "No persisted state found."
        )

        print(
            "-" * 100
        )

        return snapshot

    print(
        "Checkpoint exists: YES"
    )

    next_nodes = tuple(
        snapshot.next
    )

    if next_nodes:

        print(
            "Next node(s): "
            + ", ".join(
                next_nodes
            )
        )

    else:

        print(
            "Next node(s): none "
            "(graph already complete)"
        )

    revision_round = (
        snapshot.values.get(
            "revision_round",
            0,
        )
    )

    print(
        "Revision round: "
        f"{revision_round}"
    )

    available_keys = sorted(
        snapshot.values.keys()
    )

    print(
        "Persisted state keys:"
    )

    for key in available_keys:

        print(
            f"  - {key}"
        )

    print(
        "-" * 100
    )

    return snapshot


# ============================================================
# Full persistent run
# ============================================================

def run_full_test(
    requested_thread_id=None,
):
    """
    Start a NEW persistent raw-JD run.

    Every successfully completed LangGraph node is saved
    into SQLite.

    If a later node fails, use --resume with the same
    thread ID instead of starting from the beginning.
    """

    print(
        "[MODE] Full persistent "
        "raw-JD LangGraph run"
    )

    thread_id = (
        requested_thread_id
        or generate_thread_id()
    )

    connection, checkpointer = (
        create_checkpointer()
    )

    graph = build_application_graph(
        checkpointer=checkpointer
    )

    config = build_config(
        thread_id
    )

    try:

        # ----------------------------------------------------
        # Prevent accidentally starting a fresh run on top of
        # an existing thread.
        # ----------------------------------------------------

        existing = graph.get_state(
            config
        )

        if existing.values:

            raise ValueError(
                "\nThread already exists: "
                f"{thread_id}\n\n"
                "Use --resume instead, or start "
                "a new run without --thread-id."
            )

        save_latest_thread_id(
            thread_id
        )

        print(
            f"[CHECKPOINT] Thread ID: "
            f"{thread_id}"
        )

        print(
            f"[CHECKPOINT] SQLite DB: "
            f"{CHECKPOINT_DB_PATH}"
        )

        initial_state = {
            "jd_text": (
                DEFAULT_JD_TEXT
            ),

            "revision_round": 0,

            "max_revision_rounds": 2,

            "warnings": [],
        }

        result = graph.invoke(
            initial_state,
            config=config,
        )

        return (
            result,
            FULL_OUTPUT_PATH,
            thread_id,
        )

    except Exception:

        print(
            "\n"
            + "=" * 100
        )

        print(
            "GRAPH EXECUTION STOPPED"
        )

        print(
            "The current LangGraph state "
            "has been persisted."
        )

        print_checkpoint_state(
            graph,
            config,
        )

        print(
            "\nResume later with:"
        )

        print(
            "python -m "
            "src.evaluation.test_application_graph "
            "--resume"
        )

        print(
            "\nOr explicitly:"
        )

        print(
            "python -m "
            "src.evaluation.test_application_graph "
            "--resume "
            f"--thread-id {thread_id}"
        )

        print(
            "=" * 100
        )

        raise

    finally:

        connection.close()


# ============================================================
# Resume persistent run
# ============================================================

def run_persistent_resume(
    requested_thread_id=None,
):
    """
    Resume an existing LangGraph thread from its latest
    persisted checkpoint.

    Passing None as graph input tells LangGraph to continue
    execution from the saved state rather than injecting a
    new application input.
    """

    print(
        "[MODE] Resume persistent "
        "LangGraph run"
    )

    thread_id = (
        requested_thread_id
        or load_latest_thread_id()
    )

    connection, checkpointer = (
        create_checkpointer()
    )

    graph = build_application_graph(
        checkpointer=checkpointer
    )

    config = build_config(
        thread_id
    )

    try:

        print(
            f"[CHECKPOINT] Thread ID: "
            f"{thread_id}"
        )

        print(
            f"[CHECKPOINT] SQLite DB: "
            f"{CHECKPOINT_DB_PATH}"
        )

        snapshot = (
            print_checkpoint_state(
                graph,
                config,
            )
        )

        if not snapshot.values:

            raise ValueError(
                "No checkpoint exists for "
                f"thread: {thread_id}"
            )

        # ----------------------------------------------------
        # If the graph already finished successfully,
        # there is nothing to resume.
        # ----------------------------------------------------

        if not snapshot.next:

            print(
                "\n[CHECKPOINT] Graph is "
                "already complete."
            )

            result = dict(
                snapshot.values
            )

            return (
                result,
                FULL_OUTPUT_PATH,
                thread_id,
            )

        print(
            "\n[CHECKPOINT] Resuming "
            "graph execution..."
        )

        result = graph.invoke(
            None,
            config=config,
        )

        return (
            result,
            FULL_OUTPUT_PATH,
            thread_id,
        )

    except Exception:

        print(
            "\n"
            + "=" * 100
        )

        print(
            "RESUMED RUN STOPPED AGAIN"
        )

        print(
            "The newest checkpoint "
            "has been preserved."
        )

        print_checkpoint_state(
            graph,
            config,
        )

        print(
            "\nYou can resume the same "
            "thread again with:"
        )

        print(
            "python -m "
            "src.evaluation.test_application_graph "
            "--resume"
        )

        print(
            "=" * 100
        )

        raise

    finally:

        connection.close()


# ============================================================
# Existing frozen regression test
# ============================================================

def run_regression_test():
    """
    Existing fast regression test.

    This intentionally does NOT use persistent checkpoints.
    """

    print(
        "[MODE] Frozen regression "
        "draft + validation"
    )

    fixture = load_json(
        FIXTURE_PATH
    )

    cv_draft = load_json(
        DRAFT_PATH
    )

    validation = load_json(
        VALIDATION_PATH
    )

    graph = (
        build_revision_resume_graph()
    )

    initial_state = {
        "expanded_sources": (
            fixture[
                "expanded_sources"
            ]
        ),

        "cv_draft": (
            cv_draft
        ),

        "cv_validation": (
            validation
        ),

        "revision_round": 0,

        "max_revision_rounds": 2,

        "warnings": [],
    }

    result = graph.invoke(
        initial_state
    )

    return (
        result,
        RESUME_OUTPUT_PATH,
        None,
    )


# ============================================================
# Result output
# ============================================================

def print_result(
    result,
    thread_id=None,
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "LANGGRAPH RESULT"
    )

    if thread_id:

        print(
            f"Thread ID: "
            f"{thread_id}"
        )

    print(
        f"Final status: "
        f"{result.get('final_status')}"
    )

    print(
        f"Revision rounds: "
        f"{result.get('revision_round')}"
    )

    # --------------------------------------------------------
    # Upstream
    # --------------------------------------------------------

    jd_analysis = result.get(
        "jd_analysis",
        {},
    )

    if jd_analysis:

        print(
            "\nJob title:"
        )

        print(
            "  "
            f"{jd_analysis.get('job_title')}"
        )

    queries = result.get(
        "queries",
        [],
    )

    if queries:

        print(
            "\nRetrieval queries:"
        )

        for query in queries:

            print(
                f"  - {query}"
            )

    evidence_package = result.get(
        "evidence_package",
        {},
    )

    evidence = evidence_package.get(
        "evidence",
        [],
    )

    if evidence:

        print(
            "\nRetrieved evidence:"
        )

        print(
            f"  {len(evidence)} "
            "final evidence items"
        )

    match_gap = result.get(
        "match_gap",
        {},
    )

    match_summary = match_gap.get(
        "summary",
        {},
    )

    if match_summary:

        print(
            "\nMatch / Gap:"
        )

        print(
            "  SUPPORTED: "
            f"{match_summary.get('SUPPORTED', 0)}"
        )

        print(
            "  PARTIAL:   "
            f"{match_summary.get('PARTIAL', 0)}"
        )

        print(
            "  GAP:       "
            f"{match_summary.get('GAP', 0)}"
        )

    strategy = result.get(
        "application_strategy",
        {},
    )

    priority_sources = strategy.get(
        "priority_sources",
        [],
    )

    if priority_sources:

        print(
            "\nPriority sources:"
        )

        for source in priority_sources:

            print(
                "  - "
                f"{source.get('source_id')} "
                f"[{source.get('entity_type')}]"
            )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    validation = result.get(
        "cv_validation",
        {},
    )

    summary = validation.get(
        "summary",
        {},
    )

    print(
        "\nFinal CV validation:"
    )

    print(
        "  SUPPORTED:   "
        f"{summary.get('SUPPORTED', 0)}"
    )

    print(
        "  OVERSTATED:  "
        f"{summary.get('OVERSTATED', 0)}"
    )

    print(
        "  UNSUPPORTED: "
        f"{summary.get('UNSUPPORTED', 0)}"
    )

    if "grounded_rate" in summary:

        print(
            "  Grounded rate: "
            f"{summary['grounded_rate']:.1%}"
        )

    warnings = result.get(
        "warnings",
        [],
    )

    if warnings:

        print(
            "\nWarnings:"
        )

        for warning in warnings:

            print(
                f"  - {warning}"
            )

    print(
        "\n"
        + "=" * 100
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the "
            "job-application LangGraph."
        )
    )

    mode = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--full",
        action="store_true",
        help=(
            "Start a NEW persistent "
            "raw-JD run."
        ),
    )

    mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume the latest "
            "persistent full run."
        ),
    )

    mode.add_argument(
        "--regression",
        action="store_true",
        help=(
            "Run the old frozen "
            "CV regression test."
        ),
    )

    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help=(
            "Optional explicit "
            "LangGraph thread ID."
        ),
    )

    args = parser.parse_args()

    print(
        "[1/4] Preparing LangGraph..."
    )

    if args.full:

        (
            result,
            output_path,
            thread_id,
        ) = run_full_test(
            requested_thread_id=(
                args.thread_id
            )
        )

    elif args.resume:

        (
            result,
            output_path,
            thread_id,
        ) = run_persistent_resume(
            requested_thread_id=(
                args.thread_id
            )
        )

    else:

        (
            result,
            output_path,
            thread_id,
        ) = run_regression_test()

    print(
        "[1/4] Done."
    )

    print(
        "[2/4] Graph execution complete."
    )

    print(
        "[3/4] Saving final result..."
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[3/4] Done."
    )

    print_result(
        result=result,
        thread_id=thread_id,
    )

    print(
        f"\nSaved to: "
        f"{output_path}"
    )

    if thread_id:

        print(
            f"Checkpoint DB: "
            f"{CHECKPOINT_DB_PATH}"
        )

    print(
        "[4/4] Done."
    )


if __name__ == "__main__":
    main()