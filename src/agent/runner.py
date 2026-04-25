"""CLI runner functions for DeepDossier."""

from __future__ import annotations

import uuid

import structlog

from src.agent.config import Settings
from src.agent.memory import BaseMemory
from src.agent.schemas import DossierOutput

logger = structlog.get_logger()


def run_once(
    query: str,
    settings: Settings,
    memory: BaseMemory,
    graph,
    thread_id: str | None = None,
) -> dict:
    """Run the full graph once and return the final state."""
    thread_id = thread_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_query": query,
        "sub_queries": [],
        "sub_results": [],
        "aggregated_results": [],
        "human_approved": False,
        "dossier_output": None,
        "run_id": run_id,
        "iteration_count": 0,
        "messages": [],
    }

    logger.info("run_once.start", query=query, run_id=run_id, thread_id=thread_id)
    final_state = graph.invoke(initial_state, config=config)
    logger.info("run_once.done", run_id=run_id)

    return final_state


def run_cli(settings: Settings, memory: BaseMemory, graph) -> None:
    """Interactive CLI entrypoint."""

    print("\n🔍 DeepDossier — Multi-Agent Research Pipeline\n")

    query = input("Enter research query: ").strip()
    if not query:
        print("No query entered. Exiting.")
        return

    print("\n⏳ Running research pipeline...\n")

    try:
        final_state = run_once(query=query, settings=settings, memory=memory, graph=graph)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise

    dossier_dict = final_state.get("dossier_output")
    if not dossier_dict:
        print("\n❌ No dossier output produced.")
        return

    dossier = DossierOutput(**dossier_dict)

    print("\n" + "=" * 60)
    print(dossier.final_markdown)
    print("=" * 60)
    print(f"\n✅ Run complete. run_id={dossier.run_id}")
