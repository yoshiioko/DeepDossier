"""CLI runner functions for DeepDossier."""

from __future__ import annotations

import asyncio
import uuid

import structlog

from src.agent.config import Settings
from src.agent.memory import BaseMemory
from src.agent.schemas import DossierOutput

logger = structlog.get_logger()


async def run_once(
    query: str,
    settings: Settings,
    memory: BaseMemory,
    graph,
    thread_id: str | None = None,
) -> tuple[dict, str]:
    """Run the full graph once and return the final state."""
    thread_id = thread_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_query": query,
        "sub_queries": [],
        "sub_results": [],
        "aggregated_results": [],
        "memory_context": [],
        "human_approved": False,
        "dossier_output": None,
        "run_id": run_id,
        "iteration_count": 0,
        "messages": [],
    }

    logger.info("run_once.start", query=query, run_id=run_id, thread_id=thread_id)
    final_state = await graph.ainvoke(initial_state, config=config)
    logger.info("run_once.paused_or_done", run_id=run_id)

    return final_state, thread_id


async def run_with_approval(
    thread_id: str,
    approved: bool,
    graph,
) -> dict:
    """Resume a paused graph after human approval or rejection."""
    config = {"configurable": {"thread_id": thread_id}}
    await graph.aupdate_state(config, {"human_approved": approved})
    final_state = await graph.ainvoke(None, config=config)
    return final_state


async def run_cli_async(settings: Settings, memory: BaseMemory, graph) -> None:
    """Interactive CLI — handles the full HITL cycle."""
    print("\n🔍 DeepDossier — Multi-Agent Research Pipeline\n")

    query = input("Enter research query: ").strip()
    if not query:
        print("No query entered. Exiting.")
        return

    print("\n⏳ Running research pipeline...\n")

    try:
        state, thread_id = await run_once(
            query=query, settings=settings, memory=memory, graph=graph
        )
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise

    if state.get("dossier_output") is None:
        _print_hitl_preview(state)
        answer = input("\n✅ Approve and save to memory? [y/N]: ").strip().lower()
        approved = answer == "y"

        if approved:
            print("\n⏳ Resuming pipeline (approved)...\n")
        else:
            print("\n🚫 Rejected — research will NOT be saved to memory.\n")

        try:
            state = await run_with_approval(thread_id=thread_id, approved=approved, graph=graph)
        except Exception as e:
            print(f"\n❌ Resume failed: {e}")
            raise

        if not approved:
            print("Run terminated. No dossier produced.")
            return

    dossier_dict = state.get("dossier_output")
    if not dossier_dict:
        print("\n❌ No dossier output produced.")
        return

    dossier = DossierOutput(**dossier_dict)

    print("\n" + "=" * 60)
    print(dossier.final_markdown)
    print("=" * 60)

    if dossier.memory_chunks_used > 0:
        print(f"\n💾 {dossier.memory_chunks_used} cached chunk(s) from local memory used in this dossier.")

    print(f"\n✅ Run complete. run_id={dossier.run_id}")


def _print_hitl_preview(state: dict) -> None:
    """Print aggregated research findings for human review."""
    results = [
        r.model_dump() if hasattr(r, "model_dump") else r
        for r in state.get("aggregated_results", [])
    ]
    memory_chunks = state.get("memory_context", [])

    print("\n" + "=" * 60)
    print("📋 HUMAN REVIEW — Research Findings")
    print("=" * 60)

    if memory_chunks:
        print(f"\n💾 {len(memory_chunks)} cached chunk(s) from local memory will be used by the compiler.")

    if not results:
        print("No research results to review.")
        return

    for r in results:
        confidence = r.get("confidence", 0.0)
        flag = "⚠️  " if confidence < 0.5 else ""
        print(f"\n{flag}Topic: {r.get('topic', 'unknown')} (confidence: {confidence:.2f})")
        print(f"  Summary: {r.get('summary', '')[:200]}...")

    print("\n" + "=" * 60)


def run_cli(settings: Settings, memory: BaseMemory, graph_cm) -> None:
    """Sync shim so main.py stays a plain script."""
    async def _run():
        async with graph_cm as graph:
            await run_cli_async(settings, memory, graph)
    asyncio.run(_run())
