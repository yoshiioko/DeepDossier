"""Supervisor node functions for the DeepDossier graph."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.generativeai|instructor",
)

import instructor
import structlog
import google.generativeai as genai
from langgraph.types import Send

from src.agent.agents import compiler_agent
from src.agent.config import Settings
from src.agent.memory import BaseMemory
from src.agent.prompts import build_compiler_prompt, build_planner_prompt
from src.agent.schemas import DossierOutput, MemoryChunk, PlannerOutput, SubResult
from src.agent.state import SupervisorState

logger = structlog.get_logger()


async def planner_node(state: SupervisorState, settings: Settings, memory: BaseMemory) -> dict:
    """Decompose the user query into sub-queries using Instructor + Gemini."""
    log = logger.bind(node_name="planner_node", run_id=state.get("run_id", ""))
    log.info("planner_node.start")

    # 1. Query ChromaDB for already-known topics
    known_chunks = await memory.retrieve(state["user_query"], n_results=5, min_score=settings.memory_hit_confidence_threshold)
    known_topics = [c.metadata.get("topic", "") for c in known_chunks if c.metadata.get("topic")]

    if known_topics:
        log.info(
            "planner_node.memory_hit",
            num_chunks=len(known_chunks),
            known_topics=known_topics,
        )
    else:
        log.info("planner_node.memory_miss", num_chunks=len(known_chunks))

    # 2. Build prompt
    prompt = build_planner_prompt(
        query=state["user_query"],
        known_topics=known_topics,
        max_sub_queries=settings.max_parallel_researchers,
    )

    # 3. Call Gemini via Instructor for structured output with auto-retry on ValidationError
    genai.configure(api_key=settings.google_api_key)
    client = instructor.from_gemini(
        client=genai.GenerativeModel(model_name=settings.planner_model_name),
        mode=instructor.Mode.GEMINI_JSON,
    )
    result: PlannerOutput = client.chat.completions.create(
        response_model=PlannerOutput,
        messages=[{"role": "user", "content": prompt}],
        max_retries=3,
    )

    # 4. Cap sub-queries
    sub_queries = result.sub_queries[: settings.max_parallel_researchers]
    log.info("planner_node.done", num_sub_queries=len(sub_queries))

    return {
        "sub_queries": sub_queries,
        "memory_context": [c.model_dump() for c in known_chunks],
    }


def dispatcher_node(state: SupervisorState, settings: Settings) -> list[Send]:
    """Fan out one ResearcherSubgraph per sub-query using the Send API."""
    log = logger.bind(node_name="dispatcher_node", run_id=state.get("run_id", ""))
    log.info("dispatcher_node.start", num_queries=len(state["sub_queries"]))

    return [
        Send(
            "researcher",
            {
                "sub_query": sq,
                "raw_content": "",
                "sub_results": [],
            },
        )
        for sq in state["sub_queries"]
    ]


def aggregator_node(
    state: SupervisorState,
    settings: Settings,
) -> dict:
    """Deduplicate sub_results by topic, keeping the highest confidence.

    Writes to `aggregated_results` (no reducer) rather than back into
    `sub_results` (operator.add reducer) to avoid doubling the list.
    """
    log = logger.bind(node_name="aggregator_node", run_id=state.get("run_id", ""))
    log.info("aggregator_node.start", num_results=len(state.get("sub_results", [])))

    seen: dict[str, SubResult] = {}
    for result in state.get("sub_results", []):
        topic = result.topic.lower().strip()
        if topic not in seen or result.confidence > seen[topic].confidence:
            seen[topic] = result

    deduplicated = list(seen.values())
    log.info("aggregator_node.done", num_deduplicated=len(deduplicated))

    return {"aggregated_results": deduplicated}


async def _write_memory(
    sub_results: list[SubResult],
    run_id: str,
    memory: BaseMemory,
) -> None:
    """Async helper — writes all SubResults to ChromaDB as MemoryChunks."""
    chunks = [
        MemoryChunk(
            content=r.summary,
            metadata={
                "topic": r.topic,
                "source_url": r.sources[0] if r.sources else "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
            },
        )
        for r in sub_results
    ]
    await memory.write_chunks(chunks)


async def memory_writer_node(
    state: SupervisorState,
    settings: Settings,
    memory: BaseMemory,
) -> dict:
    """Persist approved, high-confidence research findings to ChromaDB."""
    log = logger.bind(node_name="memory_writer_node", run_id=state.get("run_id", ""))
    log.info("memory_writer_node.start")

    all_results = state.get("aggregated_results", [])
    threshold = settings.memory_write_confidence_threshold
    to_write = [r for r in all_results if r.confidence >= threshold]
    skipped = len(all_results) - len(to_write)

    log.info(
        "memory_writer_node.filtered",
        total=len(all_results),
        writing=len(to_write),
        skipped=skipped,
        threshold=threshold,
    )

    if to_write:
        try:
            await _write_memory(
                sub_results=to_write,
                run_id=state.get("run_id", ""),
                memory=memory,
            )
            log.info("memory_writer_node.done", num_chunks=len(to_write))
        except Exception as e:
            log.warning("memory_writer_node.failed", error=str(e))
    else:
        log.info("memory_writer_node.nothing_to_write")

    return {}


async def compiler_node(
    state: SupervisorState,
    settings: Settings,
) -> dict:
    """Compile all sub_results into a DossierOutput using compiler_agent."""
    log = logger.bind(node_name="compiler_node", run_id=state.get("run_id", ""))
    log.info("compiler_node.start")

    memory_context: list[dict] = state.get("memory_context", [])  # ← pull from state

    prompt = build_compiler_prompt(
        query=state["user_query"],
        sub_results=state.get("aggregated_results", []),
        memory_context=memory_context,                              # ← pass to prompt
    )

    result = await compiler_agent.run(prompt, deps=settings)
    dossier: DossierOutput = result.output

    dossier = dossier.model_copy(update={
        "run_id": state.get("run_id", ""),
        "memory_chunks_used": len(memory_context),                 # ← track how many cached chunks were used
    })

    log.info("compiler_node.done", title=dossier.title, memory_chunks_used=len(memory_context))

    return {"dossier_output": dossier.model_dump()}
