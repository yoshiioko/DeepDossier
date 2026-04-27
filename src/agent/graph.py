"""Build and compile the DeepDossier supervisor graph."""

from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agent.config import Settings
from src.agent.memory import BaseMemory
from src.agent.nodes import (
    aggregator_node,
    compiler_node,
    dispatcher_node,
    memory_writer_node,
    planner_node,
)
from src.agent.schemas import SubResult
from src.agent.state import SupervisorState
from src.agent.subgraph import build_researcher_subgraph


def interrupt_gate(state: SupervisorState) -> dict:
    """Pause graph execution for human review."""
    sub_results: list[SubResult] = state.get("aggregated_results", [])
    preview_lines = [
        f"**{r.topic}** (confidence: {r.confidence:.2f}): {r.summary[:120]}..."
        for r in sub_results
    ]
    preview_md = "\n".join(f"- {line}" for line in preview_lines) or "No results to review."
    interrupt({
        "preview_markdown": preview_md,
        "sub_results": [r.model_dump() for r in sub_results],
        "run_id": state.get("run_id", ""),
    })
    return {}


def route_after_interrupt(state: SupervisorState) -> str:
    """Route to memory_writer if approved, else END."""
    return "memory_writer" if state.get("human_approved") else END


def _build_graph(settings: Settings, memory: BaseMemory, checkpointer):
    """Wire nodes and edges; attach checkpointer.

    Graph topology:
        START -> planner -> researcher (×N, parallel Send) -> aggregator
             -> interrupt_gate [HITL pause] -> memory_writer -> compiler -> END
             (or -> END on rejection)

    Named adapter functions bridge LangGraph's single-argument call convention
    (state only) with our node convention (state + settings [+ memory]).
    """
    builder = StateGraph(SupervisorState)  # type: ignore[arg-type]

    async def planner_adapter(state: SupervisorState) -> dict:
        return await planner_node(state, settings, memory)

    def dispatcher_adapter(state: SupervisorState):
        return dispatcher_node(state, settings)

    def aggregator_adapter(state: SupervisorState) -> dict:
        return aggregator_node(state, settings)

    async def memory_writer_adapter(state: SupervisorState) -> dict:
        return await memory_writer_node(state, settings, memory)

    async def compiler_adapter(state: SupervisorState) -> dict:
        return await compiler_node(state, settings)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("planner",       planner_adapter)       # type: ignore[arg-type]
    builder.add_node("researcher",    build_researcher_subgraph(settings))
    builder.add_node("aggregator",    aggregator_adapter)  # type: ignore[arg-type]
    builder.add_node("interrupt_gate", interrupt_gate)         # type: ignore[arg-type]
    builder.add_node("memory_writer", memory_writer_adapter)  # type: ignore[arg-type]
    builder.add_node("compiler",      compiler_adapter)       # type: ignore[arg-type]

    # ── Wire edges ────────────────────────────────────────────────────────────
    builder.add_edge(START,           "planner")
    builder.add_conditional_edges("planner", dispatcher_adapter, ["researcher"])
    builder.add_edge("researcher",    "aggregator")
    builder.add_edge("aggregator",     "interrupt_gate")
    builder.add_conditional_edges(                             # ← new conditional edge
        "interrupt_gate",
        route_after_interrupt,
        {"memory_writer": "memory_writer", END: END},
    )
    builder.add_edge("memory_writer", "compiler")
    builder.add_edge("compiler",      END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["interrupt_gate"],                   # ← new: pause before this node
    )


@asynccontextmanager
async def build_supervisor_graph(settings: Settings, memory: BaseMemory):
    """Async context manager — yields a compiled graph backed by AsyncSqliteSaver."""
    async with AsyncSqliteSaver.from_conn_string("./graph_state.db") as checkpointer:
        yield _build_graph(settings, memory, checkpointer)
