"""Build and compile the DeepDossier supervisor graph."""

from __future__ import annotations

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agent.config import Settings
from src.agent.memory import BaseMemory
from src.agent.nodes import (
    aggregator_node,
    compiler_node,
    dispatcher_node,
    memory_writer_node,
    planner_node,
)
from src.agent.state import SupervisorState
from src.agent.subgraph import build_researcher_subgraph


def build_supervisor_graph(settings: Settings, memory: BaseMemory):
    """Build and compile the full supervisor graph with SqliteSaver checkpointing."""

    builder = StateGraph(SupervisorState)  # type: ignore[arg-type]

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("planner",       lambda s: planner_node(s, settings, memory))
    builder.add_node("dispatcher",    lambda s: dispatcher_node(s, settings))
    builder.add_node("researcher",    build_researcher_subgraph(settings))
    builder.add_node("aggregator",    lambda s: aggregator_node(s, settings))
    builder.add_node("memory_writer", lambda s: memory_writer_node(s, settings, memory))
    builder.add_node("compiler",      lambda s: compiler_node(s, settings))

    # ── Wire edges ────────────────────────────────────────────────────────────
    builder.add_edge(START,           "planner")
    builder.add_edge("planner",       "dispatcher")
    builder.add_conditional_edges("dispatcher", lambda s: "researcher", ["researcher"])
    builder.add_edge("researcher",    "aggregator")
    builder.add_edge("aggregator",    "memory_writer")
    builder.add_edge("memory_writer", "compiler")
    builder.add_edge("compiler",      END)

    # ── Checkpointer ──────────────────────────────────────────────────────────
    checkpointer = SqliteSaver.from_conn_string("./graph_state.db")

    return builder.compile(checkpointer=checkpointer)
