"""LangGraph state types for supervisor and researcher subgraph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import MessagesState

from src.agent.schemas import SubQuery, SubResult


class SupervisorState(MessagesState):
    """Shared state for the supervisor graph.

    All fields except `messages` (inherited) are plain Python types so that
    SqliteSaver can serialize the state to JSON without any custom encoder.
    """

    user_query: str
    sub_queries: list[SubQuery]                            # populated by planner_node
    sub_results: Annotated[list[SubResult], operator.add]  # append reducer — required for Send fan-out
    aggregated_results: list[SubResult]                    # populated by aggregator_node — plain replace, no reducer
    memory_context: list[dict]  # written by planner_node
    human_approved: bool             # HITL gate; default False
    dossier_output: dict | None      # DossierOutput.model_dump() — never the Pydantic obj
    run_id: str                      # UUID string bound to every log event
    iteration_count: int             # runaway-loop guard


class SubgraphState(TypedDict):
    """State for one ResearcherSubgraph instance.

    Each parallel researcher receives a fresh copy of this state.
    """

    sub_query: SubQuery
    raw_content: str                                           # populated by fetch_node
    sub_results: Annotated[list[SubResult], operator.add]      # populated by synthesize_node; matches supervisor reducer
