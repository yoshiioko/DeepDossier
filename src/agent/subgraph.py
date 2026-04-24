"""ResearcherSubgraph: fetch_node + synthesize_node."""

from __future__ import annotations

import structlog
from langgraph.graph import END, START, StateGraph

from src.agent.agents import synthesize_agent
from src.agent.config import Settings
from src.agent.prompts import build_synthesize_prompt
from src.agent.schemas import SubResult
from src.agent.state import SubgraphState
from src.agent.tools import arxiv_search, web_search, wikipedia_search

logger = structlog.get_logger()


def fetch_node(state: SubgraphState, settings: Settings) -> dict:
    """Fetch raw content for a sub-query using the appropriate tool."""
    sub_query = state["sub_query"]
    log = logger.bind(node_name="fetch_node", topic=sub_query.topic)
    log.info("fetch_node.start")

    hint = sub_query.tool_hint
    if hint == "web":
        raw = web_search(sub_query.topic, settings=settings)
    elif hint == "wiki":
        raw = wikipedia_search(sub_query.topic)
    else:  # "paper"
        raw = arxiv_search(sub_query.topic)

    log.info("fetch_node.done", content_length=len(raw))
    return {"raw_content": raw}


def synthesize_node(state: SubgraphState, settings: Settings) -> dict:
    """Synthesize raw content into a structured SubResult using synthesize_agent"""
    sub_query = state["sub_query"]
    raw_content = state["raw_content"]
    log = logger.bind(node_name="synthesize_node", topic=sub_query.topic)
    log.info("synthesize_node.start")

    prompt = build_synthesize_prompt(sub_query, raw_content)
    result = synthesize_agent.run_sync(prompt, deps=settings)
    sub_result: SubResult = result.output

    log.info("synthesize_node.done", confidence=sub_result.confidence)
    # Return as a list so the operator.add reducer in SupervisorState appends it correctly
    return {"sub_results": [sub_result]}


def build_researcher_subgraph(settings: Settings):
    """Build and compile the researcher subgraph."""
    sg = StateGraph(SubgraphState) # type: ignore[arg-type]

    sg.add_node("fetch", lambda state: fetch_node(state, settings))
    sg.add_node("synthesize", lambda state: synthesize_node(state, settings))

    sg.add_edge(START, "fetch")
    sg.add_edge("fetch", "synthesize")
    sg.add_edge("synthesize", END)

    return sg.compile()
