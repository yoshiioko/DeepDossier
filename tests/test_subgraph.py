"""Unit tests for src/agent/subgraph.py — zero API calls."""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from src.agent.agents import synthesize_agent
from src.agent.config import Settings
from src.agent.schemas import SubQuery, SubResult
from src.agent.state import SubgraphState
from src.agent.subgraph import build_researcher_subgraph, fetch_node, synthesize_node


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def settings() -> Settings:
    return Settings(google_api_key="x", tavily_api_key="x")


@pytest.fixture
def web_query() -> SubQuery:
    return SubQuery(topic="AI agents", tool_hint="web", rationale="latest developments")


@pytest.fixture
def wiki_query() -> SubQuery:
    return SubQuery(topic="neural networks", tool_hint="wiki", rationale="background")


@pytest.fixture
def paper_query() -> SubQuery:
    return SubQuery(topic="transformer architecture", tool_hint="paper", rationale="academic")


# ── fetch_node tool routing ───────────────────────────────────────────────────

def test_fetch_node_routes_web(monkeypatch, settings, web_query) -> None:
    monkeypatch.setattr("src.agent.subgraph.web_search", lambda q, settings: "web result")
    state: SubgraphState = {"sub_query": web_query, "raw_content": "", "sub_results": []}
    result = fetch_node(state, settings)
    assert result["raw_content"] == "web result"


def test_fetch_node_routes_wiki(monkeypatch, settings, wiki_query) -> None:
    monkeypatch.setattr("src.agent.subgraph.wikipedia_search", lambda q: "wiki result")
    state: SubgraphState = {"sub_query": wiki_query, "raw_content": "", "sub_results": []}
    result = fetch_node(state, settings)
    assert result["raw_content"] == "wiki result"


def test_fetch_node_routes_paper(monkeypatch, settings, paper_query) -> None:
    monkeypatch.setattr("src.agent.subgraph.arxiv_search", lambda q: "arxiv result")
    state: SubgraphState = {"sub_query": paper_query, "raw_content": "", "sub_results": []}
    result = fetch_node(state, settings)
    assert result["raw_content"] == "arxiv result"


def test_fetch_node_returns_dict_with_raw_content(monkeypatch, settings, web_query) -> None:
    monkeypatch.setattr("src.agent.subgraph.web_search", lambda q, settings: "anything")
    state: SubgraphState = {"sub_query": web_query, "raw_content": "", "sub_results": []}
    result = fetch_node(state, settings)
    assert "raw_content" in result
    assert isinstance(result["raw_content"], str)


# ── synthesize_node ───────────────────────────────────────────────────────────

async def test_synthesize_node_returns_sub_result(settings, web_query) -> None:
    state: SubgraphState = {
        "sub_query": web_query,
        "raw_content": "AI agents are software systems that perceive and act.",
        "sub_results": [],
    }
    with synthesize_agent.override(model=TestModel()):
        result = await synthesize_node(state, settings)
    assert "sub_results" in result
    assert isinstance(result["sub_results"][0], SubResult)


# ── Full subgraph ─────────────────────────────────────────────────────────────

async def test_full_subgraph_produces_sub_result(monkeypatch, settings, wiki_query) -> None:
    monkeypatch.setattr("src.agent.subgraph.wikipedia_search", lambda q: "Neural networks are...")
    with synthesize_agent.override(model=TestModel()):
        subgraph = build_researcher_subgraph(settings)
        final = await subgraph.ainvoke({
            "sub_query": wiki_query,
            "raw_content": "",
            "sub_results": [],
        })
    assert len(final["sub_results"]) > 0
    assert isinstance(final["sub_results"][0], SubResult)


async def test_full_subgraph_state_has_all_keys(monkeypatch, settings, web_query) -> None:
    monkeypatch.setattr("src.agent.subgraph.web_search", lambda q, settings: "some content")
    with synthesize_agent.override(model=TestModel()):
        subgraph = build_researcher_subgraph(settings)
        final = await subgraph.ainvoke({
            "sub_query": web_query,
            "raw_content": "",
            "sub_results": [],
        })
    assert "sub_query" in final
    assert "raw_content" in final
    assert "sub_results" in final
