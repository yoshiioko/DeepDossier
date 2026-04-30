"""Unit tests for src/agent/schemas.py — zero API calls."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent.schemas import (
    DossierOutput,
    DossierSection,
    PlannerOutput,
    SubQuery,
    SubResult,
)


# ── SubQuery ──────────────────────────────────────────────────────────────────

def test_sub_query_valid() -> None:
    sq = SubQuery(topic="black holes", tool_hint="paper", rationale="academic")
    assert sq.topic == "black holes"
    assert sq.tool_hint == "paper"


@pytest.mark.parametrize("hint", ["email", "database", "llm", ""])
def test_sub_query_rejects_invalid_tool_hint(hint: str) -> None:
    with pytest.raises(ValidationError):
        SubQuery(topic="test", tool_hint=hint, rationale="r")  # type: ignore[arg-type]


def test_sub_query_strips_whitespace() -> None:
    sq = SubQuery(topic="  climate change  ", tool_hint="web", rationale=" test ")
    assert sq.topic == "climate change"
    assert sq.rationale == "test"


def test_sub_query_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SubQuery(topic="t", tool_hint="web", rationale="r", unexpected_field="x")  # type: ignore[call-arg]


# ── PlannerOutput ─────────────────────────────────────────────────────────────

def test_planner_output_valid() -> None:
    po = PlannerOutput(
        sub_queries=[SubQuery(topic="AI safety", tool_hint="web", rationale="recent")],
        planning_notes="standard decomposition",
    )
    assert len(po.sub_queries) == 1
    assert po.known_topics == []


def test_planner_output_rejects_empty_sub_queries() -> None:
    with pytest.raises(ValidationError, match="sub_queries must not be empty"):
        PlannerOutput(sub_queries=[], planning_notes="nothing")


# ── SubResult ─────────────────────────────────────────────────────────────────

def test_sub_result_defaults() -> None:
    sr = SubResult(topic="t", summary="s")
    assert sr.sources == []
    assert sr.confidence == 0.0


# ── DossierOutput ─────────────────────────────────────────────────────────────

def test_dossier_output_serialises_to_dict() -> None:
    do = DossierOutput(
        title="AI in 2026",
        executive_summary="A comprehensive look.",
        final_markdown="# AI in 2026\n\nSummary here.",
        run_id="run-001",
        sections=[DossierSection(heading="Background", body="...", sources=[])],
    )
    dumped = do.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["title"] == "AI in 2026"
    assert isinstance(dumped["sections"], list)


def test_dossier_output_round_trips() -> None:
    original = DossierOutput(
        title="Test",
        executive_summary="Ex",
        final_markdown="md",
        run_id="r1",
    )
    restored = DossierOutput(**original.model_dump())
    assert restored.run_id == original.run_id

