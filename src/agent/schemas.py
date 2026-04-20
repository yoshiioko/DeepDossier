"""All Pydantic v2 data models for DeepDossier."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, model_validator


# ──────────────────────────────────────────────
# Planning
# ──────────────────────────────────────────────

class SubQuery(BaseModel):
    """One decomposed research sub-task produced by planner_node."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: str
    tool_hint: Literal["web", "wiki", "paper"]
    rationale: str


class PlannerOutput(BaseModel):
    """Structured output of planner_node."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sub_queries: list[SubQuery]
    known_topics: list[str] = []
    planning_notes: str

    @model_validator(mode="after")
    def require_sub_queries(self) -> "PlannerOutput":
        if not self.sub_queries:
            raise ValueError("sub_queries must not be empty")
        return self


# ──────────────────────────────────────────────
# Research Results
# ──────────────────────────────────────────────

class SubResult(BaseModel):
    """One synthesized research result from a ResearcherSubgraph instance."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: str
    summary: str
    sources: list[str] = []
    confidence: float = 0.0


class MemoryChunk(BaseModel):
    """One unit stored in / retrieved from ChromaDB."""
    model_config = ConfigDict(extra="forbid")

    content: str
    metadata: dict[str, str]   # keys: topic, source_url, timestamp, run_id


# ──────────────────────────────────────────────
# Final Output
# ──────────────────────────────────────────────

class DossierSection(BaseModel):
    """One section of the compiled dossier."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    heading: str
    body: str
    sources: list[str] = []


class DossierOutput(BaseModel):
    """The complete research dossier produced by compiler_node."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str
    executive_summary: str
    sections: list[DossierSection] = []
    all_sources: list[str] = []
    limitations: list[str] = []
    final_markdown: str
    run_id: str
    token_usage: dict[str, int] = {}   # prompt_tokens, completion_tokens, total_tokens
    cost_usd: float | None = None


# ──────────────────────────────────────────────
# API / SSE
# ──────────────────────────────────────────────

class EventPayload(BaseModel):
    """Typed SSE event — always use this; never raw dicts over SSE."""
    model_config = ConfigDict(extra="forbid")

    event: str    # "status" | "result" | "error" | "interrupt"
    data: dict
    run_id: str


class HumanReviewPayload(BaseModel):
    """Payload emitted at the HITL interrupt point."""
    model_config = ConfigDict(extra="forbid")

    preview_markdown: str
    sub_results: list[SubResult]
    pending_run_id: str






