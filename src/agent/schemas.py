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
    score: float | None = None  # cosine similarity from retrieval; None when writing


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
    memory_chunks_used: int = 0  # how many cached chunks were injected into the compiler?







