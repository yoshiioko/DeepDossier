"""Pydantic AI agent singletons for DeepDossier."""

from __future__ import annotations

from pydantic_ai import Agent

from src.agent.config import Settings
from src.agent.schemas import DossierOutput, SubResult


# ── Synthesize Agent ──────────────────────────────────────────────────────────

synthesize_agent: Agent[Settings, SubResult] = Agent( # type: ignore[assignment]
    model="google-gla:gemini-2.5-flash",
    output_type=SubResult,
    deps_type=Settings,
    defer_model_check=True,
    system_prompt=(
        "You are a precise research synthesizer. "
        "Given raw content fetched from the web, Wikipedia, or ArXiv, "
        "distil it into a concise, accurate, and well-sourced SubResult. "
        "Always extract URLs into sources. "
        "Set confidence=0.0 if the content is empty, an error, or irrelevant."
    ),
)


# ── Compiler Agent ────────────────────────────────────────────────────────────

compiler_agent: Agent[Settings, DossierOutput] = Agent( # type: ignore[assignment]
    model="google-gla:gemini-2.5-flash",
    output_type=DossierOutput,
    deps_type=Settings,
    defer_model_check=True,
    system_prompt=(
        "You are a professional research compiler. "
        "Given a set of research findings, synthesise them into a comprehensive, "
        "well-structured dossier with clear sections, honest limitations, "
        "and complete source attribution. "
        "Never fabricate information — note gaps honestly."
    ),
)
