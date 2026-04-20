"""Phase 1 smoke test — run with: uv run python -m src.agent.sanity"""

from __future__ import annotations

import asyncio
import sys

import chromadb

from src.agent.config import Settings
from src.agent.memory import ChromaDBMemory
from src.agent.schemas import (
    DossierOutput,
    MemoryChunk,
    PlannerOutput,
    SubQuery,
)


def ok(msg: str) -> None:
    print(f"\033[32m[ok]\033[0m  {msg}")


def fail(msg: str) -> None:
    print(f"\033[31m[FAIL]\033[0m {msg}")
    sys.exit(1)


# ── Check 1: Settings load ────────────────────────────────────────────────────

def check_settings() -> None:
    try:
        settings = Settings(google_api_key="test-key", tavily_api_key="test-key")
        assert settings.planner_model_name == "gemini-2.5-flash"
        ok("Settings loads with overridden keys")
    except Exception as e:
        fail(f"Settings failed: {e}")


# ── Check 2: Schema validation ────────────────────────────────────────────────

def check_schemas() -> None:
    try:
        sq = SubQuery(topic="quantum computing", tool_hint="paper", rationale="peer-reviewed")
        assert sq.topic == "quantum computing"
        ok("SubQuery validates correctly")
    except Exception as e:
        fail(f"SubQuery failed: {e}")

    try:
        po = PlannerOutput(
            sub_queries=[
                SubQuery(topic="LLMs", tool_hint="web", rationale="latest news")
            ],
            planning_notes="test",
        )
        assert len(po.sub_queries) == 1
        ok("PlannerOutput validates correctly")
    except Exception as e:
        fail(f"PlannerOutput failed: {e}")

    try:
        PlannerOutput(sub_queries=[], planning_notes="empty")
        fail("PlannerOutput should reject empty sub_queries")
    except ValueError:
        ok("PlannerOutput correctly rejects empty sub_queries")

    try:
        do = DossierOutput(
            title="Test",
            executive_summary="summary",
            final_markdown="# Test",
            run_id="abc-123",
        )
        dumped = do.model_dump()
        assert isinstance(dumped, dict)
        ok("DossierOutput serialises to dict cleanly")
    except Exception as e:
        fail(f"DossierOutput failed: {e}")


# ── Check 3: ChromaDB round-trip ──────────────────────────────────────────────

async def check_memory() -> None:
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("sanity_test")

    mem = ChromaDBMemory.__new__(ChromaDBMemory)
    mem._client = client
    mem._collection = collection

    chunks = [
        MemoryChunk(
            content="Quantum entanglement allows faster-than-light correlation.",
            metadata={"topic": "quantum", "source_url": "", "timestamp": "2026-01-01", "run_id": "s1"},
        )
    ]

    try:
        await mem.write_chunks(chunks)
        ok("ChromaDBMemory.write_chunks succeeded")
    except Exception as e:
        fail(f"write_chunks failed: {e}")

    try:
        results = await mem.retrieve("quantum entanglement", n_results=1)
        assert len(results) == 1
        assert "quantum" in results[0].content.lower()
        ok(f"ChromaDBMemory.retrieve returned: '{results[0].content[:60]}...'")
    except Exception as e:
        fail(f"retrieve failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n=== DeepDossier Phase 1 Sanity Check ===\n")
    check_settings()
    check_schemas()
    await check_memory()
    print("\n✅ All Phase 1 checks passed.\n")


if __name__ == "__main__":
    asyncio.run(main())
