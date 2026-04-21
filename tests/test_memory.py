"""Unit tests for src/agent/memory.py — uses ephemeral ChromaDB, zero API calls."""

from __future__ import annotations

import chromadb
import uuid

from src.agent.memory import BaseMemory, ChromaDBMemory
from src.agent.schemas import MemoryChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_memory() -> ChromaDBMemory:
    """Return a ChromaDBMemory backed by an in-memory ephemeral client."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(f"test_{uuid.uuid4().hex}")
    mem = ChromaDBMemory.__new__(ChromaDBMemory)
    mem._client = client
    mem._collection = collection
    return mem


def sample_chunk(topic: str = "quantum", content: str = "Quantum computing uses qubits.") -> MemoryChunk:
    return MemoryChunk(
        content=content,
        metadata={"topic": topic, "source_url": "https://example.com", "timestamp": "2026-01-01", "run_id": "r1"},
    )


# ── Protocol compliance ───────────────────────────────────────────────────────

def test_chroma_memory_satisfies_base_memory_protocol() -> None:
    mem = make_memory()
    assert isinstance(mem, BaseMemory), "ChromaDBMemory must satisfy BaseMemory protocol"


# ── write_chunks ──────────────────────────────────────────────────────────────

async def test_write_chunks_succeeds() -> None:
    mem = make_memory()
    await mem.write_chunks([sample_chunk()])  # should not raise


async def test_write_empty_chunks_is_noop() -> None:
    mem = make_memory()
    await mem.write_chunks([])  # should not raise


async def test_write_chunks_is_idempotent() -> None:
    mem = make_memory()
    chunk = sample_chunk()
    await mem.write_chunks([chunk])
    await mem.write_chunks([chunk])  # upsert — should not duplicate
    results = await mem.retrieve("qubits", n_results=10)
    assert len(results) == 1, "Duplicate chunks should be deduplicated by content hash"


# ── retrieve ──────────────────────────────────────────────────────────────────

async def test_retrieve_returns_written_content() -> None:
    mem = make_memory()
    chunk = sample_chunk(content="Photosynthesis converts sunlight to glucose.")
    await mem.write_chunks([chunk])
    results = await mem.retrieve("photosynthesis", n_results=1)
    assert len(results) == 1
    assert "photosynthesis" in results[0].content.lower()


async def test_retrieve_on_empty_collection_returns_empty_list() -> None:
    mem = make_memory()
    results = await mem.retrieve("anything")
    assert results == []


async def test_retrieve_returns_memory_chunks() -> None:
    mem = make_memory()
    await mem.write_chunks([sample_chunk()])
    results = await mem.retrieve("quantum")
    for r in results:
        assert isinstance(r, MemoryChunk)
        assert isinstance(r.content, str)
        assert isinstance(r.metadata, dict)
