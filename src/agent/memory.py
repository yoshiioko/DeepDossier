"""BaseMemory protocol and ChromaDBMemory implementation."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Protocol, runtime_checkable

import chromadb

from src.agent.config import Settings
from src.agent.schemas import MemoryChunk


@runtime_checkable
class BaseMemory(Protocol):
    """Protocol that all memory backends must satisfy.

    Callers (nodes) depend only on this interface — never on ChromaDBMemory directly.
    """

    async def retrieve(self, query: str, n_results: int = 5, min_score: float = 0.0) -> list[MemoryChunk]:
        """Return the top-n semantically similar chunks for `query`."""
        ...

    async def write_chunks(self, chunks: list[MemoryChunk]) -> None:
        """Persist all chunks; idempotent (content-hash deduplication)."""
        ...


class ChromaDBMemory:
    """ChromaDB-backed memory store.

    All ChromaDB calls are wrapped in asyncio.to_thread() because
    the ChromaDB Python client is synchronous.
    """
    def __init__(self, settings: Settings) -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},  # score = 1 - distance
        )

    async def retrieve(self, query: str, n_results: int = 5, min_score: float = 0.0) -> list[MemoryChunk]:
        """Query ChromaDB; return only chunks with cosine similarity >= min_score."""

        def _query():
            return self._collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],  # ← request distances
            )

        raw = await asyncio.to_thread(_query)

        chunks: list[MemoryChunk] = []
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]  # ← new

        for doc, meta, dist in zip(documents, metadatas, distances):
            score = 1.0 - float(dist)  # ← cosine: 0=identical, 1=orthogonal
            if score < min_score:  # ← filter below threshold
                continue
            chunks.append(MemoryChunk(
                content=doc,
                metadata={k: str(v) for k, v in (meta or {}).items()},
                score=score,  # ← populate score field
            ))

        return chunks

    async def write_chunks(self, chunks: list[MemoryChunk]) -> None:
        """Upsert chunks; uses MD5 of content as document ID (deduplication)."""
        if not chunks:
            return
        ids = [hashlib.md5(c.content.encode()).hexdigest() for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        await asyncio.to_thread(
            self._collection.upsert,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
