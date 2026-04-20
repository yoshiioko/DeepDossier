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

    async def retrieve(self, query: str, n_results: int = 5) -> list[MemoryChunk]:
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
            name=settings.chroma_collection
        )

    async def retrieve(self, query: str, n_results: int = 5) -> list[MemoryChunk]:
        """Query ChromaDB for the top-n most similar chunks."""
        results = await asyncio.to_thread(
            self._collection.query,
            query_texts=[query],
            n_results=n_results,
        )
        chunks: list[MemoryChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for doc, meta in zip(documents, metadatas):
            chunks.append(MemoryChunk(content=doc, metadata=meta or {}))
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
