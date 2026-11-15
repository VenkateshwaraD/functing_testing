"""In-memory vector index — reference VectorStore used by chunk-compare
and tests. Real adapters (Chroma/FAISS/pgvector) arrive with the vector DB
pillar; this is the baseline scoring implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from aitestkit.rag.chunking import Chunk, _cosine

if TYPE_CHECKING:
    from aitestkit.providers.base import Embedder


class VectorStore(Protocol):
    """Minimal vector store contract for retrieval metrics."""

    async def upsert(self, chunks: list[Chunk]) -> None: ...

    async def query(
        self, text: str, *, top_k: int = 5
    ) -> list[tuple[Chunk, float]]: ...


class InMemoryVectorStore:
    """Naive cosine-similarity store — correct, not fast. Fine for CI sets."""

    def __init__(self, embedder: Embedder, *, model: str = "fake") -> None:
        self.embedder = embedder
        self.model = model
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    async def upsert(self, chunks: list[Chunk]) -> None:
        vectors, _ = await self.embedder.embed(
            [c.text for c in chunks], model=self.model
        )
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    async def query(
        self, text: str, *, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        vectors, _ = await self.embedder.embed([text], model=self.model)
        scored = [
            (chunk, _cosine(vectors[0], vec))
            for chunk, vec in zip(self._chunks, self._vectors, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    async def reembed_sample(
        self, *, fraction: float = 1.0
    ) -> list[tuple[Chunk, float]]:
        """Embedding drift check: re-embed stored texts, measure cosine
        shift between stored vectors and fresh embeddings."""
        import random

        if not self._chunks:
            return []
        count = max(1, round(len(self._chunks) * fraction))
        sample_idx = random.Random(0).sample(  # noqa: S311 — deterministic sampling, not crypto
            range(len(self._chunks)), count
        )
        chunks = [self._chunks[i] for i in sample_idx]
        stored = [self._vectors[i] for i in sample_idx]
        fresh, _ = await self.embedder.embed(
            [c.text for c in chunks], model=self.model
        )
        return [
            (chunk, _cosine(old, new))
            for chunk, old, new in zip(chunks, stored, fresh, strict=True)
        ]
