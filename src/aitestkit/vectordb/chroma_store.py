"""Chroma vector store adapter — requires ``aitestkit[chroma]``.

Embeddings come from our Embedder (not Chroma's built-in functions) so the
embedder under test is the one being scored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aitestkit.rag.chunking import Chunk, _cosine

if TYPE_CHECKING:

    from aitestkit.providers.base import Embedder


class ChromaVectorStore:
    """Ephemeral (default) or persistent Chroma collection."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        collection: str = "aitestkit",
        path: str | None = None,
        model: str = "fake",
    ) -> None:
        import chromadb

        client = (
            chromadb.PersistentClient(path=path)
            if path
            else chromadb.EphemeralClient()
        )
        self._collection = client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder
        self.model = model

    async def upsert(self, chunks: list[Chunk]) -> None:
        vectors, _ = await self.embedder.embed(
            [c.text for c in chunks], model=self.model
        )
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[dict(c.metadata) for c in chunks],
        )

    async def query(
        self, text: str, *, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        vectors, _ = await self.embedder.embed([text], model=self.model)
        result = self._collection.query(
            query_embeddings=vectors, n_results=top_k
        )
        out: list[tuple[Chunk, float]] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        for cid, doc, dist, meta in zip(ids, docs, distances, metadatas, strict=False):
            # cosine distance -> similarity
            out.append(
                (
                    Chunk(id=cid, text=doc or "", metadata=dict(meta or {})),
                    1.0 - float(dist),
                )
            )
        return out

    async def count(self) -> int:
        return int(self._collection.count())

    async def reembed_sample(
        self, *, fraction: float = 1.0
    ) -> list[tuple[Chunk, float]]:
        import random

        got = self._collection.get(include=["embeddings", "documents", "metadatas"])
        ids = got.get("ids") or []
        if not ids:
            return []
        count = max(1, round(len(ids) * fraction))
        picked = random.Random(0).sample(  # noqa: S311 — deterministic sample
            range(len(ids)), count
        )
        out: list[tuple[Chunk, float]] = []
        for i in picked:
            stored = got["embeddings"][i]
            doc = got["documents"][i] or ""
            fresh, _ = await self.embedder.embed([doc], model=self.model)
            out.append(
                (
                    Chunk(
                        id=ids[i],
                        text=doc,
                        metadata=dict(got["metadatas"][i] or {}),
                    ),
                    _cosine(list(stored), fresh[0]),
                )
            )
        return out
