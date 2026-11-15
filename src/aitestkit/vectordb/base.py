"""Vector store protocol + factory.

All adapters conform to :class:`VectorStore`; optional backends lazy-import
so ``aitestkit`` works without the extra installed and errors with a useful
message when it isn't.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from aitestkit.exceptions import ConfigError
from aitestkit.rag.chunking import Chunk

if TYPE_CHECKING:
    from aitestkit.providers.base import Embedder


def _require_extra(module: str, extra: str) -> None:
    """Fail fast with an install hint when an optional dep is absent."""
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise ConfigError(
            f"vectordb type {extra!r} requires: pip install 'aitestkit[{extra}]'"
        )


class VectorStore(Protocol):
    """Minimal async vector store contract for retrieval metrics/checks."""

    async def upsert(self, chunks: list[Chunk]) -> None: ...

    async def query(
        self, text: str, *, top_k: int = 5
    ) -> list[tuple[Chunk, float]]: ...

    async def count(self) -> int: ...

    async def reembed_sample(
        self, *, fraction: float = 1.0
    ) -> list[tuple[Chunk, float]]:
        """Re-embed stored texts; cosine similarity old-vs-new per chunk."""
        ...


def build_vector_store(
    config: dict[str, object], *, embedder: Embedder
) -> VectorStore:
    """Construct the configured store; lazy-imports its backend.

    Config keys: ``type`` (memory|chroma|faiss|qdrant|pgvector|pinecone),
    plus backend-specific options (collection, path, host, index_name...).
    """
    kind = str(config.get("type", "memory"))

    if kind == "memory":
        from aitestkit.rag.index import InMemoryVectorStore

        return InMemoryVectorStore(embedder, model=str(config.get("model", "fake")))

    if kind == "chroma":
        _require_extra("chromadb", "chroma")
        from aitestkit.vectordb.chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            embedder,
            collection=str(config.get("collection", "aitestkit")),
            path=str(config["path"]) if config.get("path") else None,
            model=str(config.get("model", "fake")),
        )

    if kind == "faiss":
        _require_extra("faiss", "faiss")
        from aitestkit.vectordb.faiss_store import FaissVectorStore

        return FaissVectorStore(
            embedder,
            dims=int(str(config.get("dims", 8))),
            path=str(config["path"]) if config.get("path") else None,
            model=str(config.get("model", "fake")),
        )

    remote_modules = {
        "qdrant": "qdrant_client",
        "pgvector": "psycopg",
        "pinecone": "pinecone",
    }
    if kind in remote_modules:
        _require_extra(remote_modules[kind], kind)
        from aitestkit.vectordb.remote import build_remote_store

        return build_remote_store(kind, config, embedder=embedder)

    raise ConfigError(
        f"unknown vectordb type {kind!r} "
        "(memory|chroma|faiss|qdrant|pgvector|pinecone)"
    )
