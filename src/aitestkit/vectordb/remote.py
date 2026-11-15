"""Remote vector DB adapters — qdrant, pgvector, pinecone.

Each lazy-imports its client inside the constructor so the module itself
always imports; missing extras raise ConfigError with the install hint.
They share the VectorStore protocol — reembed_sample is not supported
remotely (drift checks run locally or via InMemoryVectorStore mirrors).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aitestkit.exceptions import ConfigError
from aitestkit.rag.chunking import Chunk
from aitestkit.vectordb.base import VectorStore

if TYPE_CHECKING:
    from aitestkit.providers.base import Embedder


def build_remote_store(
    kind: str, config: dict[str, object], *, embedder: Embedder
) -> VectorStore:
    if kind == "qdrant":
        return QdrantVectorStore(
            embedder,
            host=str(config.get("host", "localhost")),
            port=int(str(config.get("port", 6333))),
            collection=str(config.get("collection", "aitestkit")),
            model=str(config.get("model", "fake")),
        )
    if kind == "pgvector":
        return PgVectorStore(
            embedder,
            dsn=str(config.get("dsn", "")),
            table=str(config.get("table", "aitestkit_chunks")),
            dims=int(str(config.get("dims", 8))),
            model=str(config.get("model", "fake")),
        )
    if kind == "pinecone":
        return PineconeVectorStore(
            embedder,
            api_key_env=str(config.get("api_key_env", "PINECONE_API_KEY")),
            index_name=str(config.get("index_name", "aitestkit")),
            dims=int(str(config.get("dims", 8))),
            model=str(config.get("model", "fake")),
        )
    raise ConfigError(f"unknown remote vectordb type {kind!r}")


class QdrantVectorStore:
    """Qdrant collection — requires ``aitestkit[qdrant]``."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        host: str,
        port: int,
        collection: str,
        model: str,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ConfigError(
                "qdrant store requires: pip install 'aitestkit[qdrant]'"
            ) from e
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection
        self.embedder = embedder
        self.model = model

    async def upsert(self, chunks: list[Chunk]) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        vectors, _ = await self.embedder.embed(
            [c.text for c in chunks], model=self.model
        )
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=len(vectors[0]), distance=Distance.COSINE
                ),
            )
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=i,
                    vector=v,
                    payload={"chunk_id": c.id, "text": c.text, **c.metadata},
                )
                for i, (c, v) in enumerate(
                    zip(chunks, vectors, strict=True), start=self._client.count(
                        self._collection
                    ).count
                )
            ],
        )

    async def query(
        self, text: str, *, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        vectors, _ = await self.embedder.embed([text], model=self.model)
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=vectors[0],
            limit=top_k,
        )
        return [
            (
                Chunk(
                    id=str(hit.payload.get("chunk_id", hit.id)),
                    text=str(hit.payload.get("text", "")),
                    metadata={
                        k: str(v)
                        for k, v in hit.payload.items()
                        if k not in {"chunk_id", "text"}
                    },
                ),
                float(hit.score),
            )
            for hit in hits
        ]

    async def count(self) -> int:
        return int(self._client.count(self._collection).count)

    async def reembed_sample(
        self, *, fraction: float = 1.0
    ) -> list[tuple[Chunk, float]]:
        raise ConfigError(
            "reembed_sample is not supported for remote stores; "
            "mirror a sample into a local store instead"
        )


class PgVectorStore:
    """pgvector table — requires ``aitestkit[pgvector]`` (psycopg 3)."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        dsn: str,
        table: str,
        dims: int,
        model: str,
    ) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError as e:
            raise ConfigError(
                "pgvector store requires: pip install 'aitestkit[pgvector]'"
            ) from e
        if not dsn:
            raise ConfigError("pgvector store requires a 'dsn' in vectordb config")
        self._dsn = dsn
        self._table = table
        self._dims = dims
        self.embedder = embedder
        self.model = model

    def _connect(self):  # type: ignore[no-untyped-def]
        import psycopg

        return psycopg.connect(self._dsn)

    @staticmethod
    def _ident(name: str) -> str:
        """SQL identifiers can't be parameterized — whitelist the charset."""
        if not name.replace("_", "").isalnum():
            raise ConfigError(f"unsafe table name {name!r} (alnum+_ only)")
        return f'"{name}"'

    async def upsert(self, chunks: list[Chunk]) -> None:
        vectors, _ = await self.embedder.embed(
            [c.text for c in chunks], model=self.model
        )
        table = self._ident(self._table)
        with self._connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "  # noqa: S608
                "(id TEXT PRIMARY KEY, text TEXT, metadata JSONB, "
                f"embedding vector({self._dims}))"
            )
            for chunk, vector in zip(chunks, vectors, strict=True):
                conn.execute(
                    f"INSERT INTO {table} (id, text, metadata, embedding) "  # noqa: S608
                    "VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
                    "text = EXCLUDED.text, metadata = EXCLUDED.metadata, "
                    "embedding = EXCLUDED.embedding",
                    (chunk.id, chunk.text, _json(chunk.metadata), str(vector)),
                )

    async def query(
        self, text: str, *, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        vectors, _ = await self.embedder.embed([text], model=self.model)
        table = self._ident(self._table)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, text, metadata, 1 - (embedding <=> %s::vector) AS score "  # noqa: S608
                f"FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (str(vectors[0]), str(vectors[0]), top_k),
            ).fetchall()
        return [
            (
                Chunk(id=r[0], text=r[1], metadata=r[2] or {}),
                float(r[3]),
            )
            for r in rows
        ]

    async def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {self._ident(self._table)}"  # noqa: S608
            ).fetchone()
        return int(row[0]) if row else 0

    async def reembed_sample(
        self, *, fraction: float = 1.0
    ) -> list[tuple[Chunk, float]]:
        raise ConfigError(
            "reembed_sample is not supported for remote stores; "
            "mirror a sample into a local store instead"
        )


class PineconeVectorStore:
    """Pinecone index — requires ``aitestkit[pinecone]``."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        api_key_env: str,
        index_name: str,
        dims: int,
        model: str,
    ) -> None:
        import os

        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as e:
            raise ConfigError(
                "pinecone store requires: pip install 'aitestkit[pinecone]'"
            ) from e
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ConfigError(
                f"pinecone store requires {api_key_env} to be set"
            )
        pc = Pinecone(api_key=api_key)
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=dims,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = pc.Index(index_name)
        self.embedder = embedder
        self.model = model

    async def upsert(self, chunks: list[Chunk]) -> None:
        vectors, _ = await self.embedder.embed(
            [c.text for c in chunks], model=self.model
        )
        records: list[dict[str, object]] = []
        for c, v in zip(chunks, vectors, strict=True):
            metadata: dict[str, object] = {"text": c.text}
            metadata.update(c.metadata)
            records.append({"id": c.id, "values": v, "metadata": metadata})
        self._index.upsert(vectors=records)

    async def query(
        self, text: str, *, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        vectors, _ = await self.embedder.embed([text], model=self.model)
        result = self._index.query(
            vector=vectors[0], top_k=top_k, include_metadata=True
        )
        return [
            (
                Chunk(
                    id=match["id"],
                    text=str(match.get("metadata", {}).get("text", "")),
                    metadata={
                        k: str(v)
                        for k, v in match.get("metadata", {}).items()
                        if k != "text"
                    },
                ),
                float(match["score"]),
            )
            for match in result.get("matches", [])
        ]

    async def count(self) -> int:
        stats = self._index.describe_index_stats()
        return int(stats.get("total_vector_count", 0))

    async def reembed_sample(
        self, *, fraction: float = 1.0
    ) -> list[tuple[Chunk, float]]:
        raise ConfigError(
            "reembed_sample is not supported for remote stores; "
            "mirror a sample into a local store instead"
        )


def _json(value: object) -> str:
    import json

    return json.dumps(value)
