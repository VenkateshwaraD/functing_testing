"""FAISS vector store adapter — requires ``aitestkit[faiss]``.

Inner-product index over L2-normalized vectors (= cosine). Optional
``path`` persists the index + sidecar chunk map for restart checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aitestkit.rag.chunking import Chunk, _cosine

if TYPE_CHECKING:
    from aitestkit.providers.base import Embedder


class FaissVectorStore:
    """Flat (exact) FAISS index — deterministic, good for CI sets."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        dims: int = 8,
        path: str | None = None,
        model: str = "fake",
    ) -> None:
        import faiss  # noqa: F401 — ensures extra present, fails fast
        import numpy as np

        self._np = np
        self._faiss = faiss
        self._index: Any = faiss.IndexFlatIP(dims)
        self._chunks: list[Chunk] = []
        self.embedder = embedder
        self.model = model
        self.dims = dims
        self._path = Path(path) if path else None
        if self._path and self._path.with_suffix(".faiss").exists():
            self._load()

    def _normalized(self, vectors: list[list[float]]) -> object:
        array = self._np.asarray(vectors, dtype="float32")
        norms = self._np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms

    async def upsert(self, chunks: list[Chunk]) -> None:
        vectors, _ = await self.embedder.embed(
            [c.text for c in chunks], model=self.model
        )
        self._index.add(self._normalized(vectors))
        self._chunks.extend(chunks)
        self._save()

    async def query(
        self, text: str, *, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        vectors, _ = await self.embedder.embed([text], model=self.model)
        scores, idxs = self._index.search(
            self._normalized(vectors), min(top_k, len(self._chunks))
        )
        return [
            (self._chunks[int(i)], float(s))
            for i, s in zip(idxs[0], scores[0], strict=True)
            if int(i) >= 0
        ]

    async def count(self) -> int:
        return int(self._index.ntotal)

    async def reembed_sample(
        self, *, fraction: float = 1.0
    ) -> list[tuple[Chunk, float]]:
        import random

        if not self._chunks:
            return []
        count = max(1, round(len(self._chunks) * fraction))
        picked = random.Random(0).sample(  # noqa: S311 — deterministic sample
            range(len(self._chunks)), count
        )
        out: list[tuple[Chunk, float]] = []
        for i in picked:
            stored = self._np.asarray(
                self._index.reconstruct(i), dtype="float32"
            ).tolist()
            fresh, _ = await self.embedder.embed(
                [self._chunks[i].text], model=self.model
            )
            norm = sum(x * x for x in fresh[0]) ** 0.5 or 1.0
            fresh_norm = [x / norm for x in fresh[0]]
            out.append((self._chunks[i], _cosine(stored, fresh_norm)))
        return out

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(
            self._index, str(self._path.with_suffix(".faiss"))
        )
        sidecar = [
            {"id": c.id, "text": c.text, "doc_id": c.doc_id, "metadata": c.metadata}
            for c in self._chunks
        ]
        self._path.with_suffix(".json").write_text(json.dumps(sidecar))

    def _load(self) -> None:
        assert self._path is not None
        self._index = self._faiss.read_index(str(self._path.with_suffix(".faiss")))
        sidecar = json.loads(self._path.with_suffix(".json").read_text())
        self._chunks = [Chunk(**entry) for entry in sidecar]
