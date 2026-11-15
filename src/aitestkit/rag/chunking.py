"""Chunking strategies under test: fixed, semantic, hierarchical.

Each strategy yields ``Chunk`` objects with ids stable across runs so golden
sets can reference them. Used by ``aitestkit chunk-compare`` and ingestion
stage checks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from aitestkit.providers.base import Embedder


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit produced by a chunking strategy."""

    id: str
    text: str
    doc_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def chunk_id(text: str, *, prefix: str = "") -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    return f"{prefix}{digest}"


class Chunker(Protocol):
    """Anything that splits a document into chunks."""

    name: str

    def chunk(self, text: str, *, doc_id: str = "") -> list[Chunk]: ...


class FixedChunker:
    """Character-window chunking with overlap."""

    name = "fixed"

    def __init__(self, *, size: int = 512, overlap: int = 64) -> None:
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError(f"invalid size={size} overlap={overlap}")
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, *, doc_id: str = "") -> list[Chunk]:
        chunks = []
        step = self.size - self.overlap
        for i, start in enumerate(range(0, len(text), step)):
            piece = text[start : start + self.size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        id=f"{doc_id or chunk_id(text)}:{i}",
                        text=piece,
                        doc_id=doc_id,
                        metadata={"strategy": self.name, "idx": str(i)},
                    )
                )
            if start + self.size >= len(text):
                break
        return chunks


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class SemanticChunker:
    """Groups consecutive sentences by embedding similarity.

    Splits when adjacent-sentence cosine similarity drops below
    ``threshold``; ``max_size`` bounds chunk character length.
    """

    name = "semantic"

    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold: float = 0.6,
        max_size: int = 1500,
        model: str = "fake",
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.max_size = max_size
        self.model = model

    async def achunk(self, text: str, *, doc_id: str = "") -> list[Chunk]:
        sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        if not sentences:
            return []
        vectors, _ = await self.embedder.embed(sentences, model=self.model)
        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            if (
                _cosine(vectors[i - 1], vectors[i]) < self.threshold
                or sum(len(s) for s in groups[-1]) > self.max_size
            ):
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])
        return [
            Chunk(
                id=f"{doc_id or chunk_id(text)}:{i}",
                text=" ".join(group),
                doc_id=doc_id,
                metadata={"strategy": self.name, "idx": str(i)},
            )
            for i, group in enumerate(groups)
        ]

    def chunk(self, text: str, *, doc_id: str = "") -> list[Chunk]:
        import asyncio

        return asyncio.run(self.achunk(text, doc_id=doc_id))


class HierarchicalChunker:
    """Parent/child chunking — large parents split into child windows."""

    name = "hierarchical"

    def __init__(
        self, *, parent_size: int = 2048, child_size: int = 256
    ) -> None:
        self.parent = FixedChunker(size=parent_size, overlap=0)
        self.child = FixedChunker(
            size=child_size, overlap=min(32, child_size // 4)
        )

    def chunk(self, text: str, *, doc_id: str = "") -> list[Chunk]:
        chunks: list[Chunk] = []
        for p_idx, parent in enumerate(self.parent.chunk(text, doc_id=doc_id)):
            chunks.append(
                Chunk(
                    id=f"{parent.id}:p",
                    text=parent.text,
                    doc_id=doc_id,
                    metadata={
                        "strategy": self.name,
                        "level": "parent",
                        "idx": str(p_idx),
                    },
                )
            )
            for c_idx, child in enumerate(
                self.child.chunk(parent.text, doc_id=doc_id)
            ):
                chunks.append(
                    Chunk(
                        id=f"{parent.id}:c{c_idx}",
                        text=child.text,
                        doc_id=doc_id,
                        metadata={
                            "strategy": self.name,
                            "level": "child",
                            "parent": f"{parent.id}:p",
                            "idx": str(c_idx),
                        },
                    )
                )
        return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def build_chunker(
    name: str,
    *,
    embedder: Embedder | None = None,
    size: int = 512,
    overlap: int = 64,
) -> Chunker:
    """Construct a named chunker; semantic requires an embedder."""
    if name == "fixed":
        return FixedChunker(size=size, overlap=overlap)
    if name == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking requires an embedder")
        return SemanticChunker(embedder)
    if name == "hierarchical":
        return HierarchicalChunker()
    raise ValueError(f"unknown chunker: {name!r} (fixed|semantic|hierarchical)")
