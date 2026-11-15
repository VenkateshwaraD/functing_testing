"""Golden retrieval sets — query → relevant chunk/doc ids.

Loaded from YAML; each query maps to the ids that SHOULD be retrieved,
which powers deterministic recall@k / MRR / nDCG.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class GoldenQuery(BaseModel):
    """One labelled retrieval example."""

    id: str
    query: str
    relevant_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    expected_answer: str | None = None


class GoldenSet(BaseModel):
    """Versioned golden set file."""

    schema_version: int = 1
    corpus_id: str = ""
    queries: list[GoldenQuery] = Field(default_factory=list)


def load_golden(path: str | Path) -> GoldenSet:
    data = yaml.safe_load(Path(path).read_text()) or {}
    version = data.get("schema_version", 1)
    if version != 1:
        raise ValueError(f"unsupported golden set version: {version}")
    return GoldenSet.model_validate(data)


def save_golden(golden: GoldenSet, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(
            golden.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
    )
