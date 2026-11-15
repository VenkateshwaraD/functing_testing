"""Dataset loaders: YAML, JSON, and JSONL files of test cases.

Case files contain a list of case mappings (or a top-level ``cases:`` list).
Loaded lazily per-file so large datasets don't all sit in memory at once.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from aitestkit.exceptions import DatasetError
from aitestkit.models import ConversationCase, TestCase


def _iter_payload(payload: Any, path: Path) -> Iterator[dict[str, Any]]:
    if isinstance(payload, dict) and "cases" in payload:
        payload = payload["cases"]
    elif isinstance(payload, dict) and "conversations" in payload:
        return  # conversations-only file — load_conversations() handles it
    if not isinstance(payload, list):
        raise DatasetError(f"{path}: expected a list of cases or a 'cases:' mapping")
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise DatasetError(f"{path}: case #{index} is not a mapping")
        yield item


def _iter_yaml(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        payload = yaml.safe_load(fh)
    yield from _iter_payload(payload, path)


def _iter_json(path: Path) -> Iterator[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise DatasetError(f"{path}: invalid JSON: {e}") from e
    yield from _iter_payload(payload, path)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{line_number}: invalid JSON") from e


def load_cases(path: str | Path) -> list[TestCase]:
    """Load a dataset file into ``TestCase`` objects (validated by pydantic)."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset not found: {path}")
    if path.suffix in (".yaml", ".yml"):
        raw_cases = _iter_yaml(path)
    elif path.suffix == ".json":
        raw_cases = _iter_json(path)
    elif path.suffix == ".jsonl":
        raw_cases = _iter_jsonl(path)
    else:
        raise DatasetError(
            f"{path}: unsupported extension {path.suffix} (use .yaml/.json/.jsonl)"
        )
    cases: list[TestCase] = []
    for index, raw in enumerate(raw_cases):
        try:
            case = TestCase.model_validate(raw)
        except Exception as e:
            raise DatasetError(f"{path}: case #{index} invalid: {e}") from e
        if case.id is None:
            case.id = f"{path.stem}-{index}"
        cases.append(case)
    return cases


def load_conversations(path: str | Path) -> list[ConversationCase]:
    """Load multi-turn conversation cases from the ``conversations:`` key."""
    path = Path(path)
    if not path.exists() or path.suffix not in (".yaml", ".yml", ".json"):
        return []
    try:
        payload = (
            json.loads(path.read_text())
            if path.suffix == ".json"
            else yaml.safe_load(path.read_text())
        )
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise DatasetError(f"{path}: invalid dataset file: {e}") from e
    if not isinstance(payload, dict) or "conversations" not in payload:
        return []
    raw = payload["conversations"]
    if not isinstance(raw, list):
        raise DatasetError(f"{path}: 'conversations' must be a list")
    conversations: list[ConversationCase] = []
    for index, item in enumerate(raw):
        try:
            conversation = ConversationCase.model_validate(item)
        except Exception as e:
            raise DatasetError(f"{path}: conversation #{index} invalid: {e}") from e
        if conversation.id is None:
            conversation.id = f"{path.stem}-conv-{index}"
        conversations.append(conversation)
    return conversations


def iter_cases(path: str | Path) -> Iterator[TestCase]:
    """Lazily iterate cases from a dataset file."""
    yield from load_cases(path)
