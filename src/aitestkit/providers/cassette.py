"""LLM cassettes: record provider traffic, replay it offline.

JSONL format — one ``cassette_header`` record then one record per call keyed
by a canonical request hash. Recorded files are sanitized by the shared
``Redactor`` before they hit disk, so committed cassettes are safe. Fresh
clones replay suites with zero provider keys.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from aitestkit.exceptions import ConfigError
from aitestkit.logging_utils import get_logger
from aitestkit.models import Usage
from aitestkit.providers.base import (
    ChatProvider,
    Embedder,
    Message,
    ProviderResponse,
)
from aitestkit.redact import Redactor

logger = get_logger(__name__)

CassetteMode = Literal["record", "replay", "off"]
CASSETTE_SCHEMA_VERSION = 1


class Cassette:
    """Read/write a cassette file; mode controls behavior."""

    def __init__(
        self,
        path: Path,
        *,
        mode: CassetteMode = "replay",
        redactor: Redactor | None = None,
    ) -> None:
        if mode not in ("record", "replay", "off"):
            raise ConfigError(f"cassette mode must be record|replay|off, got {mode!r}")
        self.path = path
        self.mode = mode
        self._redactor = redactor or Redactor()
        self._records: dict[str, dict[str, Any]] = {}
        if mode == "replay":
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise ConfigError(
                f"cassette {self.path} not found — record it first or remove replay mode"
            )
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if record.get("type") == "cassette_header":
                    _check_version(record.get("schema_version", 1))
                elif record.get("type") == "call":
                    self._records[record["key"]] = record["response"]

    def lookup(self, key: str) -> dict[str, Any] | None:
        """Return a recorded response for a request key."""
        return self._records.get(key)

    def append(self, key: str, request: dict[str, Any], response: dict[str, Any]) -> None:
        """Persist one request/response pair, sanitized."""
        if not self._records:
            self._write(
                {"type": "cassette_header", "schema_version": CASSETTE_SCHEMA_VERSION}
            )
        self._write(
            self._redactor.redact(
                {"type": "call", "key": key, "request": request, "response": response}
            )
        )
        self._records[key] = response

    def _write(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def request_key(payload: dict[str, Any]) -> str:
    """Canonical hash of a request — identical prompts replay identically."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def _check_version(version: int) -> None:
    if version > CASSETTE_SCHEMA_VERSION:
        raise ConfigError(
            f"cassette schema_version {version} is newer than supported "
            f"{CASSETTE_SCHEMA_VERSION} — upgrade aitestkit"
        )


class CassetteChatProvider:
    """Wrap a ChatProvider: replay from cassette, or call+record."""

    def __init__(
        self,
        inner: ChatProvider | None,
        cassette: Cassette,
        *,
        name: str | None = None,
    ) -> None:
        self.name = name or f"cassette:{getattr(inner, 'name', 'replay')}"
        self._inner = inner
        self._cassette = cassette

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        response_format: str | None = None,
        seed: int | None = None,
    ) -> ProviderResponse:
        request = {
            "kind": "chat",
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": response_format,
            "seed": seed,
        }
        key = request_key(request)
        if self._cassette.mode == "replay":
            return self._replay(key, model=model)
        if self._inner is None:
            raise ConfigError("record mode needs a real provider under the cassette")
        response = await self._inner.complete(
            messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
            seed=seed,
        )
        self._cassette.append(
            key, request, {"text": response.text, "usage": response.usage.model_dump()}
        )
        return response

    def _replay(self, key: str, *, model: str) -> ProviderResponse:
        record = self._cassette.lookup(key)
        if record is None:
            raise ConfigError(
                f"cassette has no recording for this request "
                f"(model={model}, key={key}) — re-record the cassette"
            )
        return ProviderResponse(
            text=record["text"], usage=Usage(**record.get("usage", {}))
        )


class CassetteEmbedder:
    """Wrap an Embedder: replay from cassette, or call+record."""

    def __init__(
        self, inner: Embedder | None, cassette: Cassette, *, name: str | None = None
    ) -> None:
        self.name = name or f"cassette:{getattr(inner, 'name', 'replay')}"
        self._inner = inner
        self._cassette = cassette

    async def embed(
        self, texts: list[str], *, model: str
    ) -> tuple[list[list[float]], Usage]:
        request = {"kind": "embed", "model": model, "texts": texts}
        key = request_key(request)
        if self._cassette.mode == "replay":
            return self._replay(key, model=model)
        if self._inner is None:
            raise ConfigError("record mode needs a real embedder under the cassette")
        vectors, usage = await self._inner.embed(texts, model=model)
        self._cassette.append(
            key, request, {"vectors": vectors, "usage": usage.model_dump()}
        )
        return vectors, usage

    def _replay(self, key: str, *, model: str) -> tuple[list[list[float]], Usage]:
        record = self._cassette.lookup(key)
        if record is None:
            raise ConfigError(
                f"cassette has no embedding recording (model={model}, key={key})"
            )
        return record["vectors"], Usage(**record.get("usage", {}))
