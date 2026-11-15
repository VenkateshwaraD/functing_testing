"""Deterministic fake providers for tests and offline runs.

``FakeChatProvider`` returns scripted JSON ``{"score": ..., "reason": ...}``
responses matched by prompt substring, so the whole judge→metric→runner path
exercises without any API key. ``FakeEmbedder`` returns stable hash vectors.
"""

from __future__ import annotations

import hashlib
import json

from aitestkit.models import Usage
from aitestkit.providers.base import Message, ProviderResponse


class FakeChatProvider:
    """Scripted chat provider. Responses keyed by prompt substring."""

    name = "fake"

    def __init__(self, *, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[list[Message]] = []

    def respond_to(self, substring: str, payload: dict[str, object]) -> None:
        """Register a JSON response for prompts containing ``substring``."""
        self._responses[substring] = json.dumps(payload)

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        response_format: str | None = None,
        seed: int | None = None,
    ) -> ProviderResponse:
        self.calls.append(messages)
        prompt_text = " ".join(m.get("content", "") for m in messages)
        for substring, payload in self._responses.items():
            if substring in prompt_text:
                return ProviderResponse(
                    text=payload,
                    usage=Usage(
                        prompt_tokens=10, completion_tokens=5,
                        provider="fake", model=model,
                    ),
                )
        return ProviderResponse(
            text=json.dumps({"score": 1.0, "reason": "fake default pass"}),
            usage=Usage(prompt_tokens=10, completion_tokens=5, provider="fake", model=model),
        )


class FakeEmbedder:
    """Deterministic embedding provider (hash-based 8-dim vectors)."""

    name = "fake"

    def __init__(self, *, dims: int = 8) -> None:
        self._dims = dims

    async def embed(self, texts: list[str], *, model: str) -> tuple[list[list[float]], Usage]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append([b / 255.0 for b in digest[: self._dims]])
        usage = Usage(prompt_tokens=sum(len(t) for t in texts), provider="fake", model=model)
        return vectors, usage
