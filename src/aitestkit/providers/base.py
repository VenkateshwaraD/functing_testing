"""Provider protocols: chat completion and embeddings.

All framework-internal model use (judge, similarity metrics) goes through
these protocols so any provider can back them.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from aitestkit.models import Usage

Message = dict[str, str]


class ProviderResponse(BaseModel):
    """One chat completion response with usage accounting."""

    text: str
    usage: Usage
    raw: dict[str, Any] | None = None


@runtime_checkable
class ChatProvider(Protocol):
    """Async chat completion provider (e.g. OpenAI, Anthropic, LiteLLM)."""

    name: str

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        response_format: str | None = None,
        seed: int | None = None,
    ) -> ProviderResponse:
        """Complete a chat conversation and return text + usage."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Async embedding provider."""

    name: str

    async def embed(self, texts: list[str], *, model: str) -> tuple[list[list[float]], Usage]:
        """Return embedding vectors and usage for the given texts."""
        ...
