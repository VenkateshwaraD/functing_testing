"""Anthropic chat provider (optional extra: aitestkit[anthropic])."""

from __future__ import annotations

import os
from typing import Any

from aitestkit.config import default_api_key_env
from aitestkit.exceptions import ConfigError
from aitestkit.models import Usage
from aitestkit.providers.base import Message, ProviderResponse
from aitestkit.providers.errors import (
    PermanentProviderError,
    TransientProviderError,
    error_for_status,
)
from aitestkit.providers.pricing import estimate_cost
from aitestkit.providers.ratelimit import RateLimiter
from aitestkit.providers.retry import call_with_retry

_SYSTEM_MARKER = "system"


class AnthropicChatProvider:
    """AsyncAnthropic-backed chat provider with retry + rate limit."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        timeout_seconds: float = 60.0,
        **kwargs: Any,
    ) -> None:
        import anthropic

        key = api_key or os.environ.get(default_api_key_env("anthropic"))
        if not key:
            raise ConfigError(
                f"provider 'anthropic' requires {default_api_key_env('anthropic')}"
            )
        self._client = anthropic.AsyncAnthropic(api_key=key, timeout=timeout_seconds)
        self._rate_limiter = rate_limiter
        self._max_retries = max_retries

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        response_format: str | None = None,
        seed: int | None = None,
    ) -> ProviderResponse:
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        response = await call_with_retry(
            lambda: self._call(messages, model=model, temperature=temperature),
            attempts=self._max_retries,
            what=f"anthropic chat ({model})",
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return ProviderResponse(text=text, usage=_usage(response, model=model))

    async def _call(self, messages: list[Message], **params: Any) -> Any:
        import anthropic

        system, chat = _split_system(messages)
        kwargs: dict[str, Any] = {
            "model": params["model"],
            "messages": chat,
            "temperature": params["temperature"],
            "max_tokens": 4096,
        }
        if system:
            kwargs["system"] = system
        try:
            return await self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise error_for_status(429, str(e)) from e
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
            raise TransientProviderError(str(e)) from e
        except anthropic.APIStatusError as e:
            raise error_for_status(e.status_code, str(e)) from e
        except anthropic.APIError as e:
            raise PermanentProviderError(str(e)) from e


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Anthropic takes system separately from the message list."""
    system_parts = [m["content"] for m in messages if m.get("role") == _SYSTEM_MARKER]
    chat = [m for m in messages if m.get("role") != _SYSTEM_MARKER]
    return ("\n\n".join(system_parts) or None), chat


def _usage(response: Any, *, model: str) -> Usage:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
    completion_tokens = getattr(usage, "output_tokens", 0) or 0
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=estimate_cost(
            model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        provider="anthropic",
        model=model,
    )
