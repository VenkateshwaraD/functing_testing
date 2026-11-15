"""OpenAI chat + embedding providers (core dependency).

Wraps ``AsyncOpenAI`` with eager env validation, retry on transient errors,
optional rate limiting, and per-call usage/cost accounting.
"""

from __future__ import annotations

import time
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


class OpenAIChatProvider:
    """AsyncOpenAI-backed chat provider with retry + rate limit."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        timeout_seconds: float = 60.0,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        import os

        import openai

        key = api_key or os.environ.get(default_api_key_env("openai"))
        if not key:
            raise ConfigError(
                f"provider 'openai' requires {default_api_key_env('openai')} to be set"
            )
        self._client = openai.AsyncOpenAI(
            api_key=key, base_url=base_url, timeout=timeout_seconds
        )
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
        start = time.monotonic()
        response = await call_with_retry(
            lambda: self._call(
                messages,
                model=model,
                temperature=temperature,
                response_format=response_format,
                seed=seed,
            ),
            attempts=self._max_retries,
            what=f"openai chat ({model})",
        )
        _ = time.monotonic() - start  # latency recorded by caller via usage
        choice = response.choices[0]
        usage = _usage(response, model=model)
        return ProviderResponse(text=choice.message.content or "", usage=usage)

    async def _call(self, messages: list[Message], **params: Any) -> Any:
        import openai

        kwargs: dict[str, Any] = {
            "model": params["model"],
            "messages": messages,
            "temperature": params["temperature"],
        }
        if params.get("response_format") == "json":
            kwargs["response_format"] = {"type": "json_object"}
        if params.get("seed") is not None:
            kwargs["seed"] = params["seed"]
        try:
            return await self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            raise error_for_status(
                429, str(e), retry_after=_retry_after(e)
            ) from e
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            raise TransientProviderError(str(e)) from e
        except openai.APIStatusError as e:
            raise error_for_status(e.status_code, str(e)) from e
        except openai.APIError as e:
            raise PermanentProviderError(str(e)) from e


class OpenAIEmbedder:
    """AsyncOpenAI-backed embedding provider."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        timeout_seconds: float = 60.0,
        **kwargs: Any,
    ) -> None:
        import os

        import openai

        key = api_key or os.environ.get(default_api_key_env("openai"))
        if not key:
            raise ConfigError(
                f"provider 'openai' requires {default_api_key_env('openai')} to be set"
            )
        self._client = openai.AsyncOpenAI(api_key=key, timeout=timeout_seconds)
        self._rate_limiter = rate_limiter
        self._max_retries = max_retries

    async def embed(
        self, texts: list[str], *, model: str
    ) -> tuple[list[list[float]], Usage]:
        if self._rate_limiter:
            await self._rate_limiter.acquire()

        async def _do() -> Any:
            try:
                return await self._client.embeddings.create(
                    model=model, input=texts
                )
            except Exception as e:  # noqa: BLE001 — mapped below
                raise _map_openai_error(e) from e

        response = await call_with_retry(
            _do, attempts=self._max_retries, what=f"openai embed ({model})"
        )
        usage = _usage(response, model=model)
        return [item.embedding for item in response.data], usage


def _map_openai_error(e: Exception) -> Exception:
    import openai

    if isinstance(e, openai.RateLimitError):
        return error_for_status(429, str(e), retry_after=_retry_after(e))
    if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
        return TransientProviderError(str(e))
    if isinstance(e, openai.APIStatusError):
        return error_for_status(e.status_code, str(e))
    return PermanentProviderError(str(e))


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    raw = headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _usage(response: Any, *, model: str) -> Usage:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or getattr(
        usage, "total_tokens", 0
    )
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=estimate_cost(
            model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        provider="openai",
        model=model,
    )
