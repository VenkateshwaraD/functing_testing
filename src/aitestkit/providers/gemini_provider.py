"""Gemini chat provider (optional extra: aitestkit[gemini]).

Uses the ``google-genai`` SDK async client.
"""

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
)
from aitestkit.providers.pricing import estimate_cost
from aitestkit.providers.ratelimit import RateLimiter
from aitestkit.providers.retry import call_with_retry


class GeminiChatProvider:
    """google-genai-backed chat provider with retry + rate limit."""

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        **kwargs: Any,
    ) -> None:
        from google import genai

        key = api_key or os.environ.get(default_api_key_env("gemini"))
        if not key:
            raise ConfigError(
                f"provider 'gemini' requires {default_api_key_env('gemini')}"
            )
        self._client = genai.Client(api_key=key)
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
        prompt, system = _to_prompt(messages)
        response = await call_with_retry(
            lambda: self._call(
                prompt, system=system, model=model, temperature=temperature,
                json_mode=response_format == "json",
            ),
            attempts=self._max_retries,
            what=f"gemini chat ({model})",
        )
        return ProviderResponse(
            text=response.text or "", usage=_usage(response, model=model)
        )

    async def _call(
        self, prompt: str, *, system: str | None, **params: Any
    ) -> Any:
        from google.genai import errors as genai_errors
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=params["temperature"],
            system_instruction=system,
            response_mime_type="application/json" if params["json_mode"] else None,
        )
        try:
            return await self._client.aio.models.generate_content(
                model=params["model"], contents=prompt, config=config
            )
        except genai_errors.APIError as e:
            if e.code == 429:
                raise TransientProviderError(str(e)) from e
            if e.code >= 500:
                raise TransientProviderError(str(e)) from e
            raise PermanentProviderError(str(e)) from e


def _to_prompt(messages: list[Message]) -> tuple[str, str | None]:
    system = "\n\n".join(
        m["content"] for m in messages if m.get("role") == "system"
    )
    lines = [
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in messages
        if m.get("role") != "system"
    ]
    return "\n\n".join(lines), system or None


def _usage(response: Any, *, model: str) -> Usage:
    meta = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
    completion_tokens = getattr(meta, "candidates_token_count", 0) or 0
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=estimate_cost(
            model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        provider="gemini",
        model=model,
    )
