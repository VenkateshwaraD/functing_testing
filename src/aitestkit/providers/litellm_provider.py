"""LiteLLM provider passthrough (optional extra: aitestkit[litellm]).

Covers azure, bedrock, ollama, groq, mistral, vertex, and any other
LiteLLM-supported backend through ``litellm.acompletion``.
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
    error_for_status,
)
from aitestkit.providers.pricing import estimate_cost
from aitestkit.providers.ratelimit import RateLimiter
from aitestkit.providers.retry import call_with_retry


class LiteLLMProvider:
    """litellm.acompletion-backed chat provider."""

    def __init__(
        self,
        *,
        provider: str = "litellm",
        api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        **kwargs: Any,
    ) -> None:
        import litellm  # noqa: F401 — import check, raises ImportError for extra

        self.name = provider
        self._provider = provider
        if api_key is None:
            env = default_api_key_env(provider)
            if env and not os.environ.get(env) and provider not in _NO_KEY:
                raise ConfigError(
                    f"provider '{provider}' requires {env} to be set "
                    "(or configure providers.*.api_key_env)"
                )
        self._api_key = api_key
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
            lambda: self._call(
                messages,
                model=model,
                temperature=temperature,
                json_mode=response_format == "json",
                seed=seed,
            ),
            attempts=self._max_retries,
            what=f"{self._provider} chat ({model})",
        )
        text = response.choices[0].message.content or ""
        return ProviderResponse(
            text=text, usage=_usage(response, model=model, provider=self._provider)
        )

    async def _call(self, messages: list[Message], **params: Any) -> Any:
        import litellm

        model = params["model"]
        litellm_model = model if "/" in model else f"{self._provider}/{model}"
        kwargs: dict[str, Any] = {
            "model": litellm_model,
            "messages": messages,
            "temperature": params["temperature"],
        }
        if params.get("json_mode"):
            kwargs["response_format"] = {"type": "json_object"}
        if params.get("seed") is not None:
            kwargs["seed"] = params["seed"]
        if self._api_key:
            kwargs["api_key"] = self._api_key
        try:
            return await litellm.acompletion(**kwargs)
        except litellm.RateLimitError as e:
            raise error_for_status(429, str(e)) from e
        except (litellm.Timeout, litellm.APIConnectionError) as e:
            raise TransientProviderError(str(e)) from e
        except litellm.AuthenticationError as e:
            raise PermanentProviderError(str(e)) from e
        except litellm.APIError as e:
            status = getattr(e, "status_code", 500) or 500
            raise error_for_status(status, str(e)) from e


class LiteLLMEmbedder:
    """litellm.aembedding-backed embedder."""

    def __init__(
        self,
        *,
        provider: str = "litellm",
        api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        **kwargs: Any,
    ) -> None:
        import litellm  # noqa: F401

        self.name = provider
        self._provider = provider
        self._api_key = api_key
        self._rate_limiter = rate_limiter
        self._max_retries = max_retries

    async def embed(
        self, texts: list[str], *, model: str
    ) -> tuple[list[list[float]], Usage]:
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        response = await call_with_retry(
            lambda: self._call(texts, model=model),
            attempts=self._max_retries,
            what=f"{self._provider} embed ({model})",
        )
        vectors = [item["embedding"] for item in response.data]
        usage = _usage(response, model=model, provider=self._provider)
        return vectors, usage

    async def _call(self, texts: list[str], **params: Any) -> Any:
        import litellm

        litellm_model = params["model"]
        if "/" not in litellm_model:
            litellm_model = f"{self._provider}/{litellm_model}"
        kwargs: dict[str, Any] = {"model": litellm_model, "input": texts}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        try:
            return await litellm.aembedding(**kwargs)
        except litellm.RateLimitError as e:
            raise error_for_status(429, str(e)) from e
        except (litellm.Timeout, litellm.APIConnectionError) as e:
            raise TransientProviderError(str(e)) from e
        except litellm.AuthenticationError as e:
            raise PermanentProviderError(str(e)) from e
        except litellm.APIError as e:
            status = getattr(e, "status_code", 500) or 500
            raise error_for_status(status, str(e)) from e


_NO_KEY = {"ollama"}


def _usage(response: Any, *, model: str, provider: str) -> Usage:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or getattr(
        usage, "input_tokens", 0
    ) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or getattr(
        usage, "output_tokens", 0
    ) or 0
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=estimate_cost(
            model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        provider=provider,
        model=model,
    )
