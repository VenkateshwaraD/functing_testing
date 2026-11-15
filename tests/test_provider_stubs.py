"""Provider adapter coverage via sys.modules stubs.

The real SDKs (openai/anthropic/google-genai/litellm) are optional extras —
these tests inject fake modules so request shaping, response parsing, and
error mapping get exercised without installing them or hitting the network.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from aitestkit.providers.errors import (
    PermanentProviderError,
    TransientProviderError,
)

pytestmark = pytest.mark.anyio


@contextmanager
def stub_modules(modules: dict[str, types.ModuleType]):
    sentinel = object()
    old = {name: sys.modules.get(name, sentinel) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, prev in old.items():
            if prev is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


# --------------------------------------------------------------------------
# OpenAI — shared error hierarchy so except clauses match injected classes


class _OaiAPIError(Exception):
    pass


class _OaiAPIConnectionError(_OaiAPIError):
    pass


class _OaiAPITimeoutError(_OaiAPIConnectionError):
    pass


class _OaiAPIStatusError(_OaiAPIError):
    def __init__(self, msg: str, *, status_code: int = 500) -> None:
        super().__init__(msg)
        self.status_code = status_code


class _OaiRateLimitError(_OaiAPIStatusError):
    def __init__(
        self, msg: str, *, status_code: int = 429, response: Any = None
    ) -> None:
        super().__init__(msg, status_code=status_code)
        self.response = response


def _openai_module(
    *,
    chat_result: Any = None,
    chat_exc: Exception | None = None,
    embed_result: Any = None,
    embed_exc: Exception | None = None,
) -> types.ModuleType:
    mod = types.ModuleType("openai")

    class _Completions:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            if chat_exc is not None:
                raise chat_exc
            return chat_result

    class _Embeddings:
        async def create(self, **kwargs: Any) -> Any:
            if embed_exc is not None:
                raise embed_exc
            return embed_result

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.chat = SimpleNamespace(completions=_Completions())
            self.embeddings = _Embeddings()

    mod.APIError = _OaiAPIError  # type: ignore[attr-defined]
    mod.APIStatusError = _OaiAPIStatusError  # type: ignore[attr-defined]
    mod.APIConnectionError = _OaiAPIConnectionError  # type: ignore[attr-defined]
    mod.APITimeoutError = _OaiAPITimeoutError  # type: ignore[attr-defined]
    mod.RateLimitError = _OaiRateLimitError  # type: ignore[attr-defined]
    mod.AsyncOpenAI = _Client  # type: ignore[attr-defined]
    return mod


def _chat_response(text: str = "hi") -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=3, completion_tokens=5, total_tokens=8
        ),
    )


async def test_openai_complete_success() -> None:
    from aitestkit.providers.openai_provider import OpenAIChatProvider

    mod = _openai_module(chat_result=_chat_response("hello"))
    with stub_modules({"openai": mod}):
        provider = OpenAIChatProvider(api_key="x", max_retries=1)
        resp = await provider.complete(
            [{"role": "user", "content": "say hi"}],
            model="gpt-4o-mini",
            seed=7,
            response_format="json",
        )
    assert resp.text == "hello"
    assert resp.usage.prompt_tokens == 3
    call = provider._client.chat.completions.calls[0]
    assert call["seed"] == 7
    assert call["response_format"] == {"type": "json_object"}


async def test_openai_error_mapping() -> None:
    from aitestkit.providers.openai_provider import OpenAIChatProvider

    retry_resp = SimpleNamespace(headers={"retry-after": "0.5"})
    cases: list[tuple[Exception, type[Exception]]] = [
        (
            _OaiRateLimitError("slow down", response=retry_resp),
            TransientProviderError,
        ),
        (_OaiAPITimeoutError("t/o"), TransientProviderError),
        (_OaiAPIConnectionError("conn"), TransientProviderError),
        (_OaiAPIStatusError("boom", status_code=503), TransientProviderError),
        (_OaiAPIStatusError("bad", status_code=400), PermanentProviderError),
        (_OaiAPIError("other"), PermanentProviderError),
    ]
    for exc, expected in cases:
        mod = _openai_module(chat_exc=exc)
        with stub_modules({"openai": mod}):
            provider = OpenAIChatProvider(api_key="x", max_retries=1)
            with pytest.raises(expected):
                await provider.complete(
                    [{"role": "user", "content": "x"}], model="gpt-4o-mini"
                )


async def test_openai_retry_after_honored() -> None:
    from aitestkit.providers.openai_provider import OpenAIChatProvider

    resp = SimpleNamespace(headers={"retry-after": "2.5"})
    exc = _OaiRateLimitError("rl", response=resp)
    mod = _openai_module(chat_exc=exc)
    with stub_modules({"openai": mod}):
        provider = OpenAIChatProvider(api_key="x", max_retries=1)
        with pytest.raises(TransientProviderError) as caught:
            await provider.complete([{"role": "user", "content": "x"}], model="m")
        assert getattr(caught.value, "retry_after", None) == 2.5


async def test_openai_embedder() -> None:
    from aitestkit.providers.openai_provider import OpenAIEmbedder

    result = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[0.1, 0.2]),
            SimpleNamespace(embedding=[0.3, 0.4]),
        ],
        usage=SimpleNamespace(prompt_tokens=10, total_tokens=10),
    )
    mod = _openai_module(embed_result=result)
    with stub_modules({"openai": mod}):
        embedder = OpenAIEmbedder(api_key="x", max_retries=1)
        vectors, usage = await embedder.embed(
            ["a", "b"], model="text-embedding-3-small"
        )
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert usage.prompt_tokens == 10


async def test_openai_embedder_error_mapped() -> None:
    from aitestkit.providers.openai_provider import OpenAIEmbedder

    exc = _OaiAPIStatusError("gone", status_code=404)
    mod = _openai_module(embed_exc=exc)
    with stub_modules({"openai": mod}):
        embedder = OpenAIEmbedder(api_key="x", max_retries=1)
        with pytest.raises(PermanentProviderError):
            await embedder.embed(["a"], model="m")


# --------------------------------------------------------------------------
# Anthropic — shared hierarchy


class _AntAPIError(Exception):
    pass


class _AntAPIConnectionError(_AntAPIError):
    pass


class _AntAPITimeoutError(_AntAPIConnectionError):
    pass


class _AntAPIStatusError(_AntAPIError):
    def __init__(self, msg: str, *, status_code: int = 500) -> None:
        super().__init__(msg)
        self.status_code = status_code


class _AntRateLimitError(_AntAPIStatusError):
    pass


def _anthropic_module(
    *, result: Any = None, exc: Exception | None = None
) -> types.ModuleType:
    mod = types.ModuleType("anthropic")

    class _Messages:
        async def create(self, **kwargs: Any) -> Any:
            if exc is not None:
                raise exc
            return result

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.messages = _Messages()

    mod.APIError = _AntAPIError  # type: ignore[attr-defined]
    mod.APIStatusError = _AntAPIStatusError  # type: ignore[attr-defined]
    mod.APIConnectionError = _AntAPIConnectionError  # type: ignore[attr-defined]
    mod.APITimeoutError = _AntAPITimeoutError  # type: ignore[attr-defined]
    mod.RateLimitError = _AntRateLimitError  # type: ignore[attr-defined]
    mod.AsyncAnthropic = _Client  # type: ignore[attr-defined]
    return mod


async def test_anthropic_complete_joins_text_blocks() -> None:
    from aitestkit.providers.anthropic_provider import AnthropicChatProvider

    result = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="first "),
            SimpleNamespace(type="tool_use", text="ignored"),
            SimpleNamespace(type="text", text="second"),
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2),
    )
    mod = _anthropic_module(result=result)
    with stub_modules({"anthropic": mod}):
        provider = AnthropicChatProvider(api_key="x", max_retries=1)
        resp = await provider.complete(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            model="claude-sonnet-4-5",
        )
    assert resp.text == "first second"
    assert resp.usage.prompt_tokens == 4
    assert resp.usage.provider == "anthropic"


async def test_anthropic_error_mapping() -> None:
    from aitestkit.providers.anthropic_provider import AnthropicChatProvider

    for exc, expected in [
        (_AntRateLimitError("rl"), TransientProviderError),
        (_AntAPITimeoutError("t"), TransientProviderError),
        (_AntAPIStatusError("s", status_code=500), TransientProviderError),
        (_AntAPIStatusError("b", status_code=401), PermanentProviderError),
        (_AntAPIError("e"), PermanentProviderError),
    ]:
        mod = _anthropic_module(exc=exc)
        with stub_modules({"anthropic": mod}):
            provider = AnthropicChatProvider(api_key="x", max_retries=1)
            with pytest.raises(expected):
                await provider.complete(
                    [{"role": "user", "content": "x"}], model="m"
                )


# --------------------------------------------------------------------------
# Gemini — shared error class


class _GeminiAPIError(Exception):
    def __init__(self, msg: str, *, code: int = 500) -> None:
        super().__init__(msg)
        self.code = code


def _google_modules(
    *, result: Any = None, exc: Exception | None = None
) -> dict[str, types.ModuleType]:
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai_types = types.ModuleType("google.genai.types")
    genai_errors = types.ModuleType("google.genai.errors")

    class GenerateContentConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _Models:
        async def generate_content(self, **kwargs: Any) -> Any:
            if exc is not None:
                raise exc
            return result

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.aio = SimpleNamespace(models=_Models())

    genai_types.GenerateContentConfig = GenerateContentConfig  # type: ignore[attr-defined]
    genai_errors.APIError = _GeminiAPIError  # type: ignore[attr-defined]
    genai.Client = _Client  # type: ignore[attr-defined]
    genai.types = genai_types  # type: ignore[attr-defined]
    genai.errors = genai_errors  # type: ignore[attr-defined]
    google.genai = genai  # type: ignore[attr-defined]
    return {
        "google": google,
        "google.genai": genai,
        "google.genai.types": genai_types,
        "google.genai.errors": genai_errors,
    }


async def test_gemini_complete() -> None:
    from aitestkit.providers.gemini_provider import GeminiChatProvider

    result = SimpleNamespace(
        text="gemini says hi",
        usage_metadata=SimpleNamespace(
            prompt_token_count=6, candidates_token_count=3
        ),
    )
    with stub_modules(_google_modules(result=result)):
        provider = GeminiChatProvider(api_key="x", max_retries=1)
        resp = await provider.complete(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ],
            model="gemini-2.0-flash",
            response_format="json",
        )
    assert resp.text == "gemini says hi"
    assert resp.usage.prompt_tokens == 6
    assert resp.usage.provider == "gemini"


async def test_gemini_error_mapping() -> None:
    from aitestkit.providers.gemini_provider import GeminiChatProvider

    for code, expected in [
        (429, TransientProviderError),
        (503, TransientProviderError),
        (400, PermanentProviderError),
    ]:
        with stub_modules(_google_modules(exc=_GeminiAPIError("x", code=code))):
            provider = GeminiChatProvider(api_key="x", max_retries=1)
            with pytest.raises(expected):
                await provider.complete([{"role": "user", "content": "x"}], model="m")


# --------------------------------------------------------------------------
# LiteLLM — shared error hierarchy


class _LlmAPIError(Exception):
    def __init__(self, msg: str, *, status_code: int = 500) -> None:
        super().__init__(msg)
        self.status_code = status_code


class _LlmRateLimitError(_LlmAPIError):
    pass


class _LlmTimeout(_LlmAPIError):  # noqa: N818 — mirrors litellm.Timeout
    pass


class _LlmAPIConnectionError(_LlmAPIError):
    pass


class _LlmAuthenticationError(_LlmAPIError):
    pass


def _litellm_module(
    *,
    chat_result: Any = None,
    chat_exc: Exception | None = None,
    embed_result: Any = None,
    embed_exc: Exception | None = None,
    captured: dict[str, Any] | None = None,
) -> types.ModuleType:
    mod = types.ModuleType("litellm")

    async def acompletion(**kwargs: Any) -> Any:
        if captured is not None:
            captured.update(kwargs)
        if chat_exc is not None:
            raise chat_exc
        return chat_result

    async def aembedding(**kwargs: Any) -> Any:
        if captured is not None:
            captured.update(kwargs)
        if embed_exc is not None:
            raise embed_exc
        return embed_result

    mod.APIError = _LlmAPIError  # type: ignore[attr-defined]
    mod.RateLimitError = _LlmRateLimitError  # type: ignore[attr-defined]
    mod.Timeout = _LlmTimeout  # type: ignore[attr-defined]
    mod.APIConnectionError = _LlmAPIConnectionError  # type: ignore[attr-defined]
    mod.AuthenticationError = _LlmAuthenticationError  # type: ignore[attr-defined]
    mod.acompletion = acompletion  # type: ignore[attr-defined]
    mod.aembedding = aembedding  # type: ignore[attr-defined]
    return mod


async def test_litellm_model_prefix_and_key_passthrough() -> None:
    from aitestkit.providers.litellm_provider import LiteLLMProvider

    result = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done"))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
    )
    captured: dict[str, Any] = {}
    mod = _litellm_module(chat_result=result, captured=captured)
    with stub_modules({"litellm": mod}):
        provider = LiteLLMProvider(provider="groq", api_key="k", max_retries=1)
        resp = await provider.complete(
            [{"role": "user", "content": "x"}], model="llama3"
        )
    assert resp.text == "done"
    assert captured["model"] == "groq/llama3"
    assert captured["api_key"] == "k"


async def test_litellm_ollama_needs_no_key() -> None:
    from aitestkit.providers.litellm_provider import LiteLLMProvider

    mod = _litellm_module(
        chat_result=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )
    )
    with stub_modules({"litellm": mod}):
        provider = LiteLLMProvider(provider="ollama", max_retries=1)
        resp = await provider.complete([{"role": "user", "content": "x"}], model="m")
    assert resp.text == "ok"


async def test_litellm_error_mapping() -> None:
    from aitestkit.providers.litellm_provider import LiteLLMProvider

    for exc, expected in [
        (_LlmRateLimitError("rl"), TransientProviderError),
        (_LlmTimeout("t"), TransientProviderError),
        (_LlmAPIConnectionError("c"), TransientProviderError),
        (_LlmAPIError("s", status_code=503), TransientProviderError),
        (_LlmAPIError("b", status_code=400), PermanentProviderError),
        (_LlmAuthenticationError("auth"), PermanentProviderError),
    ]:
        mod = _litellm_module(chat_exc=exc)
        with stub_modules({"litellm": mod}):
            provider = LiteLLMProvider(
                provider="litellm", api_key="k", max_retries=1
            )
            with pytest.raises(expected):
                await provider.complete([{"role": "user", "content": "x"}], model="m")


async def test_litellm_embedder() -> None:
    from aitestkit.providers.litellm_provider import LiteLLMEmbedder

    result = SimpleNamespace(
        data=[{"embedding": [0.5, 0.6]}],
        usage=SimpleNamespace(prompt_tokens=4, total_tokens=4),
    )
    captured: dict[str, Any] = {}
    mod = _litellm_module(embed_result=result, captured=captured)
    with stub_modules({"litellm": mod}):
        embedder = LiteLLMEmbedder(provider="bedrock", api_key="k", max_retries=1)
        vectors, usage = await embedder.embed(["txt"], model="titan")
    assert vectors == [[0.5, 0.6]]
    assert captured["model"] == "bedrock/titan"
    assert usage.prompt_tokens == 4
