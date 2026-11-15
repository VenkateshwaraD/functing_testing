"""Provider registry: resolve ``provider[:model]`` specs to instances.

Provider implementations import lazily — a missing optional extra raises a
ConfigError naming exactly which extra to install, never an ImportError.
"""

from __future__ import annotations

from typing import Any, cast

from aitestkit.exceptions import ConfigError
from aitestkit.providers.base import ChatProvider, Embedder

_EXTRA_HINT = {
    "anthropic": "aitestkit[anthropic]",
    "gemini": "aitestkit[gemini]",
    "litellm": "aitestkit[litellm]",
    "azure": "aitestkit[litellm]",
    "ollama": "aitestkit[litellm]",
    "bedrock": "aitestkit[litellm]",
    "groq": "aitestkit[litellm]",
    "mistral": "aitestkit[litellm]",
    "vertex": "aitestkit[litellm]",
}

_CHAT_IMPLS = {
    "openai": ("openai_provider", "OpenAIChatProvider"),
    "anthropic": ("anthropic_provider", "AnthropicChatProvider"),
    "gemini": ("gemini_provider", "GeminiChatProvider"),
}

_EMBEDDER_IMPLS = {
    "openai": ("openai_provider", "OpenAIEmbedder"),
}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split ``provider:model`` into parts."""
    if ":" not in spec:
        raise ConfigError(f"expected 'provider:model', got {spec!r}")
    provider, model = spec.split(":", 1)
    if not provider or not model:
        raise ConfigError(f"malformed model spec {spec!r}")
    return provider, model


def make_chat_provider(
    provider_name: str,
    *,
    rpm: int | None = None,
    impl: str | None = None,
    allow_custom: bool = True,
    **kwargs: Any,
) -> ChatProvider:
    """Instantiate a chat provider by name (with optional shared rate limit)."""
    from aitestkit.providers.ratelimit import limiter_for

    kwargs.setdefault("rate_limiter", limiter_for(provider_name, rpm=rpm))
    # An explicit impl overrides the name lookup — silently ignoring it
    # would run a different provider than the suite author configured.
    custom_spec = impl or (provider_name if ":" in provider_name else None)
    if custom_spec is not None:
        if not allow_custom:
            raise ConfigError(
                f"custom provider {custom_spec!r} loads project code — "
                "re-run with --allow-python-targets"
            )
        provider = _load_impl(custom_spec, kind="provider", **kwargs)
        if not isinstance(provider, ChatProvider):
            raise ConfigError(
                f"custom provider {custom_spec!r} must implement the "
                "ChatProvider protocol (name attr + async complete())"
            )
        return provider
    if provider_name == "fake":
        from aitestkit.providers.fake import FakeChatProvider

        return FakeChatProvider()
    if provider_name in _CHAT_IMPLS:
        module_name, class_name = _CHAT_IMPLS[provider_name]
        return cast(
            ChatProvider, _load(provider_name, module_name, class_name, **kwargs)
        )
    if provider_name in _EXTRA_HINT:
        return cast(
            ChatProvider,
            _load("litellm", "litellm_provider", "LiteLLMProvider",
                  provider=provider_name, **kwargs),
        )
    raise ConfigError(
        f"unknown provider {provider_name!r}. "
        "Built-ins: openai, anthropic, gemini, litellm(+ollama/azure/bedrock/...), "
        "fake; or impl: 'module:Class' for a custom ChatProvider"
    )


def make_embedder(
    provider_name: str,
    *,
    rpm: int | None = None,
    impl: str | None = None,
    allow_custom: bool = True,
    **kwargs: Any,
) -> Embedder:
    """Instantiate an embedder by name (with optional shared rate limit)."""
    from aitestkit.providers.ratelimit import limiter_for

    kwargs.setdefault("rate_limiter", limiter_for(provider_name, rpm=rpm))
    custom_spec = impl or (provider_name if ":" in provider_name else None)
    if custom_spec is not None:
        if not allow_custom:
            raise ConfigError(
                f"custom embedder {custom_spec!r} loads project code — "
                "re-run with --allow-python-targets"
            )
        embedder = _load_impl(custom_spec, kind="embedder", **kwargs)
        if not isinstance(embedder, Embedder):
            raise ConfigError(
                f"custom embedder {custom_spec!r} must implement the "
                "Embedder protocol (name attr + async embed())"
            )
        return embedder
    if provider_name == "fake":
        from aitestkit.providers.fake import FakeEmbedder

        return FakeEmbedder()
    if provider_name in _EMBEDDER_IMPLS:
        module_name, class_name = _EMBEDDER_IMPLS[provider_name]
        return cast(
            Embedder, _load(provider_name, module_name, class_name, **kwargs)
        )
    if provider_name in _EXTRA_HINT:
        return cast(
            Embedder,
            _load("litellm", "litellm_provider", "LiteLLMEmbedder",
                  provider=provider_name, **kwargs),
        )
    raise ConfigError(f"provider {provider_name!r} does not support embeddings")


def _load_impl(spec: str, *, kind: str, **kwargs: Any) -> Any:
    """Instantiate a custom ``module:Class`` provider/embedder spec."""
    module_name, class_name = spec.split(":", 1)
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as e:
        raise ConfigError(
            f"cannot import custom {kind} module {module_name!r}: {e}"
        ) from e
    cls = getattr(module, class_name, None)
    if cls is None or not callable(cls):
        raise ConfigError(f"custom {kind} {spec!r} does not resolve to a class")
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(
            f"custom {kind} {spec!r}: bad constructor kwargs: {e}"
        ) from e


def _load(
    provider_name: str, module_name: str, class_name: str, **kwargs: Any
) -> Any:
    """Lazy-import a provider impl; ConfigError names the missing extra."""
    try:
        module = __import__(
            f"aitestkit.providers.{module_name}", fromlist=[class_name]
        )
        return getattr(module, class_name)(**kwargs)
    except ImportError as e:
        hint = _EXTRA_HINT.get(provider_name, f"aitestkit[{provider_name}]")
        raise ConfigError(
            f"provider '{provider_name}' needs an optional dependency: "
            f"install {hint}. ({e})"
        ) from e
