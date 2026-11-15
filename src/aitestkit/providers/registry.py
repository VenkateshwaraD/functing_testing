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


def make_chat_provider(provider_name: str, **kwargs: Any) -> ChatProvider:
    """Instantiate a chat provider by name."""
    if provider_name == "fake":
        from aitestkit.providers.fake import FakeChatProvider

        return FakeChatProvider(**kwargs)
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
        "Built-ins: openai, anthropic, gemini, litellm(+ollama/azure/bedrock/...), fake"
    )


def make_embedder(provider_name: str, **kwargs: Any) -> Embedder:
    """Instantiate an embedder by name."""
    if provider_name == "fake":
        from aitestkit.providers.fake import FakeEmbedder

        return FakeEmbedder(**kwargs)
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
