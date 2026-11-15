"""Suite configuration loading and validation.

A suite YAML declares the target(s), datasets, metrics, judge/embedder
selection, provider matrix, gates, and coverage mapping. Everything is
validated eagerly at load — misconfiguration fails as ConfigError before
a single case runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from aitestkit.exceptions import ConfigError

_DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"


def load_defaults() -> dict[str, Any]:
    """Load the shipped defaults.yaml."""
    with _DEFAULTS_PATH.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"shipped defaults {_DEFAULTS_PATH} is malformed")
    return data


class JudgeConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    api_key_env: str | None = None
    self_consistency: int = 1
    max_retries: int = 3
    timeout_seconds: int = 60


class EmbedderConfig(BaseModel):
    provider: str | None = None
    model: str = "text-embedding-3-small"
    api_key_env: str | None = None


class TargetConfig(BaseModel):
    type: str  # python|http|command
    function: str | None = None          # python target: module:function
    url: str | None = None               # http target
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    request_template: dict[str, Any] | None = None
    response_path: str | None = None
    tools_path: str | None = None
    context_path: str | None = None
    command: list[str] | None = None     # command target (argv list, never shell)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    stage: str | None = None             # retriever|reranker|generator|end_to_end


class ProviderEntry(BaseModel):
    provider: str
    model: str
    api_key_env: str | None = None
    impl: str | None = None  # "module:Class" — custom ChatProvider/Embedder
    rpm: int | None = None   # per-provider rate limit
    tpm: int | None = None


class GateMetricRule(BaseModel):
    min_mean: float | None = None
    min_p10: float | None = None
    max_p95: float | None = None
    min_per_case: float | None = None


class GatesConfig(BaseModel):
    min_pass_rate: float | None = None
    per_metric: dict[str, GateMetricRule] = Field(default_factory=dict)
    max_error_cases: int | None = None
    max_flaky: float | None = None
    max_regression_vs_baseline: float | None = None
    min_coverage: dict[str, int] = Field(default_factory=dict)


class MetricSpec(BaseModel):
    name: str
    threshold: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class PromptVariant(BaseModel):
    """One named prompt template in the prompt matrix.

    ``{input}`` inside ``template`` is substituted with the case input;
    targets that accept a ``prompt`` kwarg receive the rendered text.
    """

    name: str
    template: str


class SuiteConfig(BaseModel):
    """Validated suite definition."""

    name: str = "suite"
    target: TargetConfig | None = None
    stage_targets: dict[str, TargetConfig] = Field(default_factory=dict)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[MetricSpec] = Field(default_factory=list)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    models: list[str] = Field(default_factory=list)  # "provider:model" matrix
    providers: dict[str, ProviderEntry] = Field(default_factory=dict)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    coverage: dict[str, str] = Field(default_factory=dict)  # tag -> capability
    repeat: int = 1
    concurrency: int = 8
    case_timeout_seconds: int = 300
    max_cost_usd: float | None = None
    guardrails: dict[str, Any] = Field(default_factory=dict)
    redact: dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "patterns": []})
    version_tags: dict[str, str] = Field(default_factory=dict)
    # ^ prompt/kb/chunker/embedder versions
    memory: dict[str, Any] = Field(default_factory=dict)
    # ^ {store: "module:store"} — inspectable MemoryStore for memory metrics
    tools: dict[str, Any] = Field(default_factory=dict)
    # ^ tool registry spec: name -> {function, requires_confirmation, fault, mock}
    prompts: list[PromptVariant] = Field(default_factory=list)
    # ^ prompt matrix — each case runs once per variant (like models:)
    vectordb: dict[str, Any] = Field(default_factory=dict)
    # ^ {type: memory|chroma|faiss|qdrant|pgvector|pinecone, ...} store under test
    seed: int | None = None

    @property
    def config_hash(self) -> str:
        """Stable hash of the canonical config for run metadata."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _require_env(var_name: str, *, purpose: str) -> None:
    if not os.environ.get(var_name):
        raise ConfigError(
            f"{purpose} requires {var_name} but it is not set. "
            f"Export it or add it to a .env file (see .env.example)."
        )


_PROVIDER_KEY_ENVS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "pinecone": "PINECONE_API_KEY",
}


def default_api_key_env(provider: str) -> str:
    """Default env var holding the key for a provider."""
    return _PROVIDER_KEY_ENVS.get(provider, f"{provider.upper()}_API_KEY")


def validate_provider_envs(
    suite: SuiteConfig,
    *,
    providers: set[str] | None = None,
    skip: set[str] | None = None,
) -> list[str]:
    """Eagerly validate that every selected provider's key is present.

    Returns the list of provider names validated. Providers in ``skip``
    (e.g. ``fake``) need no key. When ``providers`` is None it is derived
    from the suite (judge, embedder, model matrix, providers section).
    """
    skip = (skip or set()) | {"fake"}
    checked: list[str] = []
    if providers is None:
        providers = {suite.judge.provider}
        if suite.embedder.provider:
            providers.add(suite.embedder.provider)
        for entry in suite.models:
            providers.add(entry.split(":", 1)[0])
        providers.update(suite.providers.keys())
    providers_used = providers
    for provider_name in sorted(providers_used):
        if provider_name in skip:
            continue
        provider_cfg = suite.providers.get(provider_name)
        if provider_cfg and provider_cfg.impl and not provider_cfg.api_key_env:
            continue  # custom impl manages its own auth
        if provider_cfg is None and ":" in provider_name:
            continue  # "module:Class" custom provider — class manages auth
        env_var = (
            provider_cfg.api_key_env
            if provider_cfg and provider_cfg.api_key_env
            else default_api_key_env(provider_name)
        )
        _require_env(env_var, purpose=f"provider '{provider_name}'")
        checked.append(provider_name)
    return checked


def _normalize_metrics(raw: Any) -> list[MetricSpec]:
    specs: list[MetricSpec] = []
    for item in raw or []:
        if isinstance(item, str):
            specs.append(MetricSpec(name=item))
        elif isinstance(item, dict):
            name = item.get("name")
            if not name:
                raise ConfigError(f"metric entry {item!r} missing 'name'")
            params = {k: v for k, v in item.items() if k not in {"name", "threshold"}}
            specs.append(
                MetricSpec(name=name, threshold=item.get("threshold"), params=params)
            )
        else:
            raise ConfigError(f"invalid metric entry: {item!r}")
    return specs


def load_suite(path: str | Path) -> SuiteConfig:
    """Load and validate a suite YAML file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"suite file not found: {path}")
    with path.open() as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as e:
            raise ConfigError(f"{path}: invalid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: suite must be a YAML mapping")
    raw = dict(raw)
    raw["metrics"] = _normalize_metrics(raw.get("metrics"))
    try:
        suite = SuiteConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"{path}: invalid suite config: {e}") from e
    _apply_env_overrides(suite)
    _validate_suite(suite, path)
    return suite


def _apply_env_overrides(suite: SuiteConfig) -> None:
    """Env vars fill fields the suite left unset — CLI > suite > env > defaults.

    ``model_fields_set`` holds only keys present in the YAML, so an unset
    judge/embedder field picks up ``AITESTKIT_*`` before falling back to
    the shipped defaults.
    """
    overrides = (
        (suite.judge, "provider", "AITESTKIT_JUDGE_PROVIDER"),
        (suite.judge, "model", "AITESTKIT_JUDGE_MODEL"),
        (suite.embedder, "provider", "AITESTKIT_EMBED_PROVIDER"),
        (suite.embedder, "model", "AITESTKIT_EMBED_MODEL"),
    )
    for section, field, env_var in overrides:
        if field not in section.model_fields_set and (
            value := os.environ.get(env_var)
        ):
            setattr(section, field, value)
    if suite.max_cost_usd is None and (
        raw_cost := os.environ.get("AITESTKIT_MAX_COST_USD")
    ):
        try:
            suite.max_cost_usd = float(raw_cost)
        except ValueError:
            raise ConfigError(
                f"AITESTKIT_MAX_COST_USD={raw_cost!r} is not a number"
            ) from None


def _validate_suite(suite: SuiteConfig, path: Path) -> None:
    if not suite.datasets and not suite.target:
        raise ConfigError(f"{path}: suite needs at least one of 'datasets' or 'target'")
    if suite.target and suite.target.type not in {
        "python", "http", "command", "langgraph", "custom",
    }:
        raise ConfigError(f"{path}: unknown target type {suite.target.type!r}")
    if suite.target and suite.target.type == "python" and not suite.target.function:
        raise ConfigError(f"{path}: python target requires 'function: module:func'")
    if suite.target and suite.target.type == "custom" and not suite.target.function:
        raise ConfigError(f"{path}: custom target requires 'function: module:Class'")
    if suite.target and suite.target.type == "command" and not suite.target.command:
        raise ConfigError(f"{path}: command target requires 'command: [argv, ...]'")
    if suite.target and suite.target.type == "http" and not suite.target.url:
        raise ConfigError(f"{path}: http target requires 'url'")
    for model_entry in suite.models:
        if ":" not in model_entry:
            raise ConfigError(
                f"{path}: models entries must be 'provider:model', got {model_entry!r}"
            )
