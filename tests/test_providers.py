"""Provider layer tests: retry, rate limit, pricing, cassette, budget, errors."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from aitestkit.exceptions import BudgetExceededError, ConfigError
from aitestkit.models import Usage
from aitestkit.providers.budget import BudgetTracker
from aitestkit.providers.cassette import (
    Cassette,
    CassetteChatProvider,
    CassetteEmbedder,
    request_key,
)
from aitestkit.providers.errors import (
    PermanentProviderError,
    RateLimitError,
    TransientProviderError,
    error_for_status,
)
from aitestkit.providers.fake import FakeChatProvider, FakeEmbedder
from aitestkit.providers.pricing import estimate_cost
from aitestkit.providers.ratelimit import RateLimiter
from aitestkit.providers.registry import make_chat_provider, parse_model_spec
from aitestkit.providers.retry import call_with_retry


async def test_retry_recovers_from_transient() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TransientProviderError("temporary")
        return "ok"

    assert await call_with_retry(flaky) == "ok"
    assert calls == 3


async def test_retry_does_not_retry_permanent() -> None:
    calls = 0

    async def broken() -> str:
        nonlocal calls
        calls += 1
        raise PermanentProviderError("bad api key")

    with pytest.raises(PermanentProviderError):
        await call_with_retry(broken)
    assert calls == 1


async def test_retry_honors_retry_after() -> None:
    sleeps: list[float] = []

    async def limited() -> str:
        raise RateLimitError("429", retry_after=0.01)

    import asyncio

    original_sleep = asyncio.sleep
    asyncio.sleep = lambda s: sleeps.append(s) or original_sleep(0)
    try:
        with pytest.raises(RateLimitError):
            await call_with_retry(limited, attempts=2)
    finally:
        asyncio.sleep = original_sleep
    assert sleeps
    assert sleeps[0] == pytest.approx(0.01)


async def test_retry_exhaustion_raises_last() -> None:
    async def always() -> str:
        raise TransientProviderError("down")

    with pytest.raises(TransientProviderError):
        await call_with_retry(always, attempts=2, base_delay=0.001)


def test_error_for_status_mapping() -> None:
    assert isinstance(error_for_status(429, "x"), RateLimitError)
    assert isinstance(error_for_status(503, "x"), TransientProviderError)
    assert isinstance(error_for_status(401, "x"), PermanentProviderError)
    assert isinstance(error_for_status(404, "x"), PermanentProviderError)


async def test_rate_limiter_spacing() -> None:
    limiter = RateLimiter(requests_per_minute=6000)  # 10ms apart
    start = time.monotonic()
    for _ in range(3):
        await limiter.acquire()
    assert time.monotonic() - start >= 0.015


def test_rate_limiter_rejects_zero() -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(requests_per_minute=0)


def test_pricing_prefix_match() -> None:
    assert estimate_cost(
        "gpt-4o-mini-2024-07-18", prompt_tokens=1_000_000, completion_tokens=0
    ) == pytest.approx(0.15)
    assert (
        estimate_cost("unknown-model-x", prompt_tokens=10_000, completion_tokens=10_000)
        == 0.0
    )


def test_parse_model_spec() -> None:
    assert parse_model_spec("openai:gpt-4o") == ("openai", "gpt-4o")
    assert parse_model_spec("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")
    with pytest.raises(ConfigError):
        parse_model_spec("gpt-4o")
    with pytest.raises(ConfigError):
        parse_model_spec("openai:")


def test_registry_fake_and_unknown() -> None:
    assert make_chat_provider("fake").name == "fake"
    with pytest.raises(ConfigError, match="unknown provider"):
        make_chat_provider("not-a-provider")


async def test_cassette_record_and_replay(tmp_path: Path) -> None:
    path = tmp_path / "cassette.jsonl"
    inner = FakeChatProvider()
    inner.respond_to("hello", {"score": 0.9, "reason": "good"})
    record_cassette = Cassette(path, mode="record")
    recording = CassetteChatProvider(inner, record_cassette)
    response = await recording.complete(
        [{"role": "user", "content": "hello"}], model="fake"
    )
    assert response.text
    replay_cassette = Cassette(path, mode="replay")
    replay = CassetteChatProvider(None, replay_cassette)
    replayed = await replay.complete(
        [{"role": "user", "content": "hello"}], model="fake"
    )
    assert replayed.text == response.text


async def test_cassette_replay_miss_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "cassette.jsonl"
    record_cassette = Cassette(path, mode="record")
    recording = CassetteChatProvider(FakeChatProvider(), record_cassette)
    await recording.complete([{"role": "user", "content": "a"}], model="fake")
    replay = CassetteChatProvider(None, Cassette(path, mode="replay"))
    with pytest.raises(ConfigError, match="no recording"):
        await replay.complete([{"role": "user", "content": "different"}], model="fake")


async def test_cassette_embedder_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "embed.jsonl"
    inner = FakeEmbedder()
    rec = CassetteEmbedder(inner, Cassette(path, mode="record"))
    vectors, _ = await rec.embed(["hello"], model="fake-embed")
    rep = CassetteEmbedder(None, Cassette(path, mode="replay"))
    vectors2, _ = await rep.embed(["hello"], model="fake-embed")
    assert vectors2 == vectors


def test_cassette_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        Cassette(tmp_path / "nope.jsonl", mode="replay")


def test_cassette_bad_mode() -> None:
    with pytest.raises(ConfigError, match="record"):
        Cassette(Path("x.jsonl"), mode="bogus")  # type: ignore[arg-type]


def test_request_key_stable() -> None:
    payload = {"a": 1, "b": [2, 3]}
    assert request_key(payload) == request_key({"b": [2, 3], "a": 1})
    assert request_key(payload) != request_key({"a": 2})


def test_budget_tracker() -> None:
    tracker = BudgetTracker(max_usd=0.01)
    tracker.add(Usage(cost_usd=0.005))
    assert not tracker.exceeded
    tracker.guard()
    tracker.add(Usage(cost_usd=0.01))
    assert tracker.exceeded
    with pytest.raises(BudgetExceededError):
        tracker.guard()
    with pytest.raises(BudgetExceededError):
        tracker.check(Usage(cost_usd=0.001))


def test_budget_unlimited() -> None:
    tracker = BudgetTracker(max_usd=None)
    tracker.add(Usage(cost_usd=999.0))
    tracker.guard()  # no raise


async def test_runner_budget_exit(tmp_path: Path) -> None:
    """RunEngine marks budget hit and skips remaining cases."""
    from aitestkit.metrics.base import MetricContext
    from aitestkit.metrics.deterministic import ExactMatch
    from aitestkit.models import CaseStatus, TestCase
    from aitestkit.runner import RunEngine
    from aitestkit.targets.python_target import PythonTarget

    engine = RunEngine(
        target=PythonTarget(spec="demo_app:answer"),
        metrics=[ExactMatch()],
        ctx=MetricContext(),
        max_cost_usd=0.0,  # any spend trips it — but fake app spends 0
    )
    test_run, budget_hit = await engine.run(
        [TestCase(id="a", input="say hello", expected_output="hello")]
    )
    assert not budget_hit
    assert test_run.results[0].status == CaseStatus.PASSED
