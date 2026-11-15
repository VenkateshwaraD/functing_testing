"""Retry with jittered exponential backoff, honoring server Retry-After."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from aitestkit.logging_utils import get_logger
from aitestkit.providers.errors import TransientProviderError

logger = get_logger(__name__)


async def call_with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    what: str = "provider call",
) -> T:
    """Call ``fn`` retrying only transient failures with jittered backoff.

    A server-supplied ``retry_after`` overrides the computed delay.
    Permanent errors propagate immediately.
    """
    last_error: TransientProviderError | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except TransientProviderError as e:
            last_error = e
            if attempt == attempts - 1:
                break
            delay = _delay(e.retry_after, attempt, base_delay, max_delay)
            logger.warning(
                "%s failed transiently (attempt %s/%s), retrying in %.1fs: %s",
                what, attempt + 1, attempts, delay, e,
            )
            await asyncio.sleep(delay)
    if last_error is None:  # unreachable, but keeps mypy honest
        raise RuntimeError(f"{what} failed without an exception")
    raise last_error


def _delay(
    retry_after: float | None, attempt: int, base: float, cap: float
) -> float:
    """Full-jitter exponential delay, or the server's own hint."""
    if retry_after is not None:
        return min(float(retry_after), cap)
    exponential = base * (2**attempt)
    jitter = 0.5 + random.random() * 0.5  # noqa: S311 — non-crypto timing jitter
    delay: float = min(exponential, cap) * jitter
    return delay
