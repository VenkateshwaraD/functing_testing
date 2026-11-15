"""Per-provider async rate limiter (token bucket, requests per minute)."""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Concurrency-safe token bucket; ``acquire`` waits for a slot."""

    def __init__(self, *, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        """Wait until a request slot is free."""
        async with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                await asyncio.sleep(self._next_at - now)
                now = time.monotonic()
            self._next_at = max(now, self._next_at) + self._interval


_LIMITERS: dict[str, RateLimiter] = {}


def limiter_for(provider: str, *, rpm: int | None) -> RateLimiter | None:
    """Shared limiter per provider name; None when unconfigured."""
    if rpm is None:
        return None
    key = provider
    if key not in _LIMITERS:
        _LIMITERS[key] = RateLimiter(requests_per_minute=rpm)
    return _LIMITERS[key]
