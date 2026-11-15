"""Provider error taxonomy: transient vs permanent.

Transient errors (timeouts, 5xx, rate limits, connection drops) are worth
retrying; permanent errors (auth, bad request, unknown model) are not.
Runner maps either into ``CaseStatus.ERROR`` — never a quality failure.
"""

from __future__ import annotations

from aitestkit.exceptions import ProviderError


class TransientProviderError(ProviderError):
    """Retryable provider failure (timeout, 5xx, rate limit, network)."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentProviderError(ProviderError):
    """Non-retryable provider failure (auth, bad request, unknown model)."""


class RateLimitError(TransientProviderError):
    """429 with an optional server-provided retry delay."""


def http_status_is_transient(status: int) -> bool:
    """Which HTTP statuses are worth retrying."""
    return status == 429 or status >= 500


def error_for_status(
    status: int, message: str, *, retry_after: float | None = None
) -> ProviderError:
    """Build the typed error for an HTTP status line."""
    if status == 429:
        return RateLimitError(message, retry_after=retry_after)
    if http_status_is_transient(status):
        return TransientProviderError(message, retry_after=retry_after)
    return PermanentProviderError(message)
