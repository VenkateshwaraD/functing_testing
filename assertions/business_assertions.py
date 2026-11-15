"""Business-rule assertions — banking limits, funds, state machines."""

from __future__ import annotations

from typing import Any


def assert_positive_amount(amount: Any) -> None:
    """Assert the amount is numeric and strictly greater than zero."""
    value = float(amount)
    if value <= 0:
        raise AssertionError(f"amount must be positive, got {value}")


def assert_within_transfer_limit(
    amount: Any, *, daily_limit: float = 10_000.0
) -> None:
    """Assert a transfer amount stays under the daily limit."""
    value = float(amount)
    if value > daily_limit:
        raise AssertionError(f"amount {value} exceeds daily limit {daily_limit}")


def assert_sufficient_funds(*, balance: Any, amount: Any) -> None:
    """Assert the account balance covers the requested amount."""
    available, requested = float(balance), float(amount)
    if requested > available:
        raise AssertionError(
            f"insufficient funds: requested {requested}, balance {available}"
        )


def assert_valid_transition(
    current: str, target: str, *, allowed: dict[str, list[str]]
) -> None:
    """Assert a transaction state transition is in the allowed machine."""
    permitted = allowed.get(current, [])
    if target not in permitted:
        raise AssertionError(
            f"illegal transition {current} -> {target}; allowed: {permitted}"
        )


def assert_idempotent_retry(first: dict, second: dict) -> None:
    """Assert a retried operation returns the same confirmation, not a dup."""
    for field in ("confirmation_id", "status", "txn_id"):
        if field in first and first.get(field) != second.get(field):
            raise AssertionError(
                f"non-idempotent retry: {field} "
                f"{first.get(field)!r} != {second.get(field)!r}"
            )
