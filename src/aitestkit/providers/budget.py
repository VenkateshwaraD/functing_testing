"""Run-level budget tracker — hard spend cap distinct from per-case limits."""

from __future__ import annotations

import threading

from aitestkit.exceptions import BudgetExceededError
from aitestkit.models import Usage


class BudgetTracker:
    """Accumulate spend across providers; raise when the cap is exceeded.

    Thread-safe — cases run concurrently and each updates the shared total.
    """

    def __init__(self, *, max_usd: float | None) -> None:
        self.max_usd = max_usd
        self._spent = 0.0
        self._exceeded = False
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def exceeded(self) -> bool:
        return self._exceeded

    def add(self, usage: Usage) -> None:
        """Record spend; mark exceeded if the cap was crossed."""
        if usage.cost_usd <= 0:
            return
        with self._lock:
            self._spent += usage.cost_usd
            if self.max_usd is not None and self._spent > self.max_usd:
                self._exceeded = True

    def check(self, usage: Usage) -> None:
        """Record spend and raise BudgetExceededError if over cap."""
        self.add(usage)
        if self._exceeded:
            raise BudgetExceededError(
                f"run budget ${self.max_usd:.4f} exceeded "
                f"(spent ${self._spent:.4f}) — stopping"
            )

    def guard(self) -> None:
        """Raise BudgetExceededError if already over cap (pre-case check)."""
        if self._exceeded:
            raise BudgetExceededError(
                f"run budget ${self.max_usd:.4f} exhausted "
                f"(spent ${self._spent:.4f})"
            )
