"""Typed exception taxonomy for aitestkit.

Every failure a user can hit maps to one of these — error messages carry a
remediation hint so normal errors never surface as bare tracebacks.
"""

from __future__ import annotations


class AitestkitError(Exception):
    """Base class for all aitestkit errors."""


class ConfigError(AitestkitError):
    """Suite, dataset, or environment configuration is invalid."""


class ProviderError(AitestkitError):
    """A model provider call failed (auth, quota, transient network)."""


class TargetError(AitestkitError):
    """The system under test could not be invoked or returned unusable output."""


class MetricError(AitestkitError):
    """A metric could not produce a score (e.g. judge failed)."""


class DatasetError(AitestkitError):
    """A dataset or fixture file is malformed."""


class EnforcementError(AitestkitError):
    """A guardrail (step cap, wall clock) terminated a run."""


class BudgetExceededError(AitestkitError):
    """The run-level spend cap was exceeded; results so far are preserved."""
