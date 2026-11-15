"""Exporter protocol — ship a finished TestRun to external platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from aitestkit.models import TestRun


class Exporter(Protocol):
    """One destination for run results (OTLP, Allure, ...)."""

    name: str

    def export(self, test_run: TestRun, out_dir: Path) -> str | None:
        """Export the run; return a human-readable note or None."""
        ...
