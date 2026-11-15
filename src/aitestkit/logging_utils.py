"""Logging setup and run-id context for aitestkit.

Library code logs lazily (`logger.info("msg %s", arg)`) and never prints.
CLI output goes through the console reporter, not logging.
"""

from __future__ import annotations

import contextvars
import logging

run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aitestkit_run_id", default="-"
)


class RunIdFilter(logging.Filter):
    """Inject the current run id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_var.get()
        return True


def get_logger(name: str) -> logging.Logger:
    """Return a logger tagged with the aitestkit run id."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, RunIdFilter) for f in logger.filters):
        logger.addFilter(RunIdFilter())
    return logger


def configure_logging(*, verbose: bool = False) -> None:
    """Configure root logging once for CLI entry points."""
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s [%(run_id)s] %(message)s")
    )
    handler.addFilter(RunIdFilter())
    root = logging.getLogger("aitestkit")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
