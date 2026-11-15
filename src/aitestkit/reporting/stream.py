"""Streaming results writer: results.jsonl appended per-case.

One JSON object per line — a ``run_header`` record first, then one
``case_result`` record per completed case, then a ``run_footer`` at the end.
If a run dies mid-flight, every completed case is already on disk; ``--resume``
reads the file to skip finished cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any

from pydantic import ValidationError

from aitestkit.logging_utils import get_logger
from aitestkit.models import CaseResult, TestRun
from aitestkit.redact import Redactor

logger = get_logger(__name__)


class ResultsStream:
    """Append-only per-case results writer (crash-safe)."""

    def __init__(self, path: Path, *, redactor: Redactor | None = None) -> None:
        self.path = path
        self._redactor = redactor or Redactor()
        self._fh: IO[str] | None = None

    def open(self, test_run: TestRun) -> None:
        """Open the stream and write the run header record."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        header = {
            "type": "run_header",
            "run_id": test_run.run_id,
            "suite_name": test_run.suite_name,
            "schema_version": test_run.schema_version,
            "metadata": test_run.metadata.model_dump(mode="json"),
        }
        self._write(header)

    def write_case(self, result: CaseResult) -> None:
        """Append one completed case result."""
        record = {"type": "case_result", **result.model_dump(mode="json")}
        self._write(record)

    def close(self, test_run: TestRun) -> None:
        """Write the run footer (summary + gates) and close the file."""
        footer = {
            "type": "run_footer",
            "run_id": test_run.run_id,
            "summary": test_run.summary,
            "gates": [g.model_dump(mode="json") for g in test_run.gates],
        }
        self._write(footer)
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("ResultsStream not open")
        redacted = self._redactor.redact(record)
        self._fh.write(json.dumps(redacted, default=str) + "\n")
        self._fh.flush()


def read_stream(path: Path) -> tuple[dict[str, Any] | None, list[CaseResult]]:
    """Read a results.jsonl back into (header, case results)."""
    header: dict[str, Any] | None = None
    results: list[CaseResult] = []
    if not path.exists():
        return header, results
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping corrupt line in %s", path)
                continue
            record_type = record.pop("type", None)
            if record_type == "run_header":
                header = record
            elif record_type == "case_result":
                try:
                    results.append(CaseResult.model_validate(record))
                except (ValidationError, TypeError, KeyError) as e:
                    logger.warning("skipping unreadable case record in %s: %s", path, e)
    return header, results
