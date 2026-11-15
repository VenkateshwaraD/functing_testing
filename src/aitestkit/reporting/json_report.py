"""results.json writer + loader with schema-version migrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aitestkit.exceptions import DatasetError
from aitestkit.logging_utils import get_logger
from aitestkit.models import SCHEMA_VERSION, TestRun
from aitestkit.redact import Redactor

logger = get_logger(__name__)

_MIGRATIONS: dict[int, Any] = {}


def _migrate(payload: dict[str, Any], from_version: int) -> dict[str, Any]:
    """Apply sequential migrations up to SCHEMA_VERSION."""
    data = payload
    for version in range(from_version, SCHEMA_VERSION):
        migration = _MIGRATIONS.get(version)
        if migration is None:
            logger.warning("no migration registered from schema v%s", version)
            break
        data = migration(data)
    data["schema_version"] = SCHEMA_VERSION
    return data


def write_results(test_run: TestRun, path: Path, *, redactor: Redactor) -> None:
    """Write a full results.json, redacted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = test_run.model_dump(mode="json")
    redacted = redactor.redact(payload)
    path.write_text(json.dumps(redacted, indent=2, default=str))


def load_results(path: Path) -> TestRun:
    """Load results.json, migrating older schema versions."""
    if not path.exists():
        raise DatasetError(f"results file not found: {path}")
    payload = json.loads(path.read_text())
    version = int(payload.get("schema_version", 0))
    if version > SCHEMA_VERSION:
        raise DatasetError(
            f"{path}: schema_version {version} is newer than this aitestkit "
            f"supports ({SCHEMA_VERSION})"
        )
    if version < SCHEMA_VERSION:
        logger.info("migrating %s from schema v%s to v%s", path, version, SCHEMA_VERSION)
        payload = _migrate(payload, version)
    return TestRun.model_validate(payload)
