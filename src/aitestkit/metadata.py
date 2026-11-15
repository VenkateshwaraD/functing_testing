"""RunMetadata collection: reproducibility record for every run."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys

from aitestkit.logging_utils import get_logger
from aitestkit.models import RunMetadata

logger = get_logger(__name__)

_TRACKED_DEPS = [
    "aitestkit",
    "openai",
    "pydantic",
    "anthropic",
    "litellm",
    "langgraph",
    "ragas",
    "deepeval",
    "tiktoken",
]


def _git_sha() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 — fixed args, resolved binary
            [git, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("git sha unavailable: %s", e)
        return None
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else None


def _dep_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _TRACKED_DEPS:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def collect_metadata(
    *,
    judge: str | None = None,
    embedder: str | None = None,
    version_tags: dict[str, str] | None = None,
    seed: int | None = None,
    config_hash: str | None = None,
) -> RunMetadata:
    """Capture environment + version context for a run."""
    try:
        import aitestkit

        version = aitestkit.__version__
    except (AttributeError, ImportError):
        version = "unknown"
    return RunMetadata(
        git_sha=_git_sha(),
        aitestkit_version=version,
        dependency_versions=_dep_versions(),
        judge=judge,
        embedder=embedder,
        model_versions=version_tags or {},
        seed=seed,
        config_hash=config_hash,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )
