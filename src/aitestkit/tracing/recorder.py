"""Trajectory recorder: persist and reload golden trajectories.

Single-file JSON with a ``schema_version`` — older fixtures migrate on
load, newer ones fail loudly. Used for golden-run baselines in
``trajectory_match`` and offline debugging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aitestkit.exceptions import ConfigError
from aitestkit.models import SCHEMA_VERSION, Trajectory
from aitestkit.redact import Redactor


def save_trajectory(
    trajectory: Trajectory, path: Path, *, redactor: Redactor | None = None
) -> None:
    """Write a trajectory to disk, sanitized."""
    payload = {
        "schema_version": trajectory.schema_version,
        "finish_reason": trajectory.finish_reason,
        "events": [e.model_dump(mode="json") for e in trajectory.events],
    }
    redactor = redactor or Redactor()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redactor.redact(payload), indent=2, default=str))


def load_trajectory(path: Path) -> Trajectory:
    """Load a recorded trajectory; validates schema_version."""
    if not path.exists():
        raise ConfigError(f"trajectory not found: {path}")
    payload: dict[str, Any] = json.loads(path.read_text())
    version = payload.get("schema_version", 1)
    if version > SCHEMA_VERSION:
        raise ConfigError(
            f"trajectory schema_version {version} is newer than supported "
            f"{SCHEMA_VERSION} — upgrade aitestkit"
        )
    return Trajectory.model_validate(payload)


def diff_trajectories(expected: Trajectory, actual: Trajectory) -> dict[str, Any]:
    """Aligned step diff: first divergence index + missing/unexpected steps."""
    exp_sigs = expected.step_signatures
    act_sigs = actual.step_signatures
    first_divergence = next(
        (
            i
            for i, (a, b) in enumerate(zip(exp_sigs, act_sigs, strict=False))
            if a != b
        ),
        min(len(exp_sigs), len(act_sigs)),
    )
    divergence_found = first_divergence < max(len(exp_sigs), len(act_sigs))
    return {
        "first_divergence": first_divergence if divergence_found else None,
        "expected_steps": len(exp_sigs),
        "actual_steps": len(act_sigs),
        "missing_steps": exp_sigs[len(act_sigs):],
        "unexpected_steps": act_sigs[len(exp_sigs):],
        "expected_tail": exp_sigs[first_divergence:first_divergence + 3]
        if divergence_found
        else [],
        "actual_tail": act_sigs[first_divergence:first_divergence + 3]
        if divergence_found
        else [],
        "identical": not divergence_found,
    }
