"""Run history: append-only index of runs for trend analysis.

Each finished run appends one compact record to ``.aitestkit/history.jsonl``
in the project dir (gitignored by default — commit it deliberately if the
team wants shared trend data). ``trends`` reads it back.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from aitestkit.models import TestRun
from aitestkit.stats import aggregate_metrics

HISTORY_DIR = ".aitestkit"
HISTORY_FILE = "history.jsonl"


def history_path(project_dir: Path) -> Path:
    """Default history file location for a project."""
    return project_dir / HISTORY_DIR / HISTORY_FILE


def record_run(test_run: TestRun, *, project_dir: Path) -> Path:
    """Append a compact run record to the history index."""
    path = history_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_stats = aggregate_metrics(test_run)
    record = {
        "run_id": test_run.run_id,
        "suite": test_run.suite_name,
        "finished_at": test_run.metadata.finished_at or time.time(),
        "git_sha": test_run.metadata.git_sha,
        "config_hash": test_run.metadata.config_hash,
        "baseline_tag": test_run.baseline_tag,
        "counts": _status_counts(test_run),
        "metric_means": {k: round(v["mean"], 4) for k, v in metric_stats.items()},
        "metric_pass_rates": {
            k: round(v["pass_rate"], 4) for k, v in metric_stats.items()
        },
        "judge_cost_usd": round(
            sum(r.judge_usage.cost_usd for r in test_run.results), 6
        ),
        "sut_cost_usd": round(
            sum(r.sut_usage.cost_usd for r in test_run.results), 6
        ),
        "version_tags": test_run.metadata.model_versions,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return path


def _status_counts(test_run: TestRun) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in test_run.results:
        counts[result.status.value] = counts.get(result.status.value, 0) + 1
    return counts


def load_history(path: Path) -> list[dict[str, Any]]:
    """Read all history records oldest-first."""
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def trends(path: Path) -> dict[str, Any]:
    """Compute drift across recorded runs: metric-mean trajectories."""
    records = load_history(path)
    if not records:
        return {"runs": 0, "metrics": {}, "records": []}
    metric_series: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for metric, mean in record.get("metric_means", {}).items():
            metric_series.setdefault(metric, []).append(
                {"run_id": record["run_id"], "mean": mean, "at": record["finished_at"]}
            )
    metrics: dict[str, Any] = {}
    for metric, series in metric_series.items():
        first, last = series[0]["mean"], series[-1]["mean"]
        metrics[metric] = {
            "runs": len(series),
            "first_mean": first,
            "last_mean": last,
            "drift": round(last - first, 4),
            "series": series,
        }
    return {"runs": len(records), "metrics": metrics, "records": records}
