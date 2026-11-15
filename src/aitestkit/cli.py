"""aitestkit command-line interface."""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from aitestkit.config import SuiteConfig
    from aitestkit.metrics.base import BaseMetric, MetricContext
    from aitestkit.models import TestRun

from aitestkit.exceptions import (
    AitestkitError,
    ConfigError,
)
from aitestkit.exit_codes import ExitCode
from aitestkit.logging_utils import configure_logging, get_logger, run_id_var

logger = get_logger(__name__)

app = typer.Typer(
    name="aitestkit",
    help="Functional testing framework for agentic AI, RAG, and LLM apps.",
    no_args_is_help=True,
    add_completion=False,
)


def _make_run_id() -> str:
    return f"{int(time.time() * 1000):013x}-{uuid.uuid4().hex[:10]}"


@app.command()
def init(
    directory: Annotated[
        Path, typer.Argument(help="Where to scaffold the suite.")
    ] = Path("."),
) -> None:
    """Scaffold an example suite + dataset in a directory."""
    target_dir = directory
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "datasets").mkdir(exist_ok=True)
    suite = target_dir / "suite.yaml"
    dataset = target_dir / "datasets" / "cases.yaml"
    if suite.exists():
        typer.echo(f"suite.yaml already exists in {target_dir}", err=True)
        raise typer.Exit(ExitCode.CONFIG_ERROR)
    suite.write_text(
        """name: example-suite
target:
  type: python
  function: myapp:answer   # replace with your callable
datasets:
  - datasets/cases.yaml
metrics:
  - exact_match
  - {name: latency, max_ms: 5000}
judge:
  provider: fake         # use openai/anthropic/gemini/litellm for real judging
  model: fake
"""
    )
    dataset.write_text(
        """cases:
  - id: hello
    input: "say hello"
    expected_output: "hello"
    tags: [smoke]
"""
    )
    typer.echo(f"Scaffolded {suite} — edit target.function and run: aitestkit run suite.yaml")


def _build_metrics(suite: SuiteConfig) -> list[BaseMetric]:
    import aitestkit.metrics.deterministic  # noqa: F401 — registers metrics
    import aitestkit.metrics.llm_judge  # noqa: F401
    from aitestkit.metrics.registry import build_metric

    return [
        build_metric(spec.name, threshold=spec.threshold, params=spec.params)
        for spec in suite.metrics
    ]


def _build_ctx(suite: SuiteConfig) -> MetricContext:
    from aitestkit.judge.judge import Judge
    from aitestkit.metrics.base import MetricContext
    from aitestkit.providers.registry import make_chat_provider, make_embedder
    from aitestkit.redact import Redactor

    judge = None
    needs_judge = any(
        spec.name in {"correctness", "custom_rubric", "custom"} for spec in suite.metrics
    ) or suite.judge.provider != "fake"
    if needs_judge:
        provider = make_chat_provider(suite.judge.provider)
        judge = Judge(
            provider,
            model=suite.judge.model,
            temperature=suite.judge.temperature,
            self_consistency=suite.judge.self_consistency,
        )
    embedder = make_embedder(suite.embedder.provider) if suite.embedder.provider else None
    redactor = Redactor(
        enabled=suite.redact.get("enabled", True),
        extra_patterns=suite.redact.get("patterns") or [],
    )
    return MetricContext(
        judge=judge, embedder=embedder, embed_model=suite.embedder.model, redactor=redactor
    )


async def _run_suite(suite_path: Path) -> tuple[TestRun, bool]:
    from aitestkit.config import load_suite, validate_provider_envs
    from aitestkit.datasets import load_cases
    from aitestkit.reporting import console
    from aitestkit.runner import RunEngine
    from aitestkit.targets.factory import build_target

    suite = load_suite(suite_path)
    providers_used = {suite.judge.provider, suite.embedder.provider}
    providers_used.update(m.split(":", 1)[0] for m in suite.models)
    providers_used.update(suite.providers.keys())
    if providers_used - {"fake"}:
        validate_provider_envs(suite)
    if not suite.target:
        raise ConfigError(f"{suite_path}: no 'target' configured")
    target = build_target(suite.target)
    metrics = _build_metrics(suite)
    ctx = _build_ctx(suite)
    cases = [
        case
        for dataset in suite.datasets
        for case in load_cases(suite_path.parent / dataset)
    ]
    if not cases:
        raise ConfigError(f"{suite_path}: no test cases loaded")

    engine = RunEngine(
        target=target,
        metrics=metrics,
        ctx=ctx,
        concurrency=suite.concurrency,
        case_timeout_seconds=suite.case_timeout_seconds,
        repeat=suite.repeat,
        max_cost_usd=suite.max_cost_usd,
        models=suite.models or None,
    )
    test_run, budget_hit = await engine.run(cases, on_case_result=console.report_case)
    test_run.suite_name = suite.name
    test_run.run_id = run_id_var.get()
    test_run.metadata.finished_at = time.time()
    console.report_summary(test_run)
    return test_run, budget_hit


def _exit_for(test_run: TestRun, budget_hit: bool) -> ExitCode:
    from aitestkit.models import CaseStatus

    if budget_hit:
        return ExitCode.BUDGET_EXCEEDED
    if any(g.passed is False for g in test_run.gates):
        return ExitCode.GATE_FAILURE
    if any(r.status == CaseStatus.ERROR for r in test_run.results):
        return ExitCode.ENFORCEMENT_ABORT
    if any(r.status == CaseStatus.FAILED for r in test_run.results):
        return ExitCode.TEST_FAILURES
    return ExitCode.OK


@app.command()
def run(
    suite_path: Annotated[Path, typer.Argument(help="Path to suite.yaml")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run a test suite."""
    configure_logging(verbose=verbose)
    run_id_var.set(_make_run_id())
    try:
        test_run, budget_hit = asyncio.run(_run_suite(suite_path))
    except ConfigError as e:
        typer.echo(f"config error: {e}", err=True)
        raise typer.Exit(ExitCode.CONFIG_ERROR) from e
    except AitestkitError as e:
        typer.echo(f"run failed: {e}", err=True)
        raise typer.Exit(ExitCode.ENFORCEMENT_ABORT) from e
    raise typer.Exit(_exit_for(test_run, budget_hit))


if __name__ == "__main__":
    sys.exit(app())
