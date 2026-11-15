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
    from aitestkit.models import CaseResult, TestRun
    from aitestkit.redact import Redactor

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
gates:
  min_pass_rate: 1.0
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


def _build_ctx(
    suite: SuiteConfig,
    *,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    redactor: Redactor | None = None,
) -> MetricContext:
    from aitestkit.judge.judge import Judge
    from aitestkit.metrics.base import MetricContext
    from aitestkit.providers.registry import make_chat_provider, make_embedder
    from aitestkit.redact import Redactor as _Redactor

    provider_name = judge_provider or suite.judge.provider
    model = judge_model or suite.judge.model
    judge = None
    needs_judge = any(
        spec.name in {"correctness", "custom_rubric", "faithfulness"} | _JUDGE_METRICS
        for spec in suite.metrics
    )
    if needs_judge:
        provider = make_chat_provider(provider_name)
        judge = Judge(
            provider,
            model=model,
            temperature=suite.judge.temperature,
            self_consistency=suite.judge.self_consistency,
        )
    embedder = make_embedder(suite.embedder.provider) if suite.embedder.provider else None
    return MetricContext(
        judge=judge,
        embedder=embedder,
        embed_model=suite.embedder.model,
        redactor=redactor or _Redactor(enabled=suite.redact.get("enabled", True)),
    )


_JUDGE_METRICS = {
    "answer_relevancy", "groundedness", "citation_accuracy", "task_completion",
    "reasoning_coherence", "plan_validity", "contextual_precision",
    "contextual_recall", "multi_turn_coherence", "tool_args_correct",
    "memory_accuracy", "context_retention",
}


async def _run_suite(
    suite_path: Path,
    *,
    out_dir: Path,
    report_formats: set[str],
    judge_provider: str | None,
    judge_model: str | None,
    baseline_save: str | None,
    baseline_compare: Path | None,
    redact_enabled: bool,
) -> tuple[TestRun, bool, SuiteConfig]:
    from aitestkit.baseline import compare_runs, load_baseline, save_baseline
    from aitestkit.config import load_suite, validate_provider_envs
    from aitestkit.datasets import load_cases
    from aitestkit.gates import evaluate_gates
    from aitestkit.history import record_run
    from aitestkit.metadata import collect_metadata
    from aitestkit.models import TestRun
    from aitestkit.redact import Redactor
    from aitestkit.reporting import attribution, console, github_actions
    from aitestkit.reporting.html_report import write_html
    from aitestkit.reporting.json_report import write_results
    from aitestkit.reporting.junit import write_junit
    from aitestkit.reporting.stream import ResultsStream
    from aitestkit.runner import RunEngine
    from aitestkit.targets.factory import build_target

    suite = load_suite(suite_path)
    judge_needed = any(
        spec.name in {"correctness", "custom_rubric", "faithfulness"} | _JUDGE_METRICS
        for spec in suite.metrics
    )
    providers_used: set[str] = set()
    effective_judge = judge_provider or suite.judge.provider
    if judge_needed or judge_provider:
        providers_used.add(effective_judge)
    if suite.embedder.provider:
        providers_used.add(suite.embedder.provider)
    providers_used.update(m.split(":", 1)[0] for m in suite.models)
    providers_used.update(suite.providers.keys())
    providers_used.discard("fake")
    if providers_used:
        validate_provider_envs(suite, providers=providers_used)
    if not suite.target:
        raise ConfigError(f"{suite_path}: no 'target' configured")

    redactor = Redactor(
        enabled=redact_enabled and suite.redact.get("enabled", True),
        extra_patterns=suite.redact.get("patterns") or [],
    )
    target = build_target(suite.target)
    metrics = _build_metrics(suite)
    ctx = _build_ctx(
        suite, judge_provider=judge_provider, judge_model=judge_model, redactor=redactor
    )
    cases = [
        case
        for dataset in suite.datasets
        for case in load_cases(suite_path.parent / dataset)
    ]
    if not cases:
        raise ConfigError(f"{suite_path}: no test cases loaded")

    test_run = TestRun(
        run_id=run_id_var.get(),
        suite_name=suite.name,
        metadata=collect_metadata(
            judge=f"{suite.judge.provider}:{suite.judge.model}",
            embedder=f"{suite.embedder.provider}:{suite.embedder.model}",
            version_tags=suite.version_tags,
            seed=suite.seed,
            config_hash=suite.config_hash,
        ),
    )
    stream = ResultsStream(out_dir / "results.jsonl", redactor=redactor)
    stream.open(test_run)

    def on_case(result: CaseResult) -> None:
        console.report_case(result)
        stream.write_case(result)

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
    test_run, budget_hit = await engine.run(cases, on_case_result=on_case)
    test_run.run_id = run_id_var.get()
    test_run.suite_name = suite.name
    test_run.metadata = collect_metadata(
        judge=f"{suite.judge.provider}:{suite.judge.model}",
        embedder=f"{suite.embedder.provider}:{suite.embedder.model}",
        version_tags=suite.version_tags,
        seed=suite.seed,
        config_hash=suite.config_hash,
    )
    attribution.apply_attribution(test_run)

    regression_actual = None
    if baseline_compare is not None:
        baseline = load_baseline(baseline_compare)
        comparison = compare_runs(baseline, test_run)
        from aitestkit.baseline import max_metric_regression

        regression_actual = max_metric_regression(comparison)
        _echo_compare(comparison)

    test_run.gates = evaluate_gates(
        test_run, suite.gates, regression_actual=regression_actual
    )
    stream.close(test_run)
    console.report_summary(test_run)

    if "json" in report_formats:
        write_results(test_run, out_dir / "results.json", redactor=redactor)
    if "junit" in report_formats:
        write_junit(test_run, out_dir / "junit.xml", redactor=redactor)
    if "html" in report_formats:
        write_html(test_run, out_dir / "aitestkit-report.html", redactor=redactor)
    if baseline_save:
        save_baseline(test_run, out_dir / f"baseline-{baseline_save}.json", tag=baseline_save)
    record_run(test_run, project_dir=suite_path.parent)
    github_actions.emit(test_run, redactor=redactor, suite_path=suite_path)
    return test_run, budget_hit, suite


def _echo_compare(comparison: dict) -> None:
    verdict = comparison["verdict"]
    typer.echo(f"\n== compare verdict: {verdict}")
    for entry in comparison["regressions"][:10]:
        typer.echo(
            f"   REGRESS {entry['case_id']} {entry['metric']}: "
            f"{entry.get('baseline')} -> {entry.get('current')}"
        )
    for entry in comparison["improvements"][:10]:
        typer.echo(
            f"   IMPROVE {entry['case_id']} {entry['metric']}: "
            f"{entry.get('baseline')} -> {entry.get('current')}"
        )
    for drift in comparison["metadata_drift"]:
        typer.echo(f"   metadata drift: {drift}")


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
    out_dir: Annotated[
        Path, typer.Option("--out", "-o", help="Output directory for reports")
    ] = Path("scratch"),
    report: Annotated[
        list[str] | None,
        typer.Option("--report", help="Report formats: console,json,junit,html"),
    ] = None,
    judge: Annotated[
        str | None, typer.Option("--judge", help="Override judge provider")
    ] = None,
    judge_model: Annotated[
        str | None, typer.Option("--judge-model", help="Override judge model")
    ] = None,
    baseline_save: Annotated[
        str | None, typer.Option("--baseline-save", help="Save run as named baseline")
    ] = None,
    baseline: Annotated[
        Path | None, typer.Option("--baseline", help="Compare against a baseline file")
    ] = None,
    redact: Annotated[
        bool, typer.Option("--redact/--no-redact", help="Redact secrets in reports")
    ] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run a test suite."""
    configure_logging(verbose=verbose)
    run_id_var.set(_make_run_id())
    try:
        test_run, budget_hit, _suite = asyncio.run(
            _run_suite(
                suite_path,
                out_dir=out_dir,
                report_formats={
                    fmt
                    for value in (report or ["console", "json", "html"])
                    for fmt in value.split(",")
                },
                judge_provider=judge,
                judge_model=judge_model,
                baseline_save=baseline_save,
                baseline_compare=baseline,
                redact_enabled=redact,
            )
        )
    except ConfigError as e:
        typer.echo(f"config error: {e}", err=True)
        raise typer.Exit(ExitCode.CONFIG_ERROR) from e
    except AitestkitError as e:
        typer.echo(f"run failed: {e}", err=True)
        raise typer.Exit(ExitCode.ENFORCEMENT_ABORT) from e
    raise typer.Exit(_exit_for(test_run, budget_hit))


@app.command()
def report(
    results_path: Annotated[Path, typer.Argument(help="Path to results.json")],
    out_dir: Annotated[Path, typer.Option("--out", "-o")] = Path("scratch"),
    report: Annotated[
        list[str] | None, typer.Option("--report")
    ] = None,
) -> None:
    """Regenerate reports from a saved results.json."""
    from aitestkit.redact import Redactor
    from aitestkit.reporting.html_report import write_html
    from aitestkit.reporting.json_report import load_results
    from aitestkit.reporting.junit import write_junit

    test_run = load_results(results_path)
    redactor = Redactor()
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = report or ["html"]
    if "html" in formats:
        write_html(test_run, out_dir / "aitestkit-report.html", redactor=redactor)
    if "junit" in formats:
        write_junit(test_run, out_dir / "junit.xml", redactor=redactor)
    typer.echo(f"reports written to {out_dir}")


@app.command()
def compare(
    baseline_path: Annotated[Path, typer.Argument(help="Baseline results.json")],
    current_path: Annotated[Path, typer.Argument(help="Current results.json")],
    tolerance: Annotated[float, typer.Option("--tolerance")] = 0.05,
) -> None:
    """Compare a current run against a baseline; prints a verdict."""
    from aitestkit.baseline import compare_runs, load_baseline

    baseline = load_baseline(baseline_path)
    current = load_baseline(current_path)
    comparison = compare_runs(baseline, current, tolerance=tolerance)
    _echo_compare(comparison)
    verdict = comparison["verdict"]
    raise typer.Exit(
        ExitCode.TEST_FAILURES if verdict == "REGRESSION" else ExitCode.OK
    )


@app.command()
def trends(
    project_dir: Annotated[Path, typer.Argument()] = Path("."),
) -> None:
    """Show metric drift across recorded runs (.aitestkit/history.jsonl)."""
    from aitestkit.history import history_path, trends

    data = trends(history_path(project_dir))
    if not data["runs"]:
        typer.echo("no recorded runs — run a suite first")
        raise typer.Exit(ExitCode.OK)
    typer.echo(f"{data['runs']} runs recorded")
    for metric, info in sorted(data["metrics"].items()):
        typer.echo(
            f"  {metric}: mean {info['first_mean']:.3f} -> {info['last_mean']:.3f} "
            f"(drift {info['drift']:+.3f}, {info['runs']} runs)"
        )


if __name__ == "__main__":
    sys.exit(app())
