"""aitestkit command-line interface."""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    from aitestkit.config import SuiteConfig
    from aitestkit.metrics.base import BaseMetric, MetricContext
    from aitestkit.models import CaseResult, TestRun
    from aitestkit.providers.base import ChatProvider
    from aitestkit.providers.cassette import Cassette
    from aitestkit.redact import Redactor
    from aitestkit.runner import RunEngine
    from aitestkit.targets.tool_registry import ToolRegistry

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
    import aitestkit.metrics.agentic  # noqa: F401 — registers metrics
    import aitestkit.metrics.deterministic  # noqa: F401 — registers metrics
    import aitestkit.metrics.ingestion  # noqa: F401
    import aitestkit.metrics.llm_judge  # noqa: F401
    import aitestkit.metrics.memory  # noqa: F401
    import aitestkit.metrics.prompts  # noqa: F401
    import aitestkit.metrics.rag  # noqa: F401
    import aitestkit.metrics.retrieval  # noqa: F401
    import aitestkit.vectordb.checks  # noqa: F401
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
    cassette: Cassette | None = None,
) -> MetricContext:
    from aitestkit.judge.judge import Judge
    from aitestkit.metrics.base import MetricContext
    from aitestkit.providers.cassette import (
        CassetteChatProvider,
        CassetteEmbedder,
    )
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
        provider_cfg = suite.providers.get(provider_name)
        provider: ChatProvider | None = None
        if cassette is None or cassette.mode == "record":
            provider = make_chat_provider(
                provider_name, rpm=provider_cfg.rpm if provider_cfg else None
            )
        if cassette is not None:
            provider = CassetteChatProvider(provider, cassette, name=provider_name)
        if provider is None:
            raise ConfigError(f"could not build judge provider {provider_name!r}")
        judge = Judge(
            provider,
            model=model,
            temperature=suite.judge.temperature,
            self_consistency=suite.judge.self_consistency,
        )
        for entry in suite.models:
            judge.warn_if_self_judging(sut_model=entry.split(":", 1)[-1])
    embedder = None
    if suite.embedder.provider:
        embed_rpm = (
            suite.providers[suite.embedder.provider].rpm
            if suite.embedder.provider in suite.providers
            else None
        )
        inner = None
        if cassette is None or cassette.mode == "record":
            inner = make_embedder(suite.embedder.provider, rpm=embed_rpm)
        embedder = (
            CassetteEmbedder(inner, cassette, name=suite.embedder.provider)
            if cassette is not None
            else inner
        )
    return MetricContext(
        judge=judge,
        embedder=embedder,
        embed_model=suite.embedder.model,
        redactor=redactor or _Redactor(enabled=suite.redact.get("enabled", True)),
        memory_store=_build_memory_store(suite),
        vector_store=_build_vector_store(suite, embedder),
    )


def _build_tool_registry(suite: SuiteConfig) -> ToolRegistry | None:
    """Build the suite's ToolRegistry with fault injection + mocks."""
    from aitestkit.targets.tool_registry import build_registry

    return build_registry(suite.tools or None)


def _build_memory_store(suite: SuiteConfig) -> Any:
    """Load the suite's inspectable MemoryStore (memory.store: module:obj)."""
    from aitestkit.targets.python_target import load_callable

    spec = suite.memory.get("store")
    if not spec:
        return None
    store = load_callable(spec)
    return store() if callable(store) else store


def _build_vector_store(suite: SuiteConfig, embedder: Any) -> Any:
    """Build the configured vector store (requires an embedder)."""
    if not suite.vectordb:
        return None
    if embedder is None:
        raise ConfigError(
            "suite.vectordb requires suite.embedder to be configured "
            "(stores embed via the suite's embedder)"
        )
    from aitestkit.vectordb.base import build_vector_store

    return build_vector_store(suite.vectordb, embedder=embedder)


def _install_sigint(engine: RunEngine) -> None:
    """SIGINT drains in-flight cases; completed results already on disk."""
    import contextlib
    import signal

    with contextlib.suppress(NotImplementedError, RuntimeError):
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, engine.cancel)


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
    cassette_file: Path | None = None,
    cassette_mode: str | None = None,
    resume_from: Path | None = None,
    allow_command_targets: bool = False,
    allow_python_targets: bool = True,
) -> tuple[TestRun, bool, SuiteConfig]:
    from aitestkit.baseline import compare_runs, load_baseline, save_baseline
    from aitestkit.config import load_suite, validate_provider_envs
    from aitestkit.datasets import load_cases, load_conversations
    from aitestkit.gates import evaluate_gates
    from aitestkit.history import record_run
    from aitestkit.metadata import collect_metadata
    from aitestkit.models import TestRun
    from aitestkit.providers.cassette import Cassette
    from aitestkit.redact import Redactor
    from aitestkit.reporting import attribution, console, github_actions
    from aitestkit.reporting.html_report import write_html
    from aitestkit.reporting.json_report import write_results
    from aitestkit.reporting.junit import write_junit
    from aitestkit.reporting.stream import ResultsStream, read_stream
    from aitestkit.runner import RunEngine
    from aitestkit.targets.factory import build_target

    suite = load_suite(suite_path)
    cassette = None
    if cassette_file is not None:
        cassette = Cassette(
            cassette_file,
            mode=cassette_mode or "replay",  # type: ignore[arg-type]
            redactor=None,
        )
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
    if providers_used and not (cassette and cassette.mode == "replay"):
        validate_provider_envs(suite, providers=providers_used)
    if not suite.target:
        raise ConfigError(f"{suite_path}: no 'target' configured")

    redactor = Redactor(
        enabled=redact_enabled and suite.redact.get("enabled", True),
        extra_patterns=suite.redact.get("patterns") or [],
    )
    import os

    command_allowed = allow_command_targets or bool(
        os.environ.get("AITESTKIT_ALLOW_COMMAND_TARGETS")
    )
    python_allowed = allow_python_targets and not os.environ.get(
        "AITESTKIT_DISALLOW_PYTHON_TARGETS"
    )
    target = build_target(
        suite.target,
        allow_python_targets=python_allowed,
        allow_command_targets=command_allowed,
    )
    metrics = _build_metrics(suite)
    ctx = _build_ctx(
        suite,
        judge_provider=judge_provider,
        judge_model=judge_model,
        redactor=redactor,
        cassette=cassette,
    )
    cases = [
        case
        for dataset in suite.datasets
        for case in load_cases(suite_path.parent / dataset)
    ]
    conversations = [
        conversation
        for dataset in suite.datasets
        for conversation in load_conversations(suite_path.parent / dataset)
    ]
    if not cases and not conversations:
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

    previous_results: list[CaseResult] = []
    skip_ids: set[str] = set()
    if resume_from is not None:
        jsonl_path = (
            resume_from
            if resume_from.suffix == ".jsonl"
            else resume_from / "results.jsonl"
        )
        _header, previous_results = read_stream(jsonl_path)
        skip_ids = {r.case_id for r in previous_results}
        logger.info("resume: skipping %d completed cases", len(skip_ids))
    engine = RunEngine(
        target=target,
        metrics=metrics,
        ctx=ctx,
        concurrency=suite.concurrency,
        case_timeout_seconds=suite.case_timeout_seconds,
        repeat=suite.repeat,
        max_cost_usd=suite.max_cost_usd,
        models=suite.models or None,
        conversations=conversations,
        tool_registry=_build_tool_registry(suite),
        prompts=suite.prompts or None,
    )
    _install_sigint(engine)
    test_run, budget_hit = await engine.run(
        cases, on_case_result=on_case, skip_ids=skip_ids
    )
    if previous_results:
        test_run.results = previous_results + test_run.results
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
    if suite.prompts:
        from aitestkit.prompts import save_prompt_snapshot, snapshot_prompts

        save_prompt_snapshot(
            snapshot_prompts(cases, suite.prompts),
            out_dir / "prompts.json",
        )
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
    cassette: Annotated[
        Path | None,
        typer.Option("--cassette", help="Cassette file for record/replay"),
    ] = None,
    cassette_mode: Annotated[
        str | None,
        typer.Option("--cassette-mode", help="record|replay (default replay)"),
    ] = None,
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume",
            help="Previous results.jsonl (or its directory) — skips finished cases",
        ),
    ] = None,
    allow_command_targets: Annotated[
        bool,
        typer.Option(
            "--allow-command-targets",
            help="Permit command: targets (they execute arbitrary argv)",
        ),
    ] = False,
    allow_python_targets: Annotated[
        bool,
        typer.Option(
            "--allow-python-targets/--no-allow-python-targets",
            help="Permit python: targets (default on; disable for locked-down CI)",
        ),
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
                cassette_file=cassette,
                cassette_mode=cassette_mode,
                resume_from=resume,
                allow_command_targets=allow_command_targets,
                allow_python_targets=allow_python_targets,
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


@app.command()
def chunk_compare(
    corpus: Annotated[Path, typer.Argument(help="Text file or directory of .txt/.md docs")],
    golden: Annotated[Path, typer.Argument(help="Golden set YAML (query -> relevant_ids)")],
    strategies: Annotated[
        str, typer.Option(help="Comma-separated: fixed,semantic,hierarchical")
    ] = "fixed,hierarchical",
    embedder: Annotated[str, typer.Option(help="provider:model embedder")] = "fake:fake",
    top_k: Annotated[int, typer.Option()] = 5,
    chunk_size: Annotated[int, typer.Option()] = 512,
    chunk_overlap: Annotated[int, typer.Option()] = 64,
) -> None:
    """Re-chunk a corpus per strategy and compare retrieval quality.

    For each strategy: chunk -> embed -> index -> run golden queries ->
    recall@k / MRR / nDCG. Answers "which chunker should we use" with data
    instead of vibes.
    """
    import asyncio

    from aitestkit.metrics.base import MetricContext
    from aitestkit.metrics.retrieval import MRR, NDCG, RecallAtK
    from aitestkit.models import TargetOutput, TestCase
    from aitestkit.providers.registry import make_embedder
    from aitestkit.rag.chunking import build_chunker
    from aitestkit.rag.golden import load_golden
    from aitestkit.rag.index import InMemoryVectorStore

    provider_name, _, model = embedder.partition(":")
    embedder_inst = make_embedder(provider_name or "fake")

    async def _run() -> list[dict[str, Any]]:
        docs = _load_corpus(corpus)
        golden_set = load_golden(golden)
        results = []
        for name in [s.strip() for s in strategies.split(",") if s.strip()]:
            chunker = build_chunker(
                name,
                embedder=embedder_inst,
                size=chunk_size,
                overlap=chunk_overlap,
            )
            all_chunks = []
            for doc_id, text in docs.items():
                if hasattr(chunker, "achunk"):
                    chunks = await chunker.achunk(text, doc_id=doc_id)
                else:
                    chunks = chunker.chunk(text, doc_id=doc_id)
                all_chunks.extend(chunks)
            store = InMemoryVectorStore(embedder_inst)
            await store.upsert(all_chunks)
            metric_results = []
            for query in golden_set.queries:
                hits = await store.query(query.query, top_k=top_k)
                ids = [c.id for c, _ in hits]
                case = TestCase(
                    id=query.id,
                    input=query.query,
                    expected_retrieval_ids=query.relevant_ids,
                )
                output = TargetOutput(retrieval_ids=ids)
                for metric in (
                    RecallAtK(k=top_k, threshold=0.0),
                    MRR(threshold=0.0),
                    NDCG(k=top_k, threshold=0.0),
                ):
                    metric_results.append(await metric.evaluate(case, output, MetricContext()))
            by_name: dict[str, list[float]] = {}
            for r in metric_results:
                if r.score is not None:
                    by_name.setdefault(r.metric, []).append(r.score)
            results.append(
                {
                    "strategy": name,
                    "chunks": len(all_chunks),
                    **{
                        m: sum(v) / len(v) if v else 0.0
                        for m, v in by_name.items()
                    },
                }
            )
        return results

    rows = asyncio.run(_run())
    typer.echo(f"{'strategy':<15} {'chunks':>7} {'recall@k':>9} {'mrr':>6} {'ndcg':>6}")
    for row in rows:
        typer.echo(
            f"{row['strategy']:<15} {row['chunks']:>7} "
            f"{row.get('recall_at_k', 0):>9.3f} {row.get('mrr', 0):>6.3f} "
            f"{row.get('ndcg', 0):>6.3f}"
        )


def _load_corpus(path: Path) -> dict[str, str]:
    """Corpus = one .txt/.md file, or a directory of them."""
    if path.is_file():
        return {path.stem: path.read_text(errors="replace")}
    docs = {}
    for file in sorted(path.glob("**/*")):
        if file.suffix in {".txt", ".md"} and file.is_file():
            docs[file.stem] = file.read_text(errors="replace")
    if not docs:
        raise ConfigError(f"no .txt/.md documents under {path}")
    return docs


@app.command()
def doctor(
    suite_path: Annotated[
        Path | None, typer.Argument(help="Optional suite.yaml to check")
    ] = None,
) -> None:
    """Diagnostics: env keys, provider connectivity, optional deps, git."""
    import shutil as _shutil

    ok = True

    def check(label: str, good: bool, hint: str = "") -> None:
        nonlocal ok
        ok = ok and good
        mark = "ok " if good else "MISS"
        suffix = f" — {hint}" if hint and not good else ""
        typer.echo(f"  [{mark}] {label}{suffix}")

    typer.echo("== environment")
    check("git binary", _shutil.which("git") is not None, "needed for run metadata")
    if suite_path is not None:
        import os

        from aitestkit.config import default_api_key_env, load_suite

        suite = load_suite(suite_path)
        providers = {suite.judge.provider, suite.embedder.provider} - {"fake", None}
        providers.update(m.split(":", 1)[0] for m in suite.models)
        providers.update(suite.providers.keys())
        typer.echo(f"== suite providers ({suite.name})")
        for provider_name in sorted(p for p in providers if p):
            env = default_api_key_env(provider_name)
            check(
                f"{env}",
                bool(os.environ.get(env)),
                f"export {env} or use provider: fake / cassettes",
            )
    typer.echo("== optional extras")
    for module_name, extra in _OPTIONAL_DEPS.items():
        present = _module_present(module_name)
        check(f"aitestkit[{extra}]", present, f"pip install 'aitestkit[{extra}]'")
    raise typer.Exit(ExitCode.OK if ok else ExitCode.CONFIG_ERROR)


_OPTIONAL_DEPS = {
    "anthropic": "anthropic",
    "google.genai": "gemini",
    "litellm": "litellm",
    "langgraph": "langgraph",
    "chromadb": "vectordb",
    "faiss": "vectordb",
    "ragas": "ragas",
    "deepeval": "deepeval",
    "opentelemetry": "otel",
}


def _module_present(module_name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


if __name__ == "__main__":
    sys.exit(app())
