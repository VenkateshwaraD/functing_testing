"""CLI command coverage: init, doctor, trends, compare, report, export, run edge paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from aitestkit.cli import app
from aitestkit.models import (
    CaseResult,
    CaseStatus,
    MetricResult,
    MetricStatus,
    TargetOutput,
    TestCase,
    TestRun,
    Usage,
)

runner = CliRunner()


def _write_results(path: Path, *, score: float = 0.9) -> Path:
    test_run = TestRun(suite_name="s", run_id="r1")
    test_run.results.append(
        CaseResult(
            case_id="c1",
            input="q",
            status=CaseStatus.PASSED,
            actual_output="a",
            latency_ms=10.0,
            sut_usage=Usage(prompt_tokens=5, completion_tokens=2, cost_usd=0.001),
            judge_usage=Usage(prompt_tokens=3, cost_usd=0.0005),
            metrics=[
                MetricResult(
                    metric="faithfulness",
                    status=MetricStatus.SCORED,
                    score=score,
                    passed=score >= 0.5,
                )
            ],
        )
    )
    path.write_text(json.dumps(test_run.model_dump(mode="json"), indent=2))
    return path


# --------------------------------------------------------------------------
# init / doctor / trends / compare / report


def test_init_scaffolds(tmp_path: Path) -> None:
    out = tmp_path / "proj"
    result = runner.invoke(app, ["init", str(out)])
    assert result.exit_code == 0
    assert (out / "suite.yaml").exists()
    assert (out / "datasets" / "cases.yaml").exists()

    result = runner.invoke(app, ["init", str(out)])
    assert result.exit_code == 2  # refuses to clobber


def test_doctor_runs(tmp_path: Path) -> None:
    result = runner.invoke(app, ["doctor"])
    # exit 2 when optional extras are missing — that's a valid report
    assert result.exit_code in (0, 2)
    assert "environment" in result.output
    assert "optional extras" in result.output


def test_doctor_with_suite(tmp_path: Path) -> None:
    suite = tmp_path / "s.yaml"
    suite.write_text(
        yaml.dump(
            {
                "name": "s",
                "target": {"type": "python", "function": "demo_app:answer"},
                "datasets": [],
            }
        )
    )
    result = runner.invoke(app, ["doctor", str(suite)])
    assert result.exit_code in (0, 2)


def test_trends_empty_and_populated(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["trends"])
    assert result.exit_code == 0
    assert "no recorded runs" in result.output

    history = tmp_path / ".aitestkit" / "history.jsonl"
    history.parent.mkdir(parents=True)
    rows = [
        {
            "run_id": f"r{i}",
            "suite": "s",
            "finished_at": 1000.0 + i,
            "metric_means": {"faithfulness": 0.8 + i * 0.05},
            "metric_pass_rates": {"faithfulness": 0.9},
            "counts": {"passed": 1},
        }
        for i in range(2)
    ]
    history.write_text("\n".join(json.dumps(r) for r in rows))
    result = runner.invoke(app, ["trends"])
    assert result.exit_code == 0
    assert "faithfulness" in result.output
    assert "drift" in result.output


def test_compare_command(tmp_path: Path) -> None:
    baseline = _write_results(tmp_path / "base.json", score=0.9)
    current = _write_results(tmp_path / "cur.json", score=0.5)
    result = runner.invoke(app, ["compare", str(baseline), str(current)])
    assert result.exit_code == 1  # REGRESSION verdict -> test-failures exit
    assert "regress" in result.output.lower() or "faithfulness" in result.output

    equal = _write_results(tmp_path / "cur2.json", score=0.9)
    result = runner.invoke(app, ["compare", str(baseline), str(equal)])
    assert result.exit_code == 0


def test_report_command(tmp_path: Path) -> None:
    results = _write_results(tmp_path / "results.json")
    out = tmp_path / "reports"
    result = runner.invoke(
        app, ["report", str(results), "--out", str(out), "--report", "json,junit,html"]
    )
    assert result.exit_code == 0
    assert (out / "junit.xml").exists()
    assert any(p.suffix == ".html" for p in out.iterdir())


# --------------------------------------------------------------------------
# run edge paths


def _suite(tmp_path: Path, extra: dict[str, Any] | None = None) -> Path:
    datasets = tmp_path / "datasets"
    datasets.mkdir(exist_ok=True)
    (datasets / "cases.yaml").write_text(
        yaml.dump({"cases": [{"id": "c1", "input": "say hello"}]})
    )
    cfg: dict[str, Any] = {
        "name": "cli-suite",
        "target": {"type": "python", "function": "demo_app:answer"},
        "judge": {"provider": "fake", "model": "fake"},
        "metrics": [{"name": "contains", "substrings": ["hello"]}],
        "datasets": ["datasets/cases.yaml"],
    }
    if extra:
        cfg.update(extra)
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.dump(cfg))
    return suite


def test_run_with_allure_export(tmp_path: Path, monkeypatch: Any) -> None:
    import sys

    examples = Path(__file__).parent.parent / "examples"
    monkeypatch.syspath_prepend(str(examples))
    sys.modules.pop("demo_app", None)
    suite = _suite(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["run", str(suite), "-o", str(out), "--export", "allure"]
    )
    assert result.exit_code == 0, result.output
    assert (out / "allure-results").exists()


def test_run_missing_suite(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2


def test_run_bad_metric_params(tmp_path: Path, monkeypatch: Any) -> None:
    examples = Path(__file__).parent.parent / "examples"
    monkeypatch.syspath_prepend(str(examples))
    suite = _suite(
        tmp_path, {"metrics": [{"name": "contains", "bogus_param": 1}]}
    )
    result = runner.invoke(app, ["run", str(suite)])
    assert result.exit_code == 2
    assert "bogus_param" in result.output or "bad params" in result.output


def test_run_unknown_metric(tmp_path: Path, monkeypatch: Any) -> None:
    examples = Path(__file__).parent.parent / "examples"
    monkeypatch.syspath_prepend(str(examples))
    suite = _suite(tmp_path, {"metrics": [{"name": "no_such_metric"}]})
    result = runner.invoke(app, ["run", str(suite)])
    assert result.exit_code == 2


def test_run_command_target_requires_flag(tmp_path: Path, monkeypatch: Any) -> None:
    datasets = tmp_path / "datasets"
    datasets.mkdir(exist_ok=True)
    (datasets / "cases.yaml").write_text(
        yaml.dump({"cases": [{"id": "c", "input": "x"}]})
    )
    suite = tmp_path / "cmd.yaml"
    suite.write_text(
        yaml.dump(
            {
                "name": "cmd",
                "target": {"type": "command", "command": ["echo", "hi"]},
                "metrics": [{"name": "contains", "substrings": ["hi"]}],
                "datasets": ["datasets/cases.yaml"],
            }
        )
    )
    result = runner.invoke(app, ["run", str(suite)])
    assert result.exit_code == 2  # blocked without --allow-command-targets
    result = runner.invoke(
        app, ["run", str(suite), "--allow-command-targets"]
    )
    assert result.exit_code == 0


def test_chunk_compare_command(tmp_path: Path) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "a.txt").write_text("refunds allowed within 30 days of purchase")
    (corpus / "b.txt").write_text("unrelated text about weather patterns")
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        yaml.dump(
            {
                "queries": [
                    {"id": "q1", "query": "refund window", "relevant_ids": []}
                ]
            }
        )
    )
    result = runner.invoke(
        app,
        [
            "chunk-compare",
            str(corpus),
            str(golden),
            "--strategies",
            "fixed,hierarchical",
            "--chunk-size",
            "30",
            "--chunk-overlap",
            "10",
        ],
    )
    assert result.exit_code == 0
    assert "recall" in result.output


def test_load_corpus(tmp_path: Path) -> None:
    from aitestkit.cli import _load_corpus

    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "x.txt").write_text("hello")
    (corpus / "y.md").write_text("world")
    (corpus / "skip.bin").write_bytes(b"\x00")
    docs = _load_corpus(corpus)
    assert set(docs) == {"x", "y"}
    single = tmp_path / "one.txt"
    single.write_text("solo")
    assert _load_corpus(single) == {"one": "solo"}
    from aitestkit.exceptions import ConfigError

    with pytest.raises(ConfigError):
        _load_corpus(tmp_path / "missing")


# --------------------------------------------------------------------------
# datasets edge cases


def test_dataset_jsonl_and_errors(tmp_path: Path) -> None:
    from aitestkit.datasets import load_cases, load_conversations
    from aitestkit.exceptions import DatasetError

    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text('{"id": "a", "input": "q1"}\n{"input": "q2"}\n\n')
    cases = load_cases(jsonl)
    assert len(cases) == 2
    assert cases[1].id == "cases-1"

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{invalid}\n")
    with pytest.raises(DatasetError):
        load_cases(bad)

    with pytest.raises(DatasetError):
        load_cases(tmp_path / "nope.yaml")

    weird = tmp_path / "weird.txt"
    weird.write_text("x")
    with pytest.raises(DatasetError):
        load_cases(weird)

    malformed = tmp_path / "mal.yaml"
    malformed.write_text("- just a string\n")
    with pytest.raises(DatasetError):
        load_cases(malformed)

    convo = tmp_path / "conv.yaml"
    convo.write_text(
        yaml.dump(
            {
                "conversations": [
                    {
                        "id": "conv1",
                        "turns": [{"input": "a"}, {"input": "b"}],
                    },
                    {"turns": [{"input": "c"}]},
                ]
            }
        )
    )
    convs = load_conversations(convo)
    assert len(convs) == 2
    assert convs[1].id == "conv-conv-1"

    bad_convo = tmp_path / "badconv.yaml"
    bad_convo.write_text(yaml.dump({"conversations": "not-a-list"}))
    with pytest.raises(DatasetError):
        load_conversations(bad_convo)


# --------------------------------------------------------------------------
# deterministic metric gaps


async def test_json_schema_metric() -> None:
    from aitestkit.metrics.base import MetricContext
    from aitestkit.metrics.deterministic import JsonSchema

    metric = JsonSchema(
        schema={"type": "object", "required": ["answer"]}, threshold=1.0
    )
    ctx = MetricContext()
    case = TestCase(input="q")

    good = await metric.measure(case, _out('{"answer": 4}'), ctx)
    assert good.passed is True

    bad_schema = await metric.measure(case, _out('{"other": 1}'), ctx)
    assert bad_schema.passed is False
    assert "schema violation" in bad_schema.reason

    not_json = await metric.measure(case, _out("not json"), ctx)
    assert not_json.passed is False
    assert "not valid JSON" in not_json.reason


async def test_latency_and_cost_metrics() -> None:
    from aitestkit.metrics.base import MetricContext
    from aitestkit.metrics.deterministic import Latency, MaxCost

    ctx = MetricContext()
    case = TestCase(input="q")
    slow = TargetOutput(output="x", latency_ms=900.0)
    fast = TargetOutput(output="x", latency_ms=100.0)

    latency = Latency(max_ms=500)
    assert (await latency.measure(case, fast, ctx)).passed is True
    assert (await latency.measure(case, slow, ctx)).passed is False

    cheap = TargetOutput(output="x")
    cheap.usage.cost_usd = 0.001
    pricey = TargetOutput(output="x")
    pricey.usage.cost_usd = 1.5
    cost = MaxCost(max_usd=0.01)
    assert (await cost.measure(case, cheap, ctx)).passed is True
    assert (await cost.measure(case, pricey, ctx)).passed is False


def _out(text: str) -> TargetOutput:
    return TargetOutput(output=text)


# --------------------------------------------------------------------------
# judge internals


async def test_judge_evaluate_and_parse() -> None:
    from aitestkit.judge.judge import Judge
    from aitestkit.providers.fake import FakeChatProvider

    judge = Judge(provider=FakeChatProvider(), model="fake")
    verdict, usage = await judge.evaluate('{"instructions": "score this"}')
    assert 0.0 <= verdict.score <= 1.0
    assert isinstance(verdict.raw, dict)


async def test_judge_pairwise_position_swap() -> None:
    from aitestkit.judge.judge import Judge
    from aitestkit.providers.fake import FakeChatProvider

    judge = Judge(provider=FakeChatProvider(), model="fake")
    verdict, usage = await judge.pairwise_compare(
        "compare A={output_a} vs B={output_b}",
        output_a="answer a",
        output_b="answer b",
    )
    assert 0.0 <= verdict.score <= 1.0
    assert "position_swing" in verdict.raw


def test_judge_self_judging_warning(caplog: Any) -> None:
    import logging

    from aitestkit.judge.judge import Judge
    from aitestkit.providers.fake import FakeChatProvider

    judge = Judge(provider=FakeChatProvider(), model="m")
    with caplog.at_level(logging.WARNING):
        judge.warn_if_self_judging(sut_model="m")
        judge.warn_if_self_judging(sut_model="other")
    assert "biased" in caplog.text
