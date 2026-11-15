"""pytest integration: assert_test() helper + suite-driven collection.

Two usage styles:

1. Imperative — call ``assert_test`` inside a test::

    async def test_greeting():
        output = await target.invoke(case)
        assert_test(case, output, [ExactMatch()])

2. Declarative — files named ``*.aitestkit.yaml`` are collected as tests::

    # demo.aitestkit.yaml
    target: {type: python, function: "demo_app:answer"}
    metrics:
      - name: contains
        params: {substrings: ["hello"]}
    cases:
      - id: greeting
        input: "say hello"

Each case becomes one pytest item — failures show the metric verdicts.
Registered via the ``aitestkit`` entry point; ``pytest_collect_file`` is
the hook pytest calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.models import MetricResult, TargetOutput, TestCase


async def assert_test(
    case: TestCase,
    output: TargetOutput,
    metrics: list[BaseMetric],
    *,
    ctx: MetricContext | None = None,
) -> list[MetricResult]:
    """Evaluate metrics on a case output; raise AssertionError on failure."""
    context = ctx or MetricContext()
    results = [await metric.evaluate(case, output, context) for metric in metrics]
    failures = [
        f"{r.metric} score={r.score} reason={r.reason}"
        for r in results
        if r.passed is False
    ]
    errors = [
        f"{r.metric} error={r.reason}" for r in results if r.status.value == "error"
    ]
    problems = failures + errors
    if problems:
        raise AssertionError(
            f"aitestkit case {case.id or '<unnamed>'} failed:\n  " + "\n  ".join(problems)
        )
    return results


def pytest_collect_file(file_path: Path, parent: pytest.Collector) -> Any:
    """Collect ``*.aitestkit.yaml`` files as aitestkit test items."""
    if file_path.suffix in {".yaml", ".yml"} and file_path.name.endswith(
        ".aitestkit.yaml"
    ):
        return AitestkitFile.from_parent(parent, path=file_path)
    return None


class AitestkitFile(pytest.File):
    """A YAML file containing target + metrics + cases."""

    def collect(self):
        from aitestkit.config import TargetConfig
        from aitestkit.metrics.registry import build_metric
        from aitestkit.targets.factory import build_target

        data = yaml.safe_load(self.path.read_text()) or {}
        cases = data.get("cases", [])
        metrics = [
            build_metric(
                spec["name"],
                threshold=spec.get("threshold"),
                params=spec.get("params", {}),
            )
            for spec in data.get("metrics", [])
        ]
        target_cfg = data.get("target", {})
        target = (
            build_target(
                TargetConfig.model_validate(target_cfg),
                allow_command_targets=False,  # never from collected files
            )
            if target_cfg
            else None
        )
        for index, raw in enumerate(cases):
            case = TestCase.model_validate(
                {"id": raw.get("id", f"case-{index}"), **raw}
            )
            yield AitestkitItem.from_parent(
                self,
                name=case.id or f"case-{index}",
                case=case,
                metrics=metrics,
                target=target,
            )


class AitestkitItem(pytest.Item):
    """One case: invoke the target, evaluate its metrics."""

    def __init__(
        self,
        *,
        case: TestCase,
        metrics: list[BaseMetric],
        target: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.case = case
        self.metrics = metrics
        self.target = target

    def runtest(self) -> None:
        if self.target is None:
            pytest.skip("no target configured in aitestkit file")
        output = asyncio.run(self.target.invoke(self.case))
        asyncio.run(assert_test(self.case, output, self.metrics))

    def repr_failure(self, excinfo: Any) -> str:
        if isinstance(excinfo.value, AssertionError):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self) -> tuple[Path, int, str]:
        return self.path, 0, f"aitestkit: {self.name}"
