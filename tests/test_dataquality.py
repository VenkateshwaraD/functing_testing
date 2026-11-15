"""Data-quality pillar tests: expectations, metric, GX bridge error path."""

from __future__ import annotations

import pytest

from aitestkit.dataquality.expectations import check_expectations
from aitestkit.dataquality.metric import DataQuality
from aitestkit.exceptions import ConfigError
from aitestkit.metrics.base import MetricContext
from aitestkit.models import MetricStatus, TargetOutput, TestCase

ROWS = [
    {"id": "a", "email": "x@y.com", "age": 30, "status": "active"},
    {"id": "b", "email": "not-an-email", "age": 200, "status": "weird"},
    {"id": "a", "email": None, "age": 25, "status": "active"},
]


def test_expectations_report() -> None:
    report = check_expectations(
        ROWS,
        [
            {"type": "not_null", "column": "email"},
            {"type": "unique", "column": "id"},
            {"type": "in_range", "column": "age", "min": 0, "max": 120},
            {"type": "in_set", "column": "status", "values": ["active", "idle"]},
            {"type": "row_count", "min": 1},
        ],
    )
    assert report["total"] == 5
    assert report["passed"] == 1  # only row_count passes
    by_name = {r["expectation"]: r for r in report["results"]}
    assert not by_name["not_null(email)"]["passed"]
    assert not by_name["unique(id)"]["passed"]
    assert not by_name["in_range(age)"]["passed"]
    assert not by_name["in_set(status)"]["passed"]
    assert by_name["row_count(None)"]["passed"]


def test_unknown_expectation() -> None:
    report = check_expectations(ROWS, [{"type": "bogus", "column": "id"}])
    assert report["results"][0]["passed"] is False
    assert "unknown expectation" in report["results"][0]["reason"]


def test_no_pii() -> None:
    rows = [{"text": "call 111-22-3333"}, {"text": "hello"}]
    report = check_expectations(
        rows, [{"type": "no_pii", "column": "text"}]
    )
    assert report["results"][0]["passed"] is False


async def test_data_quality_metric() -> None:
    case = TestCase(
        id="t",
        input="q",
        metadata={
            "rows": ROWS,
            "expectations": [{"type": "row_count", "min": 2}],
        },
    )
    ok = await DataQuality().measure(case, TargetOutput(), MetricContext())
    assert ok.passed is True
    assert ok.details["backend"] == "builtin"

    failing = TestCase(
        id="t",
        input="q",
        metadata={
            "rows": ROWS,
            "expectations": [{"type": "not_null", "column": "email"}],
        },
    )
    bad = await DataQuality().measure(failing, TargetOutput(), MetricContext())
    assert bad.passed is False
    assert "not_null(email)" in bad.reason


async def test_data_quality_skips() -> None:
    empty = TestCase(id="t", input="q", metadata={"rows": []})
    result = await DataQuality().measure(empty, TargetOutput(), MetricContext())
    assert result.status == MetricStatus.SKIPPED


def test_gx_bridge_requires_extra() -> None:
    try:
        import great_expectations  # noqa: F401

        pytest.skip("great_expectations installed")
    except ImportError:
        pass
    from aitestkit.dataquality.gx_bridge import run_gx_validation

    with pytest.raises(ConfigError, match="aitestkit\\[gx\\]"):
        run_gx_validation(ROWS, [{"type": "not_null", "column": "email"}])
