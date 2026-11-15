"""Column expectations over row data — the data-quality layer.

Cases supply rows in ``metadata['rows']`` (list of dicts) and an
``expectations`` spec. Each expectation is ``{type, column, ...params}``;
this module evaluates them deterministically. The GX bridge translates the
same spec to Great Expectations when the extra is installed.
"""

from __future__ import annotations

import re
from typing import Any

ExpectationReport = dict[str, Any]


def check_expectations(
    rows: list[dict[str, Any]], expectations: list[dict[str, Any]]
) -> ExpectationReport:
    """Evaluate every expectation; returns per-expectation pass/fail."""
    results = []
    for spec in expectations:
        results.append(_check_one(rows, spec))
    passed = sum(1 for r in results if r["passed"])
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }


def _check_one(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec.get("type", "")
    column = spec.get("column")
    checker = _CHECKERS.get(str(kind))
    if checker is None:
        return {
            "expectation": str(spec),
            "passed": False,
            "reason": f"unknown expectation type {kind!r}",
        }
    try:
        passed, detail = checker(rows, column, spec)
    except Exception as e:  # noqa: BLE001 — one bad spec must not kill the batch
        return {"expectation": str(spec), "passed": False, "reason": str(e)}
    return {
        "expectation": f"{kind}({column})",
        "passed": passed,
        "reason": detail,
    }


def _values(rows: list[dict[str, Any]], column: Any) -> list[Any]:
    if column is None:
        raise ValueError("expectation requires 'column'")
    return [row.get(str(column)) for row in rows]


def _not_null(rows: list[dict[str, Any]], column: Any, spec: dict[str, Any]) -> tuple[bool, str]:
    values = _values(rows, column)
    nulls = sum(1 for v in values if v is None or v == "")
    return nulls == 0, f"{nulls} null of {len(values)}"


def _unique(rows: list[dict[str, Any]], column: Any, spec: dict[str, Any]) -> tuple[bool, str]:
    values = _values(rows, column)
    dupes = len(values) - len(set(map(str, values)))
    return dupes == 0, f"{dupes} duplicates of {len(values)}"


def _matches_regex(
    rows: list[dict[str, Any]], column: Any, spec: dict[str, Any]
) -> tuple[bool, str]:
    pattern = spec.get("pattern", ".*")
    values = _values(rows, column)
    bad = [v for v in values if v is not None and not re.fullmatch(pattern, str(v))]
    return not bad, f"{len(bad)} non-matching of {len(values)}"


def _in_range(rows: list[dict[str, Any]], column: Any, spec: dict[str, Any]) -> tuple[bool, str]:
    lo, hi = spec.get("min"), spec.get("max")
    values = [v for v in _values(rows, column) if v is not None]
    out = [
        v for v in values
        if (lo is not None and v < lo) or (hi is not None and v > hi)
    ]
    return not out, f"{len(out)} out of [{lo}, {hi}] of {len(values)}"


def _in_set(rows: list[dict[str, Any]], column: Any, spec: dict[str, Any]) -> tuple[bool, str]:
    allowed = set(map(str, spec.get("values", [])))
    values = _values(rows, column)
    bad = [v for v in values if str(v) not in allowed]
    return not bad, f"{len(bad)} outside allowed set of {len(values)}"


def _row_count(rows: list[dict[str, Any]], column: Any, spec: dict[str, Any]) -> tuple[bool, str]:
    minimum = spec.get("min", 0)
    maximum = spec.get("max")
    count = len(rows)
    ok = count >= minimum and (maximum is None or count <= maximum)
    return ok, f"{count} rows (min {minimum}, max {maximum})"


def _no_pii(rows: list[dict[str, Any]], column: Any, spec: dict[str, Any]) -> tuple[bool, str]:
    patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",  # email
        r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # card-ish
    ]
    values = _values(rows, column)
    hits = [
        i for i, v in enumerate(values)
        if v is not None and any(re.search(p, str(v)) for p in patterns)
    ]
    return not hits, f"PII-shaped values in {len(hits)} rows"


_CHECKERS = {
    "not_null": _not_null,
    "unique": _unique,
    "matches_regex": _matches_regex,
    "in_range": _in_range,
    "in_set": _in_set,
    "row_count": _row_count,
    "no_pii": _no_pii,
}
