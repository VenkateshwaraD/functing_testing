"""Great Expectations bridge — requires ``aitestkit[gx]``.

Translates our expectation spec to a GX validation over an in-memory
batch so suites can reuse existing GX assets without a full deployment.
Lazy-imported: GX is a heavy dependency and stays optional.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from aitestkit.dataquality.expectations import ExpectationReport
from aitestkit.exceptions import ConfigError

_GX_TYPE_MAP = {
    "not_null": "expect_column_values_to_not_be_null",
    "unique": "expect_column_values_to_be_unique",
    "matches_regex": "expect_column_values_to_match_regex",
    "in_set": "expect_column_values_to_be_in_set",
    "in_range": "expect_column_values_to_be_between",
}


def gx_available() -> bool:
    return importlib.util.find_spec("great_expectations") is not None


def run_gx_validation(
    rows: list[dict[str, Any]], expectations: list[dict[str, Any]]
) -> ExpectationReport:
    """Validate rows through GX; returns the same report shape as the
    built-in checkers so reports stay uniform."""
    if not gx_available():
        raise ConfigError(
            "GX bridge requires: pip install 'aitestkit[gx]'"
        )
    import great_expectations as gx
    import pandas as pd

    frame = pd.DataFrame(rows)
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(name="aitestkit")
    asset = datasource.add_dataframe_asset(name="rows")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": frame})

    results = []
    for spec in expectations:
        kind = str(spec.get("type", ""))
        column = spec.get("column")
        gx_name = _GX_TYPE_MAP.get(kind)
        if gx_name is None:
            results.append({
                "expectation": str(spec),
                "passed": False,
                "reason": f"no GX mapping for {kind!r} — use builtin checker",
            })
            continue
        kwargs: dict[str, Any] = {"column": column}
        if kind == "matches_regex":
            kwargs["regex"] = spec.get("pattern", ".*")
        elif kind == "in_set":
            kwargs["value_set"] = spec.get("values", [])
        elif kind == "in_range":
            if spec.get("min") is not None:
                kwargs["min_value"] = spec["min"]
            if spec.get("max") is not None:
                kwargs["max_value"] = spec["max"]
        expectation_cls = getattr(gx.expectations, _class_name(gx_name), None)
        if expectation_cls is None:
            results.append({
                "expectation": str(spec),
                "passed": False,
                "reason": f"GX class for {gx_name!r} not found in this version",
            })
            continue
        try:
            result = batch.validate(expectation_cls(**kwargs))
            results.append({
                "expectation": f"{kind}({column})",
                "passed": bool(result.success),
                "reason": "" if result.success else str(
                    result.result.get("unexpected_count", "see gx_result")
                ),
                "gx_result": result.to_json_dict()
                if hasattr(result, "to_json_dict")
                else {},
            })
        except Exception as e:  # noqa: BLE001 — isolate per-expectation GX failures
            results.append({
                "expectation": f"{kind}({column})",
                "passed": False,
                "reason": f"GX validation error: {e}",
            })
    passed = sum(1 for r in results if r["passed"])
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "backend": "great_expectations",
        "results": results,
    }


def _class_name(expectation_name: str) -> str:
    """expect_column_values_to_not_be_null -> ExpectColumnValuesToNotBeNull."""
    return "".join(part.capitalize() for part in expectation_name.split("_"))
