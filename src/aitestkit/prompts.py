"""Prompt simulation: render suite prompts without calling the SUT.

Snapshots the fully rendered prompt per case/variant so prompt edits can be
diffed (``prompt regression``) and linted deterministically before any model
tokens are spent.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from aitestkit.models import TestCase

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def render_prompt(template: str, case: TestCase) -> str:
    """Render a template against a case.

    ``{input}`` and any ``{metadata_key}`` placeholders resolve; unknown
    placeholders stay literal so prompt_sim can flag them.
    """
    values: dict[str, Any] = {"input": case.input, **case.metadata}

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(values[key]) if key in values else match.group(0)

    return _PLACEHOLDER_RE.sub(_sub, template)


def unresolved_placeholders(rendered: str) -> list[str]:
    """``{name}`` tokens left after rendering — a template bug."""
    return sorted(set(_PLACEHOLDER_RE.findall(rendered)))


def snapshot_prompts(
    cases: list[TestCase],
    prompts: list[Any],
) -> dict[str, dict[str, str]]:
    """case_id -> {variant_name: rendered_prompt} for every matrix cell."""
    snapshot: dict[str, dict[str, str]] = {}
    for case in cases:
        case_id = case.id or "case"
        row: dict[str, str] = {}
        for variant in prompts:
            row[getattr(variant, "name", "prompt")] = render_prompt(
                variant.template, case
            )
        snapshot[case_id] = row
    return snapshot


def diff_prompt_snapshots(
    old: dict[str, dict[str, str]], new: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Compare two prompt snapshots; changed cells are regression risk."""
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed: dict[str, list[str]] = {}
    for case_id in set(old) & set(new):
        variants_changed = [
            name
            for name, rendered in new[case_id].items()
            if old[case_id].get(name) != rendered
        ]
        if variants_changed:
            changed[case_id] = sorted(variants_changed)
    return {
        "added_cases": added,
        "removed_cases": removed,
        "changed_prompts": changed,
        "changed_count": sum(len(v) for v in changed.values()),
    }


def save_prompt_snapshot(
    snapshot: dict[str, dict[str, str]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "digest": hashlib.sha256(
            json.dumps(snapshot, sort_keys=True).encode()
        ).hexdigest()[:16],
        "prompts": snapshot,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_prompt_snapshot(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text())
    if data.get("schema_version", 1) != 1:
        raise ValueError(f"unsupported prompt snapshot version in {path}")
    prompts: dict[str, dict[str, str]] = data["prompts"]
    return prompts
