"""Single-file HTML report: all CSS/JS inlined, data embedded as JSON.

Opens offline, attaches to CI artifacts. Shows per-case cards with
collapsible metric evidence, trajectory timelines, handoff trees, a
golden-vs-actual diff view, cost/usage tables, and the coverage matrix.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from aitestkit.models import CaseResult, MetricResult, TestRun, Trajectory
from aitestkit.redact import Redactor
from aitestkit.stats import aggregate_metrics, flakiness_report, pass_rates_by_tag

_CSS = """
:root { --pass:#1a7f37; --fail:#cf222e; --err:#bf8700; --skip:#57606a;
  --flaky:#9a6700; --bg:#f6f8fa; --card:#fff; --border:#d0d7de; }
* { box-sizing: border-box; }
body { font: 14px/1.5 -apple-system, "Segoe UI", Roboto, monospace;
  margin: 0; background: var(--bg); color: #1f2328; }
header { background:#0d1117; color:#fff; padding:20px 28px; }
header h1 { margin:0 0 4px; font-size:20px; }
header .meta { opacity:.75; font-size:12px; }
main { max-width: 1100px; margin: 0 auto; padding: 20px; }
section { margin-bottom: 28px; }
h2 { font-size: 16px; border-bottom:1px solid var(--border); padding-bottom:6px; }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { border:1px solid var(--border); padding:6px 10px; text-align:left;
  font-size: 13px; }
th { background:#f0f3f6; }
.badge { display:inline-block; padding:1px 8px; border-radius:10px;
  color:#fff; font-size:12px; font-weight:600; }
.b-passed{background:var(--pass)} .b-failed{background:var(--fail)}
.b-error{background:var(--err)} .b-skipped{background:var(--skip)}
.b-flaky{background:var(--flaky)}
.b-REGRESSION{background:var(--fail)} .b-IMPROVEMENT{background:var(--pass)}
.b-NEUTRAL{background:var(--skip)}
.card { background:var(--card); border:1px solid var(--border);
  border-radius:6px; margin-bottom:10px; }
.card > summary { padding:10px 14px; cursor:pointer; font-weight:600;
  list-style:none; display:flex; gap:10px; align-items:center; }
.card > summary::before { content:"▸"; }
.card[open] > summary::before { content:"▾"; }
.card .body { padding: 0 14px 12px; border-top:1px solid var(--border); }
.mono { font-family: ui-monospace, SFMono-Regular, monospace; font-size:12px;
  white-space: pre-wrap; word-break: break-word; }
.timeline { border-left:3px solid var(--border); margin-left:8px; }
.tl-step { position:relative; padding:4px 0 4px 16px; }
.tl-step::before { content:""; position:absolute; left:-7px; top:9px;
  width:9px; height:9px; border-radius:50%; background:#8c959f; }
.tl-step.t-tool_call::before{background:#0969da}
.tl-step.t-thought::before{background:#8250df}
.tl-step.t-handoff::before{background:#bf8700}
.tl-step.t-error::before{background:var(--fail)}
.evidence { background:#f6f8fa; border:1px solid var(--border);
  border-radius:4px; padding:8px; margin-top:6px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.attribution { font-weight:600; color:#8250df; }
"""

_JS = """
function toggleAll(open) {
  document.querySelectorAll('details.card').forEach(d => d.open = open);
}
function filterCases(status) {
  document.querySelectorAll('details.card').forEach(d => {
    d.style.display = (!status || d.dataset.status === status) ? '' : 'none';
  });
}
"""


def write_html(test_run: TestRun, path: Path, *, redactor: Redactor) -> None:
    """Render a TestRun into a single self-contained HTML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _document(test_run, redactor)
    path.write_text(redactor.redact_text(doc), encoding="utf-8")


def _document(test_run: TestRun, redactor: Redactor) -> str:
    data = redactor.redact(test_run.model_dump(mode="json"))
    data_json = json.dumps(data, default=str).replace("</", "<\\/")
    body = "".join(
        [
            _header(test_run),
            _gates_section(test_run),
            _metrics_section(test_run),
            _cost_section(test_run),
            _coverage_section(test_run),
            _flakiness_section(test_run),
            _cases_section(test_run),
        ]
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>aitestkit — {_esc(test_run.suite_name)}</title>"
        f"<style>{_CSS}</style></head><body>{body}"
        f"<script type='application/json' id='run-data'>{data_json}</script>"
        f"<script>{_JS}</script></body></html>"
    )


def _esc(text: object) -> str:
    return html.escape(str(text))


def _header(test_run: TestRun) -> str:
    meta = test_run.metadata
    counts: dict[str, int] = {}
    for result in test_run.results:
        counts[result.status.value] = counts.get(result.status.value, 0) + 1
    badges = " ".join(
        f"<span class='badge b-{status}'>{status}: {count}</span>"
        for status, count in sorted(counts.items())
    )
    sut_model = meta.model_versions.get("model")
    meta_bits = [
        f"model under test: {_esc(sut_model or 'unset')}",
        f"judge model: {_esc(meta.judge or 'none')}",
        f"execution region: {_esc(meta.region or 'unset')}",
        "execution time: "
        + _esc(datetime.fromtimestamp(meta.started_at).strftime("%Y-%m-%d %H:%M:%S")),
        f"execution platform: {_esc(meta.platform.split('-', 1)[0] or '?')}",
        f"run {_esc(test_run.run_id)}",
        f"git {_esc(meta.git_sha or '?')}",
        f"aitestkit {_esc(meta.aitestkit_version)}",
        f"python {_esc(meta.python_version)}",
    ]
    if meta.embedder:
        meta_bits.append(f"embedder {_esc(meta.embedder)}")
    for key, value in sorted(meta.model_versions.items()):
        if key != "model":
            meta_bits.append(f"{_esc(key)} {_esc(value)}")
    if meta.seed is not None:
        meta_bits.append(f"seed {_esc(meta.seed)}")
    if meta.dependency_versions:
        deps = " ".join(
            f"{name}=={ver}"
            for name, ver in sorted(meta.dependency_versions.items())
            if name != "aitestkit"
        )
        if deps:
            meta_bits.append(f"deps {_esc(deps)}")
    meta_bits.append(f"config {_esc(meta.config_hash or '?')}")
    return (
        f"<header><h1>aitestkit report — {_esc(test_run.suite_name)}</h1>"
        f"<div class='meta'>{' · '.join(meta_bits)}</div>"
        f"<div style='margin-top:8px'>{badges}</div></header><main>"
        "<div style='margin-bottom:12px'>"
        "<button onclick='toggleAll(true)'>expand all</button> "
        "<button onclick='toggleAll(false)'>collapse all</button> "
        "filter: <select onchange='filterCases(this.value)'>"
        "<option value=''>all</option><option>passed</option><option>failed</option>"
        "<option>error</option><option>flaky</option><option>skipped</option>"
        "</select></div>"
    )


def _gates_section(test_run: TestRun) -> str:
    if not test_run.gates:
        return ""
    rows = "".join(
        f"<tr><td><span class='badge b-{'passed' if g.passed else 'failed'}'>"
        f"{'PASS' if g.passed else 'FAIL'}</span></td>"
        f"<td>{_esc(g.name)}</td><td>{_esc(g.actual)}</td>"
        f"<td>{_esc(g.expected)}</td><td>{_esc(g.detail)}</td></tr>"
        for g in test_run.gates
    )
    return (
        "<section><h2>Quality gates</h2><table>"
        "<tr><th></th><th>gate</th><th>actual</th><th>expected</th><th>detail</th></tr>"
        f"{rows}</table></section>"
    )


def _metrics_section(test_run: TestRun) -> str:
    summary = aggregate_metrics(test_run)
    if not summary:
        return ""
    rows = "".join(
        f"<tr><td class='mono'>{_esc(name)}</td><td>{s['n']}</td>"
        f"<td>{s['mean']:.3f}</td><td>{s['median']:.3f}</td>"
        f"<td>{s['p10']:.3f}</td><td>{s['p90']:.3f}</td>"
        f"<td>{s['pass_rate']:.0%}</td>"
        f"<td>{s['statuses'].get('error',0)} err / {s['statuses'].get('skipped',0)} skip</td></tr>"
        for name, s in sorted(summary.items())
    )
    return (
        "<section><h2>Metric summary</h2><table>"
        "<tr><th>metric</th><th>n</th><th>mean</th><th>median</th>"
        "<th>p10</th><th>p90</th><th>pass</th><th>non-scored</th></tr>"
        f"{rows}</table></section>"
    )


def _cost_section(test_run: TestRun) -> str:
    judge_tokens = sum(
        r.judge_usage.prompt_tokens + r.judge_usage.completion_tokens
        for r in test_run.results
    )
    sut_tokens = sum(
        r.sut_usage.prompt_tokens + r.sut_usage.completion_tokens
        for r in test_run.results
    )
    judge_cost = sum(r.judge_usage.cost_usd for r in test_run.results)
    sut_cost = sum(r.sut_usage.cost_usd for r in test_run.results)
    total_latency = sum(r.latency_ms for r in test_run.results)
    by_provider: dict[str, float] = {}
    for result in test_run.results:
        for usage in (result.judge_usage, result.sut_usage):
            if usage.provider:
                by_provider[usage.provider] = (
                    by_provider.get(usage.provider, 0.0) + usage.cost_usd
                )
    provider_rows = "".join(
        f"<tr><td>{_esc(p)}</td><td>${c:.4f}</td></tr>"
        for p, c in sorted(by_provider.items())
    )
    return (
        "<section><h2>Cost & latency</h2><div class='grid'><table>"
        "<tr><th>scope</th><th>tokens</th><th>cost</th></tr>"
        f"<tr><td>system under test</td><td>{sut_tokens}</td><td>${sut_cost:.4f}</td></tr>"
        f"<tr><td>judge</td><td>{judge_tokens}</td><td>${judge_cost:.4f}</td></tr>"
        f"<tr><td>total latency</td><td colspan='2'>{total_latency:.0f} ms</td></tr>"
        f"</table><table><tr><th>provider</th><th>cost</th></tr>{provider_rows}</table>"
        "</div></section>"
    )


def _coverage_section(test_run: TestRun) -> str:
    by_tag = pass_rates_by_tag(test_run)
    if not by_tag:
        return ""
    rows = "".join(
        f"<tr><td>{_esc(tag)}</td><td>{c['total']}</td>"
        f"<td>{c.get('passed',0)}</td><td>{c.get('failed',0)}</td>"
        f"<td>{c.get('error',0)}</td><td>{c.get('flaky',0)}</td>"
        f"<td>{c.get('skipped',0)}</td><td>{c['pass_rate']:.0%}</td></tr>"
        for tag, c in sorted(by_tag.items())
    )
    return (
        "<section><h2>Coverage by tag</h2><table>"
        "<tr><th>tag</th><th>cases</th><th>passed</th><th>failed</th>"
        "<th>error</th><th>flaky</th><th>skipped</th><th>pass rate</th></tr>"
        f"{rows}</table></section>"
    )


def _flakiness_section(test_run: TestRun) -> str:
    flaky = flakiness_report(test_run)
    if not flaky:
        return ""
    rows = "".join(
        f"<tr><td class='mono'>{_esc(f['case_id'])}</td><td>{f['runs']}</td>"
        f"<td>{f['pass_rate']:.0%}</td><td>{_esc(f['status'])}</td>"
        f"<td class='mono'>{_esc(', '.join(f['statuses']))}</td></tr>"
        for f in flaky
    )
    return (
        "<section><h2>Flakiness (repeat runs)</h2><table>"
        "<tr><th>case</th><th>runs</th><th>pass rate</th><th>final</th>"
        "<th>outcomes</th></tr>"
        f"{rows}</table></section>"
    )


def _cases_section(test_run: TestRun) -> str:
    cards = "".join(_case_card(r) for r in test_run.results)
    return f"<section><h2>Cases ({len(test_run.results)})</h2>{cards}</section>"


def _case_card(result: CaseResult) -> str:
    metrics_rows = "".join(_metric_row(m) for m in result.metrics)
    trajectory_html = _trajectory_html(result.trajectory) if result.trajectory else ""
    attr = (
        f"<div>attribution: <span class='attribution'>{_esc(result.attribution)}</span></div>"
        if result.attribution
        else ""
    )
    error = (
        f"<div class='evidence'><b>error:</b> {_esc(result.error)}</div>"
        if result.error
        else ""
    )
    repeat = ""
    if result.repeat.get("runs", 1) > 1:
        repeat = (
            f"<div>repeat: {result.repeat['runs']} runs, "
            f"{result.repeat.get('pass_rate', 0):.0%} pass rate</div>"
        )
    return (
        f"<details class='card' data-status='{result.status.value}'>"
        f"<summary><span class='badge b-{result.status.value}'>"
        f"{result.status.value.upper()}</span>"
        f"<span class='mono'>{_esc(result.case_id)}</span>"
        f"<span style='font-weight:400;color:#57606a'>{_esc(_truncate(result.input, 80))}</span>"
        f"</summary><div class='body'>"
        f"<div class='mono' style='margin:8px 0'>input: {_esc(result.input)}</div>"
        f"<div class='mono' style='margin:8px 0'>output: {_esc(result.actual_output)}</div>"
        f"{attr}{repeat}{error}"
        f"<table><tr><th>metric</th><th>score</th><th>status</th>"
        f"<th>reason</th><th>cost</th></tr>{metrics_rows}</table>"
        f"{trajectory_html}</div></details>"
    )


def _metric_row(metric: MetricResult) -> str:
    score = f"{metric.score:.3f}" if metric.score is not None else "—"
    evidence = _evidence_html(metric.details) if metric.details else ""
    return (
        f"<tr><td class='mono'>{_esc(metric.metric)}</td><td>{score}</td>"
        f"<td><span class='badge b-{_metric_badge(metric)}'>{metric.status.value}</span></td>"
        f"<td>{_esc(metric.reason)}{evidence}</td>"
        f"<td>${metric.usage.cost_usd:.5f}</td></tr>"
    )


def _metric_badge(metric: MetricResult) -> str:
    if metric.status.value == "scored":
        return "passed" if metric.passed else "failed"
    return metric.status.value


_KNOWN_EVIDENCE_KEYS = {
    "judge_rationale",
    "claims",
    "claim_verdicts",
    "tool_call_diff",
    "trajectory_diff",
}


def _evidence_html(details: dict[str, Any]) -> str:
    """Render structured metric evidence (claims, diffs, rationale)."""
    parts: list[str] = []
    if rationale := details.get("judge_rationale"):
        parts.append(f"<div><b>judge:</b> {_esc(rationale)}</div>")
    claims = details.get("claims") or details.get("claim_verdicts")
    if claims:
        items = "".join(
            f"<li>{_esc(c.get('claim', c) if isinstance(c, dict) else c)} "
            f"— {_esc(c.get('verdict','')) if isinstance(c, dict) else ''}</li>"
            for c in claims
        )
        parts.append(f"<div><b>claims:</b><ul>{items}</ul></div>")
    if diff := details.get("tool_call_diff"):
        parts.append(f"<div class='mono'><b>tool diff:</b> {_esc(json.dumps(diff))}</div>")
    if div := details.get("trajectory_diff"):
        parts.append(
            f"<div class='mono'><b>trajectory diff:</b> {_esc(json.dumps(div))}</div>"
        )
    for key, value in details.items():
        if key in _KNOWN_EVIDENCE_KEYS:
            continue
        parts.append(f"<div class='mono'><b>{_esc(key)}:</b> {_esc(value)}</div>")
    if not parts:
        return ""
    return f"<div class='evidence'>{''.join(parts)}</div>"


def _trajectory_html(trajectory: Trajectory) -> str:
    """Render a step timeline + handoff tree from a trajectory."""
    steps = []
    for event in trajectory.events:
        payload_preview = json.dumps(event.payload, default=str)[:300]
        steps.append(
            f"<div class='tl-step t-{event.type}'>"
            f"<span class='mono'>#{event.step_index} {_esc(event.type)}"
            f"{(' @' + _esc(event.agent_id)) if event.agent_id else ''}</span>"
            f"<div class='mono' style='color:#57606a'>{_esc(payload_preview)}</div></div>"
        )
    handoffs = trajectory.agent_sequence
    handoff_html = ""
    if handoffs:
        handoff_html = (
            f"<div><b>handoffs:</b> <span class='mono'>{_esc(' → '.join(handoffs))}</span></div>"
        )
    return (
        f"<div style='margin-top:10px'><b>trajectory</b> "
        f"(finish: {_esc(trajectory.finish_reason or '?')}){handoff_html}"
        f"<div class='timeline'>{''.join(steps)}</div></div>"
    )


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[: limit - 1] + "…"
