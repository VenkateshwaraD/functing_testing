# Reliability Testing End-to-End

How to test whether your LLM app / agent is *stable* with `aitestkit`:
flakiness detection, resilience under broken tools, guardrails for runaway
behavior, and regression/drift tracking over time. Self-contained — no
other docs required.

## How it works

```
                    ┌─ repeat: N ─────────► flaky detection → max_flaky gate
                    │
 same suite ────────┼─ fault injection ───► recovery behavior scored
 run over time      │   (tool errors, timeouts, malformed results)
                    │
                    ├─ guardrails ────────► max_steps / wall_clock aborts
                    │
                    └─ baseline + trends ─► regression gate → exit code
```

Reliability here means three separate questions, each with dedicated
machinery:

1. **Is it consistent?** — same input, same verdict? (`repeat`, `max_flaky`,
   `seed`, `self_consistency`)
2. **Is it resilient?** — what happens when tools/APIs fail? (fault
   injection, guardrails)
3. **Is it stable over time?** — did this change make it worse?
   (baselines, `compare`, `trends`, `version_tags`)

---

## Step 1 — Flakiness: run cases repeatedly

Non-determinism is the #1 LLM reliability problem. `repeat:` re-runs every
case N times; cases that flip between pass/fail are flagged **flaky** and
counted by the `max_flaky` gate.

```yaml
name: reliability-check

target: {type: python, function: "myapp:answer"}
datasets: [datasets/cases.yaml]

repeat: 3                    # every case runs 3× — flaky detection on
seed: 42                     # deterministic ordering where SUT supports it

judge:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  self_consistency: 3        # 3 judge samples, majority vote — scorer noise
                             # doesn't masquerade as product noise

metrics:
  - {name: correctness, threshold: 0.7}
  - exact_match

gates:
  min_pass_rate: 0.9
  max_flaky: 0.1             # fail if >10% of cases give inconsistent verdicts
  max_error_cases: 0         # infra errors (timeouts, 5xx) count separately
```

Also set `temperature: 0` on the SUT itself where possible, and fix the
model version — `--judge-model` and `models:` entries are pinned per suite,
so a provider-side silent upgrade shows up as a baseline diff, not mystery
drift.

**Shipped demo** — `suites/reliability/flaky_suite.yaml` points at
`apps.demo_app:flaky_answer`, a deliberately stochastic SUT (50/50
correct), with `repeat: 5`:

```bash
PYTHONPATH=examples uv run aitestkit run \
    suites/reliability/flaky_suite.yaml -o evidence/flaky_suite
```

Every case lands `flaky`, and `results.json` records the full picture:
`repeat: {runs: 5, pass_rate: 0.2, statuses: [failed, passed, …]}` —
the reproducibility evidence for stochastic LLM decisions.

## Step 2 — Resilience: inject faults into tools

Declare the agent's tools in the suite; the registry can break them on
purpose — no changes to agent code:

```yaml
tools:
  search:
    function: "myapp.tools:search"
    fault: {error: "503 upstream", times: 2}   # first 2 calls fail → is there a retry?
  slow_db:
    function: "myapp.tools:slow_db"
    fault: {timeout_ms: 500}                   # injected hang → does it time out cleanly?
  legacy:
    function: "myapp.tools:legacy"
    fault: {malformed: true}                   # garbage return → does it crash?
  billing:
    function: "myapp.tools:charge"
    requires_confirmation: true                # destructive → must get human OK
```

Fault fields: `error` (raises `TargetError`), `timeout_ms` (sleeps then
raises `TimeoutError`), `malformed` (returns a sentinel dict), `times: N`
(fire only first N calls — omit = always).

Pair with metrics that score the recovery:

```yaml
metrics:
  - name: termination_check        # clean finish_reason, not timeout/error
  - name: loop_detection
    max_repeats: 3                 # isn't retrying the same failed call forever
  - name: confirmation_required    # waited for human OK on billing
  - name: task_completion          # still finished the job despite failures
```

Guardrails bound the blast radius while you do this:

```yaml
guardrails:
  max_steps: 25                    # runaway loop → abort, exit code 4
  wall_clock_seconds: 120          # hung agent → abort, exit code 4
case_timeout_seconds: 60
max_cost_usd: 1.00                 # retry storms can't drain budget → exit 3
```

## Step 3 — Regression: baseline every known-good run

```bash
# store the good run once
aitestkit run suite.yaml --baseline-save main

# every later run diffs against it — REGRESSION verdict fails the run
aitestkit run suite.yaml --baseline .aitestkit/baselines/main.json
```

Enforce it in the suite so a diff can't be ignored:

```yaml
gates:
  max_regression_vs_baseline: 0.05   # per-metric mean drop beyond 5% → fail
```

Or compare two saved runs directly:

```bash
aitestkit compare .aitestkit/baselines/main.json out/results.json \
    --tolerance 0.05        # exits 1 on REGRESSION — scriptable gate
```

## Step 4 — Drift: watch trends over time

Every run appends to `.aitestkit/history.jsonl`. `trends` summarizes drift
per metric (first mean → last mean, Δ, run count):

```bash
aitestkit trends .
#   correctness: mean 0.870 -> 0.810 (drift -0.060, 14 runs)
#   recall_at_k: mean 0.750 -> 0.740 (drift -0.010, 14 runs)
```

Make drift *attributable* by stamping versions into run metadata:

```yaml
version_tags: {prompt: v7, kb: "2025-11-15", chunker: hierarchical-512}
```

For RAG indexes add the `embedding_drift` metric — catches silent index
rot after re-ingestion.

## Step 5 — Deterministic CI with cassettes

Reliability tests are worthless if the infra is flaky. Record provider
traffic once, replay it byte-identical forever:

```bash
aitestkit run suite.yaml --cassette cassettes/rel.jsonl --cassette-mode record
aitestkit run suite.yaml --cassette cassettes/rel.jsonl   # replay, zero keys
```

Replay mode + `fake` judge = a fully deterministic reliability gate that
can run on every commit. Provider-side retries/throttling
(`max_retries`, per-provider `rpm`/`tpm`) keep real runs stable too.

## Reference suite

```yaml
name: agent-reliability
target: {type: python, function: "myapp.agent:run"}
datasets: [datasets/agentic_cases.yaml]

repeat: 3
seed: 42
concurrency: 4
case_timeout_seconds: 60
max_cost_usd: 1.00

guardrails: {max_steps: 25, wall_clock_seconds: 120}
version_tags: {prompt: v7, agent: "react-v2"}

tools:
  search:  {function: "myapp.tools:search", fault: {error: "503", times: 1}}
  billing: {function: "myapp.tools:charge", requires_confirmation: true}

judge: {provider: openai, model: gpt-4o-mini, temperature: 0.0, self_consistency: 3}

metrics:
  - {name: task_completion, threshold: 0.7}
  - termination_check
  - {name: loop_detection, max_repeats: 3}
  - confirmation_required

gates:
  min_pass_rate: 0.9
  max_flaky: 0.1
  max_error_cases: 0
  max_regression_vs_baseline: 0.05
```

## Shipped demo — run it

`suites/banking/banking_resilience_suite.yaml` injects `fault: {error:
"core-banking 503"}` on `transfer_funds` and asserts *graceful
degradation* — fully offline:

```bash
PYTHONPATH=examples uv run aitestkit run \
    suites/banking/banking_resilience_suite.yaml -o evidence/resilience
```

```
PASS transfer-degraded   → "The payment service is unavailable — please
                            retry shortly."   (no fabricated confirmation)
PASS balance-unaffected  → "Your checking balance is $1250.00."
```

The interesting part is in `results.json`: the degraded case's trajectory
shows `action → observation(tool error: …) → message` with **no**
`tool_call` event — the registry raised before the call completed —
while `termination_check` and `loop_detection` confirm it finished
cleanly instead of retry-storming. `dataset_testcases/
banking_resilience_cases.yaml` is the template: expected output is the
*degradation message*, not the happy path.

## Validate the results

`results.json` per case: a `status` plus per-repeat attempts and the
trajectory — the fields that matter for reliability reading:

- `status: flaky` — mixed pass/fail across `repeat:` runs (counts toward
  `max_flaky`, not `min_pass_rate`)
- `status: error` — infra failure (provider 5xx, timeout, tool
  `TargetError`); never a quality failure — `max_error_cases` gates it
- `trajectory.finish_reason` — `done` vs `max_steps`/`timeout`/`error`
  tells you *which* guardrail fired
- `trajectory.events[]` — tool calls show `payload.error`, so a resilience
  case shows the injected `503` right next to the agent's recovery step

```bash
# flaky + error cases — the two reliability statuses
jq '.results[] | select(.status=="flaky" or .status=="error")
    | {case_id, status, finish: .trajectory.finish_reason}' \
    out/results.json
```

Long-run validation lives outside the run dir: `.aitestkit/history.jsonl`
accumulates per-metric means (`aitestkit trends .` summarizes drift), and
`aitestkit compare baseline.json results.json --tolerance 0.05` exits `1`
on regression — the same verdict the `max_regression_vs_baseline` gate
applies in-suite.

## Exit codes

| Code | Meaning | Reliability signal |
|------|---------|--------------------|
| 0 | all gates passed | stable |
| 1 | case failures | correctness regression |
| 2 | config error | bad YAML / missing env key |
| 3 | `max_cost_usd` hit | retry storm / spend runaway |
| 4 | guardrail abort | hung or looping agent |
| 5 | gate failure | flaky / pass-rate / regression gate tripped |

CI pipelines fail the job on any non-zero code — a nightly schedule with
`repeat: 3` + baseline gate gives you a standing reliability monitor.
