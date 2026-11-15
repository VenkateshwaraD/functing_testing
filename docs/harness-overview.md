# Test Harness Overview — Trajectory Replay, Tool Mocks, Prompt Sim

The automated harness for agentic + RAG systems: how a case flows from
dataset → instrumented target → scored trajectory, and where the three
key mechanisms (trajectory replay, tool mock/fault injection, prompt
simulation) plug in. Pointers to the detailed guides throughout.

```
TestCase ──► Target (python|http|command|custom|mcp|langgraph)
                │  trace_sink ──► Trajectory events
                │  tool_registry ──► mocks / faults / call recording
                │  cassette ──► replayed provider responses
                ▼
         TargetOutput ──► metrics ──► gates ──► reports
```

---

## 1. Agent trajectory replay

Two mechanisms, two meanings of "replay":

**Golden trajectory diff** — record once, assert forever:

```bash
# run → copy trajectory from results.json → paste into the dataset
jq '.results[0].trajectory' evidence/run/results.json
```

```yaml
- id: refund-flow
  input: "refund my order"
  expected_trajectory:            # scored by trajectory_match
    events: [{type: tool_call, payload: {name: issue_refund}}]
```

`trajectory_match` diffs step signatures (`type:agent:payload-key`)
against this golden trace — the replay-as-expectation pattern.

**Cassette replay** — deterministic reproduction:

```bash
aitestkit run suite.yaml --cassette c.jsonl --cassette-mode record
aitestkit run suite.yaml --cassette c.jsonl   # replay: zero API calls,
                                              # identical trajectory
```

The cassette replays the provider *responses*; the agent regenerates the
same trajectory from them — deterministic CI without keys.

> Not supported: time-travel replay (re-injecting a recorded trajectory
> into the agent mid-execution). `resume` skips finished cases on rerun
> but doesn't replay events.

→ details: `docs/agent-testing.md` (trajectory section), `docs/llm-testing.md` (cassettes)

## 2. Tool mock + fault injection

Suite `tools:` block wires the agent's tool surface:

```yaml
tools:
  search:
    function: "myapp.tools:search"
    mock: {results: ["doc1"]}                 # canned return — no call
  transfer_funds:
    function: "myapp.tools:transfer"
    fault: {error: "HTTP 503", times: 2}      # fails first 2 calls
    requires_confirmation: true                # agent must ask first
```

- `mock:` — deterministic canned values, zero external calls
- `fault:` — `error` / `timeout_ms` / `malformed` / `times` — resilience
  testing (retry, degrade, guard behavior)
- `requires_confirmation:` — feeds `confirmation_required` metric
- every call recorded in `ToolRegistry.calls` → `tool_sequence`,
  `tool_args_correct` evidence

Shipped demo: `suites/banking/banking_resilience_suite.yaml` — injected 503s,
degraded-path assertions, real trajectories.

→ details: `docs/reliability-testing.md` §2, `docs/agent-testing.md` tools section

## 3. Prompt simulation

`prompt_sim` lints the **rendered prompt** before tokens are spent —
no provider call at all:

```yaml
- name: prompt_sim
  template: "You are a banking assistant. {input}"
  banned: ["ignore previous", "reveal your"]   # injection probes
  max_tokens: 500
```

Catches: unresolved `{placeholders}`, banned/injection phrases,
token-budget overflow. It lints a template you declare — it does **not**
diff against the actual rendered prompt (that's what `prompts.json`
snapshots are for).

Shipped demos: `suites/a_b_testing/prompt_matrix_suite.yaml`,
`suites/prompt/prompt_adversarial_suite.yaml`.

→ details: `docs/prompt-testing.md`

## 4. The rest of the harness

| Piece | What it does |
|---|---|
| `trace_sink` / `TraceEmitter` | SUT emits `thought`/`tool_call`/`handoff`/`message` events → `Trajectory` |
| `ToolRegistry` | tools + mocks + faults + confirmation flags + call log |
| `loop_detection` metric | repeated step signatures + step caps — runaway-agent detection |
| `--resume` | continue a partial run from results.jsonl |
| `report` / `--export` | HTML, JUnit, JSON, OTel/Allure exports |
| `fake` providers | judge/embedder with no keys — fully offline harness |

## Runnable harnesses in this repo

| Suite | Demonstrates |
|---|---|
| `agentic_suite.yaml` | trajectory capture + tool metrics |
| `multi_agent_suite.yaml` | handoffs + per-agent events |
| `banking_resilience_suite.yaml` | fault injection + confirmation gates |
| `rag_suite.yaml` | retrieval/generation scoring |
| `prompt_matrix_suite.yaml` | prompt variants + `prompt_sim` |
| `mcp_suite.yaml` | MCP tool-contract harness |

## Related

- `docs/agent-testing.md` — full trajectory + tools reference
- `docs/reliability-testing.md` — faults, flaky, guardrails
- `docs/prompt-testing.md` — `prompt_sim` internals
- `docs/regression-testing.md` — cassettes + baselines
- `docs/getting-started.md` — suite anatomy, CLI flags
