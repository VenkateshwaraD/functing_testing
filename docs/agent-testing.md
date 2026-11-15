# Testing AI Agents End-to-End

How to test an agentic system with `aitestkit`: instrument the agent so it
reports a **trajectory**, declare expected behavior in a dataset, score it
with agentic metrics, gate the run, and read the reports.

## How it works

```
┌────────────┐   trace events    ┌──────────────┐   per-case   ┌─────────┐
│ your agent │ ────────────────► │  Trajectory  │ ───────────► │ metrics │
│ (the SUT)  │  thought, action, │ (ordered     │  12 agentic  │ pass/   │
│            │  tool_call,       │  event log)  │  scorers     │ fail    │
│            │  handoff, message │              │              │         │
└────────────┘                   └──────────────┘              └────┬────┘
      ▲ tools: registry — real fns, mocks,                          │
      │        fault injection, confirmation flags                  ▼
      └──────────────────────────────────────────        gates → exit code
```

The agent emits semantic events through a `trace_sink` the framework passes
in. Events become a schema-versioned `Trajectory` (in `results.json`), which
is what every agentic metric scores — so the metrics work with **any** agent
framework, not just LangGraph.

Event types: `thought`, `action`, `observation`, `tool_call`, `llm_call`,
`message`, `handoff`, `memory_op`, `retrieval`.

---

## Step 1 — Instrument your agent

Pick one of three ways, depending on what your agent is.

### Option A — Python function (most common)

Target `type: python`. Your callable receives `trace_sink` (a
`TraceEmitter`) and calls it per step:

```python
# myapp/agent.py
def run(input: str, *, trace_sink=None, tool_registry=None) -> dict:
    trace_sink.thought(f"planning how to handle: {input}")

    result = tool_registry.invoke("search", {"query": input})   # recorded
    trace_sink.tool_call("search", args={"query": input}, result=result)
    trace_sink.emit("observation", text=result)

    answer = f"found: {result}"
    trace_sink.message("assistant", answer)
    trace_sink.finish("done")
    return {"output": answer}
```

`trace_sink` API — all optional; emit what your agent actually does:

```python
sink.thought("reasoning text")
sink("action", text="use tool: search")          # generic emit
sink.tool_call("search", args={...}, result=..., error=...)
sink.llm_call("gpt-4o-mini")
sink.message("assistant", "final answer")
sink.handoff("billing-agent", from_agent="router")   # multi-agent
sink.memory_op("write", key="user_prefs", value={...})
sink.retrieval(["chunk1", "chunk2"])
sink.emit("observation", text="...")
sink.finish("done")   # done|max_steps|timeout|error|cancelled
```

The callable only gets the kwargs it declares (`input`, `context`,
`tool_registry`, `tool_mocks`, `session_id`, `prompt`, `model`, `provider`)
— inspect `python_target.py` for the full list. If your agent can't take a
sink, return `tool_calls` in the result dict instead and metrics still get
the call list.

### Option B — LangGraph graph (zero instrumentation)

Target `type: langgraph` (install `aitestkit[langgraph]`). Point `function`
at a **compiled graph** (or a zero-arg factory returning one); the adapter
streams `astream_events` and builds the trajectory automatically:

```yaml
target:
  type: langgraph
  function: "myapp.graph:compiled_graph"   # needs .astream_events
  kwargs: {input_key: "input"}             # key for string inputs
```

### Option C — Agent behind HTTP

Target `type: http` — extract the answer and tool calls from the JSON
response:

```yaml
target:
  type: http
  url: "https://agent.internal/v1/run"
  method: POST
  headers: {Authorization: "Bearer ${AGENT_API_KEY}"}
  request_template: {task: "{input}"}
  response_path: "result.text"
  tools_path:   "result.tool_calls"   # list of {name, arguments}
  context_path: "result.retrieved"
```

---

## Step 2 — Write the dataset

`datasets/agentic_cases.yaml` — declare what *should* happen, not just the
final answer:

```yaml
cases:
  - id: weather-query
    input: "what's the weather in Paris"
    expected_tools:                          # scored by tool_sequence /
      - {name: search, arguments: {query: "weather"}}   # tool_args_correct
    tags: [agentic, tools]

  - id: no-tool-needed
    input: "say hi"
    expected_tools: []                       # must NOT call tools
    tags: [agentic]

  - id: handoff-flow
    input: "refund my order"
    metadata:
      expected_handoffs: ["billing-agent"]   # scored by handoff_correctness
    tags: [agentic, multi-agent]

  - id: replay-check
    input: "book the flight"
    expected_trajectory:                     # scored by trajectory_match —
      events:                                # full step-by-step golden trace
        - {type: thought,   payload: {text: "plan booking"}}
        - {type: tool_call, payload: {name: search}}
        - {type: message,   payload: {role: assistant}}
```

Per-case fields the metrics read: `expected_tools` (list of
`{name, arguments}`), `expected_trajectory`, `metadata.expected_handoffs`,
plus the usual `expected_output`, `context`, `tags`.

---

## Step 3 — Configure the suite

```yaml
name: agent-regression

target:
  type: python
  function: "myapp.agent:run"

datasets:
  - datasets/agentic_cases.yaml

# Tools the agent may call — loaded once, injected per run ---------------
tools:
  search:
    function: "myapp.tools:search"
  refund_order:
    function: "myapp.tools:refund"
    requires_confirmation: true        # confirmation_required metric checks this
  flaky_api:
    function: "myapp.tools:flaky"
    fault: {error: "503 upstream", times: 2}   # fails first 2 calls → tests retry
  slow_api:
    function: "myapp.tools:slow"
    fault: {timeout_ms: 500}           # injected timeout
  legacy_api:
    function: "myapp.tools:legacy"
    fault: {malformed: true}           # returns garbage shape
  expensive_api:
    function: "myapp.tools:expensive"
    mock: {status: ok}                 # canned result — zero cost

# Judge for subjective metrics -------------------------------------------
judge:
  provider: openai                     # or fake for fully-offline dev
  model: gpt-4o-mini

# Metrics — what to score -------------------------------------------------
metrics:
  - name: tool_sequence                # order + set vs expected_tools
  - name: tool_args_correct            # argument values match expected
  - name: forbidden_tools
    tools: ["delete", "drop"]          # params live under the entry
  - name: confirmation_required        # destructive tools got human OK
  - name: loop_detection
    max_repeats: 3                     # identical step signature repeats
  - name: termination_check            # finish_reason is a clean stop
  - name: trajectory_match             # exact match vs expected_trajectory
  - name: task_completion              # judge: did it achieve the goal
    threshold: 0.7
  - name: reasoning_coherence          # judge: is the thought chain sound
  - name: plan_validity                # judge: plan precedes + matches acts

# Guardrails — hard aborts for runaway agents ----------------------------
guardrails:
  max_steps: 25                        # breach → finish_reason: max_steps
  wall_clock_seconds: 120              # breach → timeout

# Runtime -----------------------------------------------------------------
concurrency: 4
case_timeout_seconds: 60
repeat: 2                              # catches non-deterministic agents
max_cost_usd: 1.00                     # spend cap → exit 3

# Pass/fail bar -------------------------------------------------------------
gates:
  min_pass_rate: 0.9
  max_error_cases: 0
  max_flaky: 0.1                       # with repeat: 2
  per_metric:
    task_completion: {min_mean: 0.75}
```

### Safety notes

- `requires_confirmation` tools model **human-approval gates** — the agent
  must emit a `message` event with `role: confirmation` before calling them,
  or `confirmation_required` fails.
- `fault:` fields: `error` (raises), `timeout_ms` (hangs then raises),
  `malformed` (returns a garbage dict), `times: N` (only first N calls,
  omit = always). Lets you test retry/recovery without touching agent code.
- `mock:` forces a canned return value — cheap deterministic runs.
- Every registry `invoke` is recorded as a `ToolCall` (name, args, result,
  error) into the trajectory, even failed calls.

---

## Step 4 — Run it

```bash
# offline dev loop — fake judge, cassettes for any provider calls
aitestkit run agent_suite.yaml --judge fake --judge-model fake -o out/

# real run
export OPENAI_API_KEY=...
aitestkit run agent_suite.yaml -o out/

# debug a flaky agent — verbose logging (set repeat: 3 in the suite too)
aitestkit run agent_suite.yaml -o out/ -v

# interrupted run? resume skips finished cases
aitestkit run agent_suite.yaml -o out/ --resume out/results.jsonl

# record provider traffic once, replay forever in CI
aitestkit run agent_suite.yaml --cassette cassettes/agent.jsonl --cassette-mode record
aitestkit run agent_suite.yaml --cassette cassettes/agent.jsonl   # replay = offline
```

If the suite has `command:` targets add `--allow-command-targets`; strip
python targets in locked-down CI with `--no-allow-python-targets`.

---

## Step 5 — Read the results

Output dir (`-o`, default `scratch/`):

```
out/
├── results.json              # full run: per-case scores + trajectories
├── results.jsonl             # one line per case (resumable / streamable)
├── aitestkit-report.html     # browsable report
├── junit.xml                 # with --report junit — for CI test UIs
└── console output            # pass/fail table + gate verdicts
```

Each case in `results.json` carries its `trajectory` — the full ordered
event list (thoughts, tool calls with args/results/errors, handoffs,
finish_reason) — plus per-metric scores, judge rationales, latency, and
cost. That's your debugging surface: filter by failed metric, open the
trajectory, see exactly which step diverged.

```bash
# regenerate formats from a saved run
aitestkit report out/results.json --report html,junit

# compare against a stored baseline (fails on REGRESSION)
aitestkit run agent_suite.yaml --baseline-save main     # store once
aitestkit run agent_suite.yaml --baseline .aitestkit/baselines/main.json

# watch drift across runs
aitestkit trends .
```

### Exit codes (gate verdicts)

| Code | Meaning |
|------|---------|
| 0 | all gates passed |
| 1 | case failures |
| 2 | config error (bad YAML, missing env key, unregistered tool) |
| 3 | `max_cost_usd` spend cap hit |
| 4 | guardrail abort (`max_steps` / `wall_clock_seconds`) |
| 5 | gate failure (`min_pass_rate`, `per_metric`, coverage…) |

### In CI

Gate the pipeline on the exit code — the GitLab `check` job runs every
`suites/*.yaml` the same way. Keep a cassette +
`fake` judge for deterministic CI, and run real-LLM agent suites only on a
schedule with a `max_cost_usd` cap like `live-smoke` does.

---

## Worked example

The repo ships a runnable one — a keyword-driven demo "ReAct" agent
(`examples/dummy/agent_app.py`) that thinks, calls a `search` tool, observes, and
answers:

```bash
cd examples
export PYTHONPATH=.
uv run aitestkit run suites/agentic_suite.yaml -o /tmp/agentic-out
```

It scores `tool_sequence`, `forbidden_tools`, `loop_detection`,
`termination_check`, `task_completion` against `datasets/agentic_cases.yaml`
fully offline (fake judge) — a good template to copy before wiring up your
own agent.
