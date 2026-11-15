# Getting Started with aitestkit

`aitestkit` is a functional testing framework for agentic AI, RAG pipelines,
and LLM applications. You declare a **suite** (YAML): what to run against
(target), the test data (datasets), how to score it (metrics), and what
counts as a pass (gates). The CLI executes the suite and emits reports.

---

## 1. Before you run

### Requirements

- **Python ≥ 3.13**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip

### Install

```bash
# from the repo checkout — installs deps from uv.lock into .venv
uv sync --frozen

# or as a package
uv pip install .
```

Optional provider extras — install only what your suites use:

```bash
uv sync --extra anthropic   # Claude chat provider
uv sync --extra gemini      # Gemini chat provider
uv sync --extra litellm     # azure, ollama, bedrock, groq, mistral, vertex, ...
uv sync --extra chroma --extra faiss   # vector stores
uv sync --extra deepeval --extra ragas # metric backends
```

### API keys

Copy `.env.example` to `.env` and fill in the keys your suite needs —
the framework auto-loads `.env` via `python-dotenv`:

```bash
cp .env.example .env
```

| Variable                     | Needed for                          |
|------------------------------|-------------------------------------|
| `OPENAI_API_KEY`             | `openai` provider (default judge + embedder) |
| `ANTHROPIC_API_KEY`          | `anthropic` provider                |
| `GEMINI_API_KEY`             | `gemini` provider                   |
| `AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_ENDPOINT` | `azure` (via litellm) |

Keys are validated **eagerly at suite load** — a missing key fails fast
with a `ConfigError` naming the variable. The `fake` provider needs no key,
so the `suites/` all run fully offline.

### Sanity check

```bash
aitestkit doctor   # reports python version, installed extras, key presence
```

---

## 2. Framework configuration

### Resolution order

```
CLI flag  >  suite YAML  >  environment variable  >  shipped defaults.yaml
```

- **CLI flags** — `--judge`, `--judge-model`, `--report`, `-o/--out`, etc.
- **Suite YAML** — everything under `judge:`, `embedder:`, `concurrency:`,
  `max_cost_usd:`, `redact:` … (unknown keys are config errors — all config
  models use `extra="forbid"`)
- **Env vars** — `AITESTKIT_JUDGE_PROVIDER`, `AITESTKIT_JUDGE_MODEL`,
  `AITESTKIT_EMBED_PROVIDER`, `AITESTKIT_EMBED_MODEL`,
  `AITESTKIT_MAX_COST_USD`.
- **`defaults.yaml`** — shipped inside the package
  (`src/aitestkit/defaults.yaml`): judge/embedder defaults, runner
  concurrency/timeouts, secret-redaction patterns, regression tolerance,
  and the price table used for `--max-cost-usd` budgeting.

### Suite YAML — full anatomy

```yaml
name: my-suite

# --- WHAT to run against (pick one) ---------------------------------------
target:
  type: python | http | command | langgraph | custom | mcp

  # python: a callable in your code — signature
  #   (input: str, *, tools, context, prompt) -> str | dict
  function: "myapp:answer"

  # http: a live endpoint — {input} is substituted into request_template;
  # headers support ${ENV_VAR} expansion; response_path digs out the text
  url: "https://api.openai.com/v1/chat/completions"
  method: POST
  headers: {Authorization: "Bearer ${OPENAI_API_KEY}"}
  request_template: {model: gpt-4o-mini, messages: [{role: user, content: "{input}"}]}
  response_path: "choices.0.message.content"
  # tools_path / context_path: also extract tool calls / retrieved context

  # command: an argv list (never a shell) — needs --allow-command-targets
  command: ["python", "agent.py", "--ask", "{input}"]

  # mcp: call a tool on an MCP server — command: [argv] for stdio (same
  # --allow-command-targets gate) or url: for streamable HTTP
  # kwargs: {tool: search, args_template: {query: "{input}"}}

  # langgraph: a compiled graph object
  # function: "mygraph:graph"

# --- TEST DATA -------------------------------------------------------------
datasets:
  - datasets/cases.yaml        # paths are relative to the suite file

# --- HOW to score -----------------------------------------------------------
metrics:
  - exact_match                       # string shorthand
  - {name: contains, substrings: ["4"], threshold: 0.0}
  - {name: correctness, threshold: 0.7}   # judge metric, 0–1

# --- PER-CASE assertions (optional) ------------------------------------------
# In datasets/*.yaml a case may pin its own metrics with `assert:` —
# overrides the suite list for that case only:
#   - {id: cheap, input: "...", assert: [exact_match]}
#   - {id: judged, input: "...", assert: [{name: correctness, threshold: 0.8}]}
# Unknown names fail at load; a case asserting a judge metric still gets a
# judge even if the suite's metric list is all-deterministic.

# --- WHO judges -------------------------------------------------------------
# Required (non-empty) whenever any metric needs a judge — suite YAML,
# AITESTKIT_JUDGE_PROVIDER + AITESTKIT_JUDGE_MODEL, or --judge/--judge-model.
# Must differ from every `models:` entry (same pair = self-grading → error).
judge:
  provider: openai          # openai|anthropic|gemini|litellm|azure|ollama|
                            # bedrock|groq|mistral|vertex|fake
  model: gpt-4o-mini
  temperature: 0.0
  self_consistency: 1       # >1 = majority-vote judging
  max_retries: 3
  timeout_seconds: 60
  api_key_env: OPENAI_API_KEY   # override the default env var

embedder:
  provider: openai
  model: text-embedding-3-small

# --- MODEL MATRIX — run every case against each provider:model --------------
models:
  - "openai:gpt-4o-mini"
  - "anthropic:claude-haiku-4-5"

providers:                    # per-provider knobs for the matrix
  openai:    {provider: openai, model: gpt-4o-mini, rpm: 500}
  anthropic: {provider: anthropic, model: claude-haiku-4-5, rpm: 50}

# --- PROMPT MATRIX — every case runs once per variant -----------------------
prompts:
  - {name: baseline, template: "Answer: {input}"}
  - {name: cot,      template: "Think step by step: {input}"}

# --- PASS/FAIL GATES ---------------------------------------------------------
gates:
  min_pass_rate: 0.9
  max_error_cases: 0
  max_flaky: 0.1
  max_regression_vs_baseline: 0.05
  min_coverage: {smoke: 1}          # tag -> min case count
  per_metric:
    correctness: {min_mean: 0.8, min_p10: 0.6, max_p95: null, min_per_case: 0.5}

coverage:                         # map dataset tags to capabilities
  smoke: "core answers"

# --- RUNTIME -----------------------------------------------------------------
concurrency: 8
repeat: 1                   # >1 re-runs cases — enables flaky detection
case_timeout_seconds: 300
max_cost_usd: 0.50          # hard spend cap; breach → exit code 3
seed: 42

redact:                     # secret + PII scrubbing in reports (default on)
  enabled: true
  patterns: ['sk-[A-Za-z0-9_-]{20,}']

# --- SUBSYSTEM-UNDER-TEST hooks ----------------------------------------------
vectordb: {type: chroma, path: ./db}   # memory|chroma|faiss|qdrant|pgvector|pinecone
memory:   {store: "myapp:store"}       # inspectable MemoryStore for memory metrics
tools:                               # tool registry for agentic metrics
  search: {function: "app:search", requires_confirmation: false}
version_tags: {prompt: v3, kb: "2025-11"}  # stamped into run metadata
```

### Dataset format

Files may be `.yaml`/`.yml`, `.json`, or `.jsonl` (one case object per
line) — same `cases:` / `conversations:` shapes in each. A `.json` file
can carry both keys at once:

```yaml
# datasets/cases.yaml
cases:
  - id: balance-inquiry
    input: "what is my checking balance"
    expected_output: "your checking balance is $1250.00"
    context: ["optional retrieved context for RAG metrics"]
    tools: [get_balance]         # expected tool calls (agentic metrics)
    tags: [smoke, banking, llm]
```

### Available metrics

| Group | Names |
|-------|-------|
| Deterministic | `exact_match`, `contains`, `regex_match`, `json_schema`, `latency`, `max_cost` |
| Judge-based | `correctness`, `custom_rubric` |
| RAG | `faithfulness`, `answer_relevancy`, `groundedness`, `contextual_recall`, `contextual_precision`, `citation_accuracy`, `context_window`, `multi_turn_coherence` |
| Retrieval | `recall_at_k`, `precision_at_k`, `mrr`, `ndcg`, `hit_rate`, `rerank_delta` |
| Agentic | `tool_sequence`, `tool_args_correct`, `tool_preconditions`, `forbidden_tools`, `confirmation_required`, `loop_detection`, `termination_check`, `trajectory_match`, `handoff_correctness`, `task_completion`, `reasoning_coherence`, `plan_validity` |
| Memory | `context_retention`, `contradiction_resolution`, `forgetting_check`, `memory_accuracy` |
| Ingestion | `ingestion_parse`, `ingestion_metadata`, `ingestion_dedup` |
| Vector DB | `vectordb_count`, `embedding_drift` |
| Other | `prompt_sim`, `data_quality`, plus `deepeval`/`ragas` backend bridges |

---

## 3. How to run — step by step

### Step 0 — scaffold (optional)

```bash
aitestkit init myproject/        # writes suite.yaml + datasets/cases.yaml
```

### Step 1 — try the offline examples

Everything under `examples/` runs with **zero API keys**:

```bash
cd examples
export PYTHONPATH=.            # lets python targets import the bundled example apps

aitestkit run suites/llm/llm_suite.yaml        -o /tmp/out-llm
aitestkit run suites/rag/rag_suite.yaml        -o /tmp/out-rag
aitestkit run suites/agent/agentic_suite.yaml    -o /tmp/out-agentic
aitestkit run suites/memory/memory_suite.yaml     -o /tmp/out-memory
aitestkit run suites/a_b_testing/prompt_matrix_suite.yaml -o /tmp/out-prompts
```

### Step 2 — run your own suite

```bash
aitestkit run path/to/suite.yaml -o results/
```

Reports written to `results/`: `results.json` + `results.jsonl`
(raw per-case), `aitestkit-report.html`, plus console output. Choose
formats with `--report console,json,html,junit`.

### Reading the results

Each case in `results.json` carries the input, the SUT's `actual_output`,
a `status` (`passed|failed|error|skipped|flaky`), per-metric verdicts
(`score`, `passed`, `reason`, `details` — the evidence payload), plus
`latency_ms`, `usage`, and the `trajectory` when the target emits one:

```json
{
  "case_id": "balance-inquiry",
  "status": "passed",
  "actual_output": "your checking balance is $1250.00",
  "metrics": [
    {"metric": "exact_match", "score": 1.0, "passed": true,
     "reason": "exact match",
     "details": {"expected": "your checking balance is $1250.00", "actual": "your checking balance is $1250.00"}}
  ]
}
```

Dig with `jq` or open `aitestkit-report.html` — per-guide "read the
results" sections cover the metric-specific `details` shapes:

```bash
jq '.results[] | select(.status!="passed") | {case_id, status,
    failures: [.metrics[] | select(.passed==false) | .metric]}' \
    results/results.json
```

### Step 3 — useful run flags

```bash
aitestkit run suite.yaml --judge anthropic --judge-model claude-haiku-4-5
aitestkit run suite.yaml --cassette cassettes/openai.jsonl --cassette-mode record
aitestkit run suite.yaml --cassette cassettes/openai.jsonl          # replay, offline
aitestkit run suite.yaml --resume results/                          # skip finished cases
aitestkit run suite.yaml --baseline-save main                       # store baseline
aitestkit run suite.yaml --baseline .aitestkit/baselines/main.json  # diff vs baseline
aitestkit run suite.yaml --export otel --otel-endpoint http://localhost:4318
aitestkit run suite.yaml --no-redact -v
```

Security note: `python:` targets run by default (your own code);
`command:` targets execute arbitrary argv and require
`--allow-command-targets`. Use `--no-allow-python-targets` in locked-down CI.

### Step 4 — interpret the exit code

| Code | Meaning                                   |
|------|-------------------------------------------|
| 0    | All gates passed                          |
| 1    | Case failures                             |
| 2    | Config error (bad YAML, missing key, …)   |
| 3    | `max_cost_usd` budget exceeded            |
| 4    | Guardrail/enforcement abort               |
| 5    | Gate failure (e.g. `min_pass_rate` unmet) |

### Step 5 — post-run tooling

```bash
aitestkit report results/results.json --report html,junit   # regenerate reports
aitestkit compare baseline.json results/results.json --tolerance 0.05
aitestkit trends .                                          # drift over recorded runs
aitestkit chunk-compare docs/ golden/queries.yaml \
    --strategies fixed,semantic,hierarchical --embedder openai:text-embedding-3-small
```

Run history accumulates in `.aitestkit/history.jsonl`; named baselines live
in `.aitestkit/baselines/`.

### Typical workflow

```bash
# 1. local dev — replay cassettes, zero spend
aitestkit run suite.yaml --cassette cassettes/main.jsonl

# 2. before merging — record fresh cassettes + save baseline
aitestkit run suite.yaml --cassette cassettes/main.jsonl --cassette-mode record \
    --baseline-save main

# 3. CI — replay + gate; non-zero exit fails the job
aitestkit run suite.yaml --cassette cassettes/main.jsonl -o "$OUT"
```
