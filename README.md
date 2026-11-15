# aitestkit

Production-style functional testing for LLM applications, RAG pipelines,
single- and multi-agent systems. Define test suites in YAML or code, run
them against your app through target adapters, score them with
deterministic and LLM-judged metrics, and gate releases on the results.

Everything below runs offline out of the box — the `fake` provider and
deterministic metrics need no API keys. Live providers plug in via extras.

## Install

```bash
pip install aitestkit                    # core (offline-capable)
pip install "aitestkit[anthropic]"       # provider extras
pip install "aitestkit[chroma]"          # vector DB extras
pip install "aitestkit[deepeval,ragas]"  # external metric backends
pip install "aitestkit[otel]"            # OpenTelemetry export
```

Requires Python 3.13+.

## Quickstart

```bash
# run the bundled demo suites — no keys needed
cd examples
PYTHONPATH=. aitestkit run suites/llm/llm_suite.yaml
PYTHONPATH=. aitestkit run suites/rag/rag_suite.yaml
PYTHONPATH=. aitestkit run suites/agent/agentic_suite.yaml
PYTHONPATH=. aitestkit run suites/memory/memory_suite.yaml
PYTHONPATH=. aitestkit run suites/a_b_testing/prompt_matrix_suite.yaml

# compare chunking strategies on a golden set
PYTHONPATH=. aitestkit chunk-compare dataset_testcases/corpus dataset_testcases/golden/retrieval.yaml \
    --strategies fixed,hierarchical --embedder fake:fake
```

## Anatomy of a suite

```yaml
name: my-suite
target:                        # the app under test
  type: python                 # python | http | command | langgraph
  function: "myapp.chat:run"

datasets:
  - cases.yaml                 # cases + conversations

judge:                         # LLM-as-judge provider (only if judge metrics used)
  provider: openai
  model: gpt-4o-mini
  self_consistency: 1          # k-sample judge scoring (bias control)

embedder:
  provider: openai
  model: text-embedding-3-small

models:                        # SUT matrix — each case runs per provider:model
  - openai:gpt-4o-mini
  - anthropic:claude-sonnet-4-5

prompts:                       # prompt matrix — each case runs per variant
  - name: terse
    template: "Answer in one word: {input}"

providers:                     # per-provider tuning
  openai: {rpm: 500, api_key_env: OPENAI_API_KEY}

memory:                        # inspectable MemoryStore for memory metrics
  store: "myapp.memory:store"

vectordb:                      # vector store under test
  type: chroma
  collection: docs

tools:                         # tool registry with fault injection + mocks
  search: {function: "myapp.tools:search", fault: {times: 1, error: "timeout"}}

metrics:
  - name: faithfulness
    threshold: 0.85
  - name: recall_at_k
    k: 5
  - name: tool_sequence

repeat: 3                      # --repeat: pass/fail variance -> flaky status
concurrency: 8
case_timeout_seconds: 300
max_cost_usd: 0.50             # run-level budget cap

gates:                         # quality gates (exit code 5 on failure)
  min_pass_rate: 0.9
  max_error_cases: 0
  max_flaky: 0.2               # fraction of cases allowed flaky
  max_skip_rate: 0.0           # fraction of metric results allowed skipped
  per_metric:
    faithfulness: {min_mean: 0.85}
  max_regression_vs_baseline: 0.02

coverage:                      # requirement coverage by tag
  tools: agentic-tools
  memory: memory
```

Cases (`cases.yaml`):

```yaml
cases:
  - id: refund-policy
    input: "can I get a refund?"
    expected_output: "refunds within 30 days"
    context: ["Refund policy: full refund within 30 days of purchase."]
    expected_retrieval_ids: ["policy:0"]
    expected_tools: [{name: get_balance, args: {account: checking}}]
    tags: [rag, policy]

conversations:
  - id: remember-name
    session_id: s1
    turns:
      - input: "my name is ada"
      - input: "what is my name"
    metadata: {must_remember: ["ada"]}
```

## CLI

```bash
aitestkit run suite.yaml [--out DIR] [--report console,json,junit,html]
    [--judge anthropic --judge-model claude-sonnet-4-5]
    [--baseline-save TAG] [--baseline FILE]
    [--cassette FILE --cassette-mode record|replay]
    [--resume results.jsonl]
    [--export otel,allure] [--otel-endpoint URL]
    [--allow-command-targets] [--no-allow-python-targets]

aitestkit report results.json --out DIR --report json,junit,html
aitestkit compare baseline.json results.json
aitestkit trends                          # metric drift over history
aitestkit chunk-compare corpus/ golden.yaml --strategies fixed,semantic
aitestkit doctor [suite.yaml]             # env/deps diagnostics
aitestkit init                            # scaffold a suite
```

## Exit codes

| code | meaning |
|------|---------|
| 0    | all cases passed, gates met |
| 1    | test failures or gate violations |
| 2    | configuration error (bad suite, missing keys) |
| 3    | budget cap hit mid-run |
| 5    | quality gate failed |

## Results and statuses

Cases end as `passed | failed | error | skipped | flaky` — provider
timeouts and infra failures are `error` (never quality failures), missing
required fields are `skipped`, mixed repeat outcomes are `flaky`. Metric
results carry `scored | error | skipped` plus a `details` evidence payload
(judge rationale, unsupported claims, tool diffs, trajectory diffs).

Every run writes `results.jsonl` incrementally (crash-safe; `--resume`
continues), `results.json`, `junit.xml`, and a single-file
`aitestkit-report.html` — per-case detail, metric scores, collapsible
trajectories, cost/latency/token breakdowns split into judge vs SUT.
All persisted output is redacted by default (`--no-redact` to disable).

## What you can test

- **LLM apps**: exact/regex/JSON-schema/contains, latency, cost, LLM-judge
  correctness + custom rubrics (self-consistency sampling, judge==SUT bias
  warning, pairwise position-swap)
- **RAG**: recall@k/precision@k/MRR/nDCG/hit-rate against golden ids,
  rerank delta, faithfulness (claim decomposition → per-claim
  verification), groundedness, answer relevancy, contextual
  precision/recall, citation accuracy, context-window budgets,
  multi-turn coherence, stage attribution (ingestion/retrieval/generation)
- **Ingestion**: parse, metadata completeness, dedup checks
- **Agents**: tool sequence/args/preconditions/forbidden tools/confirmation
  requirements, handoffs, loop detection, termination, task completion,
  reasoning coherence, plan validity, trajectory diff vs golden
- **Memory**: multi-turn conversations with session sharing, direct
  MemoryStore inspection, context retention, contradiction resolution,
  forgetting, restart persistence (JsonFileMemoryStore)
- **Vector DBs**: index count, embedding drift (re-embed sample, cosine
  shift) on Chroma/FAISS/pgvector/Qdrant/Pinecone
- **Data quality**: column expectations (not_null/unique/regex/range/set/
  PII) + Great Expectations bridge
- **External backends**: any deepeval or ragas metric via the adapter
  metrics

## Docs

Standalone guides under `docs/` — each covers config → run → validating
`results.json`:

| Guide | Covers |
|---|---|
| `docs/getting-started.md` | install, suite anatomy, metrics list, CLI flags |
| `docs/harness-overview.md` | trajectory replay, tool mocks/faults, prompt sim |
| `docs/response-testing.md` | output metrics (exact/contains/regex/json/latency) + judge |
| `docs/schema-testing.md` | `json_schema` + `data_quality` expectations |
| `docs/prompt-testing.md` | prompt variants, `prompt_sim`, injection/adversarial probes |
| `docs/llm-testing.md` | model/prompt matrices, cassettes, baselines, judges |
| `docs/ab-testing.md` | prompt/model variant matrices, pairwise judge |
| `docs/judge-testing.md` | LLM judge: providers, self-consistency, bias guards |
| `docs/python-assertions.md` | `assertions/` package — imperative `.py` checks for pytest |
| `docs/rag-testing.md` | retrieval + generation + ingestion metrics, chunk-compare |
| `docs/vectordb-testing.md` | index integrity, embedding drift, retrieval consistency |
| `docs/agent-testing.md` | trajectories, tool calls, guardrails, handoffs |
| `docs/memory-testing.md` | multi-turn conversations, MemoryStore inspection |
| `docs/api-testing.md` | REST endpoints via `type: http` |
| `docs/mcp-testing.md` | MCP servers via `type: mcp` |
| `docs/reliability-testing.md` | flaky detection, fault injection, baselines, drift |
| `docs/regression-testing.md` | behavioral drift across model/prompt/KB versions |
| `docs/banking-testing.md` | worked example: funds-transfer agent end-to-end |
| `docs/custom-adapters.md` | custom providers + `type: custom` target adapters |
| `docs/ci-cd.md` | GitLab pipeline, artifacts, release process |

## pytest

```bash
# files named *.aitestkit.yaml are collected as tests automatically
pytest                                  # (plugin registered via entry point)
```

```python
from aitestkit import TestCase, assert_test
from aitestkit.metrics.deterministic import Contains
from aitestkit.targets.python_target import PythonTarget

async def test_greeting():
    output = await PythonTarget(spec="myapp:chat").invoke(TestCase(input="hi"))
    await assert_test(TestCase(input="hi"), output, [Contains(substrings=["hello"])])
```

## CI

`results.jsonl`, `junit.xml`, and `aitestkit-report.html` are designed as
CI artifacts; `$GITHUB_STEP_SUMMARY` and `::error` annotations are emitted
automatically on GitHub Actions. Nightly live-provider runs can replay a
small smoke suite with `--max-cost-usd` as a spend cap. See
`.github/workflows/`.

## Design notes

- Versioned result schema (`schema_version`) — baselines/cassettes/
  trajectories migrate instead of silently breaking.
- Judge calls are attributed to metrics; `CaseResult` splits `judge_usage`
  vs `sut_usage` so costs never blur.
- `command:` targets run argv without a shell and require
  `--allow-command-targets`; `python:` targets can be disabled with
  `--no-allow-python-targets` for locked-down CI.
- Provider adapters lazy-import — missing extras produce
  `pip install 'aitestkit[extra]'` hints, not ImportErrors.
- Cassettes record/replay provider traffic (sanitized at write time) so
  PR CI needs zero secrets.
