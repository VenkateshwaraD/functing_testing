# LLM Judge — Evaluation Setup

How `aitestkit` grades subjective output: a built-in `Judge` wraps any
chat provider, sends metric-specific rubric prompts, and parses
structured `{score, reason}` verdicts. No external eval tool needed.
Self-contained guide.

```
metric prompt ──► Judge ──► ChatProvider (openai|anthropic|gemini|…)
 (rubric per metric)   │            │
                      ▼            ▼
              JSON {"score", "reason"} ──► retries + self-consistency
                      │
                      ▼
              MetricResult (score, passed, reason, judge_rationale)
```

---

## 1. When a judge is required

Metrics marked `requires_judge` error out at run start **without an
explicit judge** — CLI flag, suite `judge:` block, or env vars; the
shipped defaults alone don't count:

| Judge metrics | What they grade |
|---|---|
| `correctness` | semantic match vs `expected_output` |
| `custom_rubric` | any rubric you write |
| `faithfulness`, `groundedness`, `answer_relevancy` | RAG generation |
| `contextual_recall`, `contextual_precision`, `citation_accuracy` | RAG context quality |
| `multi_turn_coherence` | consistency across turns |
| `task_completion`, `reasoning_coherence`, `plan_validity` | agent behavior |
| `memory_accuracy` | answer vs stored facts |

Everything else (`exact_match`, `contains`, retrieval ids, tool checks,
`json_schema`, `data_quality`, `prompt_sim`…) is deterministic — no
judge, no keys.

## 2. Pick the judge — precedence

`--judge/--judge-model` **>** `judge:` block **>**
`AITESTKIT_JUDGE_PROVIDER`/`AITESTKIT_JUDGE_MODEL` **>** shipped defaults.

```yaml
judge:
  provider: openai           # openai|anthropic|gemini|litellm|fake
  model: gpt-4o-mini
  temperature: 0.0           # keep 0 — deterministic grading
  self_consistency: 3        # sample k verdicts, average the score
  max_retries: 3             # malformed JSON → backoff retry
  timeout_seconds: 60
  api_key_env: OPENAI_API_KEY
```

```bash
# per run
uv run aitestkit run suite.yaml --judge anthropic --judge-model claude-haiku-4-5

# fully offline — FakeChatProvider returns canned verdicts, no keys
AITESTKIT_JUDGE_PROVIDER=fake AITESTKIT_JUDGE_MODEL=fake \
    uv run aitestkit run suite.yaml
```

**Self-grading guard**: judge `provider:model` must differ from every
`models:` SUT entry — same pair is a `ConfigError` at load, and a
matching model name at runtime logs a bias warning.

## 3. Eval quality controls (built in)

- **JSON-mode + retries** — the judge is asked for strict
  `{"score": …, "reason": …}`; unparseable output retries with backoff,
  then a scoped `MetricError` (never a silent score)
- **`self_consistency: N`** — k samples per verdict; mean score, longest
  rationale kept as evidence. Cheap noise control for borderline cases
- **`pairwise_compare`** — A/B prompts run in *both* orders; large
  position-dependent swings get `[position-bias warning]` appended to the
  reason
- **Prompt-is-per-metric** — each judge metric ships its own rubric
  prompt (claim decomposition for faithfulness, plan-vs-actions for
  `plan_validity`…), so grading criteria stay auditable in
  `metrics/*.py`

## 4. Custom judge providers

Implement `ChatProvider` (`name` attr + `async complete(messages, *,
model, …) -> ProviderResponse`) — must return JSON verdict text
`{"score": 0..1, "reason": "…"}`:

```yaml
judge: {provider: "myco.providers:MyJudge", model: myco-v1}
providers:
  myco: {provider: myco, impl: "myco.providers:MyJudge",
         model: myco-v1, api_key_env: MYCO_API_KEY}
```

See `docs/custom-adapters.md` Part 1.

## 5. Cost attribution

Judge calls are metered separately from the SUT — `judge_usage` vs
`sut_usage` per case, both counting toward `max_cost_usd` (breach →
exit 3). `metadata.judge` in `results.json` stamps which provider:model
graded the run.

## 6. Validate the results

Every judge verdict lands in `metrics[].reason` plus evidence in
`details`:

```json
{"metric": "correctness", "score": 0.7, "passed": true,
 "reason": "covers the key facts but misses the refund deadline",
 "details": {"judge_rationale": "covers the key facts but …",
             "missing": ["refund deadline"]}}
```

```bash
# lowest-scoring judged cases + their rationales
jq '.results[].metrics[] | select(.details.judge_rationale != null)
    | {score, reason}' evidence/run/results.json
```

`faithfulness` goes further — `details.claims[]` lists per-claim
`supported`/`evidence` verdicts so a low score names the exact
unsupported claim.

## 7. External backends (optional)

`aitestkit[deepeval]` / `aitestkit[ragas]` expose those libraries' metric
suites through the same `metrics:` list — lazy-imported, scoped errors
when the extra is missing. Use them for metric coverage beyond the
built-ins; the judge stays the built-in path.

## 7. Live cassettes — record once, replay forever

`suites/judge/judge_live_suite.yaml` is the one suite that makes real
provider calls. The judge is the only live traffic (the SUT target is
deterministic), so one recorded cassette captures every verdict.

```bash
# record — needs the provider key (fails loudly without it)
ANTHROPIC_API_KEY=… uv run aitestkit run suites/judge/judge_live_suite.yaml \
    --cassette cassettes/judge_live.jsonl --cassette-mode record

# replay — zero keys, byte-identical judge responses
uv run aitestkit run suites/judge/judge_live_suite.yaml \
    --cassette cassettes/judge_live.jsonl
```

- `record` wraps the judge provider; each call is hashed on the canonical
  request and written through the `Redactor` — committed cassettes carry
  no keys, only request/response/usage payloads
- `replay` (default when a cassette path is given) returns the recorded
  response for the same request hash — deterministic and offline
- `wrong-answer` in `dataset_testcases/judge_live_cases.yaml` is
  engineered to genuinely fail (expected ≠ actual), so the evidence
  proves the judge discriminates — real score ~0.1, not a rubber-stamp
- Missing key in record mode → `config error: provider 'anthropic'
  requires ANTHROPIC_API_KEY…` (exit 2), never a silent fallback

## Config reference

| Key | Purpose |
|---|---|
| `judge.provider`/`model` | grading backend (required for judge metrics) |
| `judge.self_consistency` | k-sample majority voting |
| `judge.max_retries` | malformed-verdict retry bound |
| `judge.api_key_env` | override the provider's default env var |
| `--judge`, `--judge-model` | CLI override per run |
| `AITESTKIT_JUDGE_*` | env-var defaults |
| `details.judge_rationale` | per-verdict evidence in results |
| `judge_usage` / `sut_usage` | split cost accounting per case |

## Related

- `docs/llm-testing.md` — judge in the model/prompt matrix flow
- `docs/rag-testing.md` — faithfulness/groundedness internals
- `docs/custom-adapters.md` — `impl:` provider contract
- `docs/reliability-testing.md` — `self_consistency` + judge noise vs
  flaky detection
