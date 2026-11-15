# Regression Testing — Behavioral Drift Across Versions

Detect output drift when the model, prompt, embedder, or knowledge base
changes: save a baseline run, change one thing, compare. Self-contained
guide.

```
run (v1) --baseline-save ──► baseline-v1.json
                                   │
            change model/prompt ──►│ run (v2)
                                   ▼
                 aitestkit compare baseline.json current/results.json
                                   │
                 verdict: NEUTRAL | IMPROVEMENT | REGRESSION
```

---

## 1. The baseline workflow

```bash
# 1. capture the known-good run (the argument is a tag, not a path —
#    writes <out>/baseline-<tag>.json)
aitestkit run suites/llm/llm_suite.yaml -o evidence/v1 --baseline-save v1

# 2. change the SUT: new model version, edited prompt, re-indexed corpus…

# 3. rerun + compare
aitestkit run suites/llm/llm_v2_suite.yaml -o evidence/v2
aitestkit compare evidence/v1/baseline-v1.json \
                  evidence/v2/results.json --tolerance 0.05
```

`--baseline <file>` on `run` does the comparison inline during the run —
the regression magnitude (`max_metric_regression`) can feed gate logic.

## 2. What compare detects

Per case × per metric, plus run-level signals:

- **Score drops** beyond `--tolerance` (default 0.05) → regression entry
- **Status rank drops** (`passed → flaky → failed/error`) — counted even
  when scores are binary
- **New / missing cases** between the two runs
- **`metadata_drift`** — judge model, embedder, `model_versions`, aitestkit
  version, seed, `config_hash` — explains *why* the numbers moved
- **`metric_summary_delta`** — per-metric mean + pass-rate deltas

Verdicts: `REGRESSION` (any drop), `IMPROVEMENT` (only gains), `NEUTRAL`.

```bash
# machine-readable matrix of every case pair
jq '.matrix[] | {case_id, baseline_status, current_status}' comparison.json
```

## 3. Drift sources — how to test each

| Change | Setup | Detector |
|---|---|---|
| **Model version** | `models:` matrix in one run, or repoint `target`/`models:` in a v2 suite | `compare` + `model_versions` in metadata_drift |
| **Prompt edit** | `prompts:` variants (matrix) for A/B in one run; v2 suite for drift vs baseline | `compare` on `[variant]` case ids; `prompt_sim` lints the *new* template (placeholders/banned/tokens — it's a linter, not a differ) |
| **Embedder swap** | re-index, rerun retrieval suite | `compare` on `recall_at_k`/`mrr`; `embedding_drift` baseline vectors |
| **KB/corpus change** | versioned `GoldenSet` + rerun | retrieval metric deltas |
| **Judge swap** | `judge:` block change | flagged in `metadata_drift` — explains score shifts that aren't SUT regressions |

## 4. Separating drift from noise

- **`repeat: N`** on the suite — each case runs N times; `flaky` status and
  CI significance keep sampling noise out of the verdict
- **Cassettes** (`--cassette rec.json --cassette-mode record|replay`) —
  pin provider responses so a prompt-diff comparison isn't confounded by
  sampling; flip to `record` after an intentional change
- **`seed`** on the judge config — deterministic grading across runs
- **`trends`** — `aitestkit trends` over `.aitestkit/history.jsonl` shows
  slow drift a single pairwise compare misses

## 5. CI gating

```bash
aitestkit run suite.yaml --baseline baseline-main.json   # inline compare
aitestkit compare baseline-main.json out/results.json \
    --tolerance 0.1 && echo "no regression"
```

`compare` prints the verdict to stdout; pair with `run`'s exit codes
(`5` = gate failure) or assert on `max_metric_regression` in your pipeline.
See `docs/ci-cd.md`.

## 6. Worked example — real REGRESSION

`suites/llm/llm_v2_suite.yaml` is `llm_suite.yaml` repointed at
`apps.demo_app:answer_v2` — a simulated model upgrade with chattier
phrasing (`"The answer is: Paris."` vs `"Paris"`). Fully offline:

```bash
PYTHONPATH=examples uv run aitestkit run suites/llm/llm_suite.yaml \
    -o evidence/reg/v1 --baseline-save v1
PYTHONPATH=examples uv run aitestkit run suites/llm/llm_v2_suite.yaml \
    -o evidence/reg/v2
uv run aitestkit compare evidence/reg/v1/baseline-v1.json \
                         evidence/reg/v2/results.json
```

Actual output:

```
== compare verdict: REGRESSION
   REGRESS arithmetic exact_match: 1.0 -> 0.0
   REGRESS arithmetic _status: passed -> failed
   REGRESS wire-cutoff exact_match: 1.0 -> 0.0
   REGRESS greeting exact_match: 1.0 -> 0.0
   metadata drift: config hash differs
```

`exact_match` catches wording drift; to accept paraphrases grade with
`correctness` (judge) instead — then only *semantic* regressions flag.

## Config reference

| Flag / key | Purpose |
|---|---|
| `--baseline-save <tag>` | persist run as `baseline-<tag>.json` |
| `--baseline <file>` | compare inline during the run |
| `compare A B --tolerance` | pairwise verdict (default tol 0.05) |
| `repeat:` / `max_flaky:` | noise control — flaky ≠ regression |
| `--cassette` / `--cassette-mode` | replay-pinned comparisons |
| `trends` | longitudinal metric drift |
| `metadata.model_versions` | per-run model stamps → drift attribution |

## Related

- `docs/llm-testing.md` — model/prompt matrices, cassettes
- `docs/reliability-testing.md` — flaky detection, baselines, trends
- `docs/ci-cd.md` — gating runs on regressions
- `docs/prompt-testing.md` — `prompt_sim` template linting
