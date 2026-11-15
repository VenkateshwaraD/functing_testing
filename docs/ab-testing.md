# A/B Testing — Prompt & Model Variants in One Run

Run the same cases against variant prompts or models side-by-side: the
`prompts:`/`models:` matrices are the A/B mechanism; `compare` handles
A/B *across* runs. Self-contained guide.

---

## 1. Two kinds of A/B

| Question | Mechanism |
|---|---|
| "Which variant is better **right now**?" | `prompts:` / `models:` matrix — one run, per-variant results |
| "Did v2 change vs v1?" | `aitestkit compare` — baseline vs current run (see `docs/regression-testing.md`) |

## 2. Configure

**Prompt variants** (`prompts:`) — every case runs once per variant;
`file:` (relative to suite) or inline `template:` with `{input}` /
`{metadata_key}` placeholders:

```yaml
prompts:
  - {name: terse,   template: "Answer in one word: {input}"}
  - {name: verbose, template: "Explain thoroughly: {input}"}
```

**Model variants** (`models:`) — same matrix for SUT providers:

```yaml
models:
  - {provider: openai,    model: gpt-4o-mini}
  - {provider: anthropic, model: claude-haiku-4-5}
```

Both matrices compose: N prompts × M models × K cases. Variants appear as
`case_id[variant]` and `model:name` tags on each result.

## 3. Run

```bash
PYTHONPATH=examples uv run aitestkit run \
    suites/a_b_testing/prompt_matrix_suite.yaml -o evidence/ab
```

Shipped demo: 3 cases × 2 prompts = 6 results, fully offline. Output:

```
PASS wire-cutoff[terse]     'echo what is the wire transfer cutoff'
PASS wire-cutoff[verbose]   'echo what is the wire transfer cutoff'
PASS balance-inquiry[terse] 'echo what is my checking balance'
...
== 6 cases — passed=6 · gate PASS
```

### Same case, genuinely different prompts — `prompt_ab_suite`

`suites/a_b_testing/prompt_ab_suite.yaml` runs the banking cases under two
prompt **files** — `formal` (`customer_support_v1.txt`) vs `casual`
(`customer_support_v2.txt`). Version-by-file is the real A/B workflow: edit
a file, rerun, diff variants. The demo target
(`apps.demo_app:banking_reply`) styles its reply per variant, so A and B
outputs actually diverge:

```
balance-inquiry[formal] -> your checking balance is $1250.00
balance-inquiry[casual] -> your checking balance is $1250.00
                           — happy to help with anything else.
```

Every rendered prompt lands in `prompts.json` for audit — file contents +
`{input}` substitution, per case, per variant.

`suites/prompt/prompt_files_suite.yaml` does the same for the
`payment_assistant`/`loan_assistant` prompts — file-render coverage so a
broken placeholder or deleted policy line fails CI.

## 4. Validate results — per-variant comparison

Results are tagged, so slice by variant:

```bash
# pass rate per variant
jq -r '.results[] | .case_id' evidence/ab/results.json | sort | uniq -c

# variant-scoped scores — e.g. correctness per variant per case
jq '.results[] | {id: .case_id,
     scores: [.metrics[] | {(.metric): .score}] | add}' \
   evidence/ab/results.json
```

`prompts.json` in the out dir snapshots which rendered prompt each
variant produced — the audit trail for "what exactly did B see?":

```bash
# shape: {digest, schema_version, prompts: {case_id: {variant: rendered}}}
jq '.prompts.greeting.terse' evidence/ab/prompts.json
# → "Answer in one word: say hello"
```

**For real quality A/B**, use a judge metric (`correctness`,
`custom_rubric`) rather than `contains` — then compare mean scores
per variant (`metric_summary_delta` also breaks down by metric).

## 5. Judge-scored pairwise A/B (library API)

`Judge.pairwise_compare(prompt_template, output_a=…, output_b=…)` judges
two outputs head-to-head **in both orders** — positional-bias swings are
flagged in the reason. Not wired to a suite metric today; use it from a
custom metric or a script:

```python
verdict, usage = await judge.pairwise_compare(
    "Which answer is better? A: {output_a} B: {output_b} "
    "Reply JSON {\"score\": 0-1, \"reason\": \"...\"} where 1 = A wins",
    output_a=terse_out, output_b=verbose_out,
)
```

## 6. A/B across runs

When variants live in different runs (deploy A vs deploy B), the matrix
doesn't apply — use `compare` on the two `results.json` files; verdict +
per-case deltas + `metadata_drift` attribution, exit 1 on REGRESSION.

## Config reference

| Key | Purpose |
|---|---|
| `prompts: [{name, template|file}]` | prompt variant matrix |
| `models: [{provider, model}]` | model variant matrix |
| `case_id[variant]` | per-variant result tag in results.json |
| `prompts.json` | rendered-prompt snapshot per variant |
| `judge.pairwise_compare` | library-only head-to-head judge |
| `compare` + `baseline:` | cross-run A/B |

## Related

- `docs/regression-testing.md` — baseline/compare, drift across versions
- `docs/llm-testing.md` — full matrix anatomy (models × prompts × cases)
- `docs/judge-testing.md` — judge config, self-consistency, bias guards
- `docs/custom-adapters.md` — custom metric for `pairwise_compare`
