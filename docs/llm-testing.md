# Testing LLMs End-to-End

How to test LLM outputs with `aitestkit`: point a target at the model (or
your app around it), score responses with deterministic checks and/or an
LLM judge, compare across models and prompts with matrices, gate the run.

## How it works

```
 dataset case ──► ┌─ target ─┐        ┌─ deterministic ─┐
   input          │ http: raw │ output │ exact_match      │   per-case
   expected_output│   API     │───────►│ contains regex   │──►scores──►
                  │ python:   │        │ json_schema      │      │
                  │   your app│        ├─ judge-based ────┤      ▼
                  └───────────┘        │ correctness      │   gates
                                       │ custom_rubric    │   exit code
                  × models:  matrix    └──────────────────┘
                  × prompts: matrix
```

Two matrix axes multiply your coverage without writing more cases: every
case runs once per `models:` entry **and** once per `prompts:` variant.

---

## Step 1 — Pick the target

### A. Raw model API — `type: http`

Test the model directly; `{input}` is substituted into the request body and
`${ENV_VAR}` into headers/URL:

```yaml
target:
  type: http
  url: "https://api.openai.com/v1/chat/completions"
  method: POST
  headers: {Authorization: "Bearer ${OPENAI_API_KEY}"}
  request_template:
    model: "gpt-4o-mini"
    messages: [{role: user, content: "{input}"}]
    max_tokens: 50
  response_path: "choices.0.message.content"
  kwargs: {timeout_seconds: 30}
```

### B. Your application — `type: python`

The model is inside your code; the callable takes `input` (plus optional
`prompt`, `model`, `provider` kwargs when matrices are in play):

```python
def answer(input: str, *, prompt: str | None = None, **kw) -> str:
    return chat(model="gpt-4o-mini", user=prompt or input)
```

---

## Step 2 — Write the dataset

```yaml
# datasets/llm_cases.yaml
cases:
  - id: greeting
    input: "say hello"
    expected_output: "hello"
    tags: [smoke, llm]

  - id: arithmetic
    input: "what is 2+2"
    expected_output: "4"
    tags: [llm]

  - id: json-mode
    input: "return {\"ok\": true} as json"
    tags: [llm, structured]
```

`expected_output` feeds `exact_match`/`contains`/`correctness`; cases
without it still run for judge/rubric/latency checks.

---

## Step 3 — Configure the suite

```yaml
name: llm-regression

target:
  type: http
  url: "https://api.openai.com/v1/chat/completions"
  headers: {Authorization: "Bearer ${OPENAI_API_KEY}"}
  request_template:
    model: "gpt-4o-mini"
    messages: [{role: user, content: "{input}"}]
  response_path: "choices.0.message.content"

datasets:
  - datasets/llm_cases.yaml

# --- MODEL MATRIX: same cases, several providers -------------------------
models:
  - "openai:gpt-4o-mini"
  - "anthropic:claude-haiku-4-5"
  - "gemini:gemini-2.0-flash"

providers:                                # per-provider knobs
  openai:    {provider: openai, model: gpt-4o-mini, rpm: 500, tpm: 200000}
  anthropic: {provider: anthropic, model: claude-haiku-4-5, rpm: 50}
  gemini:    {provider: gemini, model: gemini-2.0-flash, rpm: 60}

# --- PROMPT MATRIX: same cases, several prompt variants ------------------
prompts:
  - {name: terse,   template: "Answer in one word: {input}"}
  - {name: verbose, template: "You are a thorough assistant. Explain: {input}"}

# --- JUDGE: for subjective metrics ---------------------------------------
judge:
  provider: openai              # or fake for offline dev
  model: gpt-4o-mini
  temperature: 0.0
  self_consistency: 3           # majority vote over 3 judge samples
  max_retries: 3
  timeout_seconds: 60

metrics:
  # --- deterministic (no judge, work offline) ----------------------------
  - name: exact_match                      # output == expected_output
  - name: contains                         # all substrings present
    substrings: ["4"]
    threshold: 0.0                         # informational scoring
  - name: regex_match
    pattern: '\b(Paris|4|hello)\b'         # re.search, DOTALL
  - name: json_schema                      # output is valid JSON per schema
    schema:
      type: object
      properties: {ok: {type: boolean}}
      required: [ok]
  - name: latency
    max_ms: 5000                           # per-case wall clock
  - name: max_cost
    max_usd: 0.01                          # per-case spend ceiling
  - name: prompt_sim                       # prompt-injection / similarity
    template: "Answer in one word: {input}"
    banned: ["ignore previous", "jailbreak"]
    max_tokens: 500

  # --- judge-based (needs provider key) ----------------------------------
  - name: correctness                      # semantic match vs expected
    threshold: 0.7
  - name: custom_rubric                    # free-form grading rubric
    rubric: "Score 1 if the answer is accurate and under 50 words."

concurrency: 8
case_timeout_seconds: 60
max_cost_usd: 1.00                         # run-level spend cap → exit 3

gates:
  min_pass_rate: 0.9
  per_metric:
    correctness: {min_mean: 0.75}
```

### What the matrices produce

- `models:` + `providers:` → each case fans out per `provider:model`;
  results are grouped per model so you can compare pass rates, latency and
  cost head-to-head in the report. `rpm`/`tpm` throttle per provider.
- `prompts:` → each case runs once per variant; the rendered `{input}`
  template is passed as the `prompt` kwarg (python targets) and recorded
  as `prompt_name` in results — A/B test prompt changes like code.

---

## Step 4 — Run it

```bash
# offline dev — no keys: python target + fake judge, or cassettes
aitestkit run llm_suite.yaml --judge fake --judge-model fake -o out/

# real run
export OPENAI_API_KEY=...
aitestkit run llm_suite.yaml -o out/

# record API traffic once, replay deterministically forever
aitestkit run llm_suite.yaml --cassette cassettes/llm.jsonl --cassette-mode record
aitestkit run llm_suite.yaml --cassette cassettes/llm.jsonl      # offline

# override the judge without editing the suite
aitestkit run llm_suite.yaml --judge anthropic --judge-model claude-haiku-4-5

# compare a candidate model/prompt against your stored baseline
aitestkit run llm_suite.yaml --baseline-save gpt4omini
aitestkit run llm_suite.yaml --baseline .aitestkit/baselines/gpt4omini.json \
    --judge-model gpt-4o   # does upgrading the model regress anything?
```

---

## Step 5 — Read the results

`results.json` / `results.jsonl` / `aitestkit-report.html` /
`junit.xml` (`--report junit`), same as every suite. LLM-specific detail:

- **Per-(case × model × prompt) rows** — each matrix cell is a separate
  result tagged with model and `prompt_name`, so the HTML report makes
  cross-model/prompt comparisons directly readable.
- **Metric `details`** — `regex_match` reports the matched text,
  `json_schema` the violation message, `contains` which substrings missed,
  `correctness` the judge rationale.
- **`usage` + `latency_ms` per case** — cost accumulates against
  `max_cost_usd`; a breach exits `3` mid-run.
- **`compare`** — `aitestkit compare old.json new.json --tolerance 0.05`
  prints a REGRESSION verdict and exits `1` — the model-upgrade gate.

Exit codes: `0` pass · `1` failures · `2` config · `3` budget · `4`
guardrail · `5` gate failure.

---

## Worked examples

```bash
cd examples && export PYTHONPATH=.

# basic LLM assertions, offline
uv run aitestkit run suites/llm_suite.yaml -o /tmp/llm-out

# prompt matrix — terse vs verbose variants, offline
uv run aitestkit run suites/prompt_matrix_suite.yaml -o /tmp/pm-out

# live-provider smoke (needs OPENAI_API_KEY, $0.50 cap)
uv run aitestkit run ../ci/live_suite.yaml -o /tmp/live-out --report console,json
```

`llm_suite.yaml` (`exact_match`/`contains`) and
`prompt_matrix_suite.yaml` (two prompt variants + `prompt_sim`) are
ready-made templates — repoint `target.function` at your app
(`support_chat.app:answer`) and run with your judge:

```bash
OPENAI_API_KEY=sk-... aitestkit run suites/llm_suite.yaml \
  --judge openai --judge-model gpt-4o-mini
```
