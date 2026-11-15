# Schema Testing

How to validate shapes, not strings: `json_schema` checks the SUT's response
parses as JSON and conforms to a JSON Schema; `data_quality` runs column
expectations over row data in case metadata. Both deterministic — no judge,
no API calls.

```
response text ──► json_schema ──► parse → validate → 1.0/0.0 + reason
metadata rows ──► data_quality ──► per-expectation pass/fail → fraction
```

---

## 1. `json_schema` — contract-check the response

```yaml
metrics:
  - name: json_schema
    schema:                              # full JSON Schema, any draft
      type: object                       #   jsonschema lib supports
      required: [status, amount, currency]
      properties:
        status: {type: string, enum: [approved, pending, denied]}
        amount: {type: number, minimum: 0}
        currency: {type: string}
      additionalProperties: false
```

Two distinct failure modes land in `reason`:

- `output is not valid JSON: …` — the model emitted prose/markdown fences
- `schema violation: …` — parsed, but wrong shape (`details.path` points at
  the offending key)

For adversarial/freeform cases — where a schema violation is the
*expected* outcome — mark the case `expect: fail` instead of weakening
the metric. A violation then reports `xfailed` (detector confirmed); a
surprising *pass* fails the case with an `unexpected_pass` detail.
`threshold: 0.0` is rejected at load unless the spec also sets
`informational: true`.

---

## 2. `data_quality` — expectations over rows

Rows and expectations live in **case metadata** — the metric ignores the
target output entirely (pair it with any target; skipped when no rows):

```yaml
- id: accounts-export
  input: "export check"
  metadata:
    rows:
      - {id: 1, email: "a@bank.com", balance: 50.0, status: active}
      - {id: 1, email: null, balance: -5.0, status: archived}
    expectations:
      - {type: not_null, column: email}
      - {type: unique, column: id}
      - {type: in_range, column: balance, min: 0}
      - {type: in_set, column: status, values: [active, closed]}
```

| `type` | Checks | Params |
|---|---|---|
| `not_null` | no null/empty values | `column` |
| `unique` | no duplicate values | `column` |
| `matches_regex` | every value full-matches | `column`, `pattern` |
| `in_range` | numeric bounds | `column`, `min`, `max` |
| `in_set` | values within allowed set | `column`, `values` |
| `row_count` | total row count bounds | `min`, `max` |
| `no_pii` | no SSN/email/card-shaped values | `column` |

Score = fraction of expectations passed; `reason` names the first five
failures (`unique(id): 1 duplicates of 2`); `details.results` has every
expectation verdict.

**Great Expectations backend**: `metadata.backend: gx` translates the same
specs to GX when `aitestkit[gx]` is installed — missing extra surfaces as a
scoped metric error, nothing else breaks.

---

## 3. Per-case `assert:` carries its own params

Per-case metric dicts build **fresh** — suite-level params aren't
inherited. A case that wants `json_schema` must restate the schema:

```yaml
assert:
  - name: json_schema
    informational: true               # record the score, never gate
    schema: {type: object, required: [status]}
```

Or make the case itself a detection probe — `expect: fail` turns a scored
violation into `xfailed` rather than a green pass:

```yaml
- id: freeform-response
  input: "tell me a story"
  expect: fail                        # valid JSON here would be a surprise
  assert:
    - name: json_schema
      schema: {type: object, required: [status]}
```

Bare names (`assert: [exact_match]`) reuse the suite's configured instance.

---

## 4. Run the demo

```bash
uv run aitestkit run suites/schema/schema_suite.yaml \
    --allow-command-targets -o evidence/schema
```

`suites/schema/schema_suite.yaml` + `dataset_testcases/schema_cases.yaml` — command
target emits canned JSON when the input says "json", prose otherwise.
(`--allow-command-targets` required: argv targets are opt-in.)

```
PASS structured-response   json_schema 1.0  valid against schema
XFLK freeform-response     json_schema 0.0  output is not valid JSON: …
PASS accounts-clean        data_quality 1.0
XFLK transactions-dirty    data_quality 0.0  not_null(email): 1 null of 2;
                            unique(id): 1 duplicates of 2; …
```

Detection cases report **xfailed** — the violation was the expected
outcome, and XFAIL counts as judged+passing in `min_pass_rate`. If the
target ever *did* return valid JSON on `freeform-response`, the case
would flip to `failed` (`unexpected_pass`) — detectors must keep
detecting.

---

## 5. Validate the results

Runs write `results.json`, `results.jsonl`, and `aitestkit-report.html` to
`-o <dir>`. Every metric verdict carries `score`, `passed`, `reason`, and a
`details` evidence payload — that payload is where schema verdicts live.

**`json_schema` failures** — `reason` distinguishes parse errors from
schema violations; `details.path` is the violating key path:

```json
{"metric": "json_schema", "score": 0.0, "passed": false,
 "reason": "output is not valid JSON: Expecting value: line 1 column 1"}
```

(the case above it reads `"status": "xfailed"` when `expect: fail` is set)

**`data_quality` failures** — `details.results` lists every expectation:

```json
{"metric": "data_quality", "score": 0.0, "passed": false,
 "reason": "not_null(email): 1 null of 2; unique(id): 1 duplicates of 2; …",
 "details": {"backend": "builtin", "passed": 0, "failed": 4,
   "results": [
     {"expectation": "not_null(email)", "passed": false,
      "reason": "1 null of 2"},
     {"expectation": "unique(id)", "passed": false,
      "reason": "1 duplicates of 2"}]}}
```

Quick CI checks with `jq`:

```bash
# any schema metric below its intended bar?
jq '.results[].metrics[] | select(.metric=="json_schema") | {score, reason}' \
    evidence/schema/results.json

# failed expectations across all cases
jq '.results[].metrics[] | select(.metric=="data_quality")
    | .details.results[] | select(.passed==false)' \
    evidence/schema/results.json
```

**Gate on it** — aggregate schema health over the whole run:

```yaml
gates:
  min_pass_rate: 1.0
  per_metric:
    json_schema: {min_mean: 0.95}      # ≥95% of outputs schema-valid
    data_quality: {min_mean: 1.0}      # no expectation failures tolerated
```

`min_pass_rate` counts case verdicts (xfailed counts as passing);
`per_metric.min_mean` averages scores — with `expect: fail` cases the
mean still reflects raw scores, so per-metric floors should target the
healthy cases. Exit `0` gates pass, `1` metric failure, `5` gate failure.

## Config reference

| Key | Purpose |
|---|---|
| `json_schema.schema` | JSON Schema the parsed output must satisfy |
| `metadata.rows` | list of dicts `data_quality` evaluates |
| `metadata.expectations` | `{type, column, …params}` spec list |
| `metadata.backend` | `builtin` (default) or `gx` |
| `expect: fail` | case-level detection probe → `xfailed` on violation, `failed` on unexpected pass |
| `informational: true` | metric scores but never gates (also unlocks `threshold: 0.0`) |
| `details.results` | per-expectation verdicts (data_quality) |
| `details.path` | violating key path (json_schema) |

## Related

- `docs/response-testing.md` — the other output metrics
- `docs/getting-started.md` — full metric list incl. backend bridges
