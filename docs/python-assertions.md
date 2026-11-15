# Python assertions — imperative checks for pytest & code-first tests

The `assertions/` package holds reusable `.py` assertion helpers for
banking/payment SUT testing. Every function raises `AssertionError` with a
descriptive message — use them in `test_*.py`, inside `assert_test`
wrappers, or anywhere Python runs.

## Modules

| Module | What it asserts |
|---|---|
| `schema_assertions` | JSON Schema validity, required fields, ISO currency, IBAN format, no-PII |
| `semantic_assertions` | cosine similarity, term presence, refusal detection, unresolved placeholders |
| `business_assertions` | positive amounts, daily limits, sufficient funds, state-machine transitions, idempotent retries |
| `agent_assertions` | tool called/never called, arg matching, tool order, confirmation-before-destructive, termination, loop guard |
| `workflow_assertions` | handoff sequence, agent participation, step budget, error-free tool calls |

## Usage

```python
from assertions import (
    assert_tool_called, assert_confirmation_precedes,
    assert_no_pii, assert_sufficient_funds,
)

def test_transfer_flow(trajectory, payload):
    assert_tool_called(trajectory, "transfer_funds",
                       args={"amount": 200})
    assert_confirmation_precedes(trajectory, "transfer_funds")
    assert_no_pii(payload)
    assert_sufficient_funds(balance=payload["balance"],
                            amount=payload["amount"])
```

Trajectory arguments accept `aitestkit.models.Trajectory`, a serialized
`results.json` dict, or a plain event list — all are normalized
internally. `assert_json_schema` uses the `jsonschema` package (already a
core dependency).

## Why a package, not metrics?

Suites declare expectations in YAML (`assert:`/`metrics:` blocks). These
helpers are the same idea in code — for tests that live in pytest, need
custom data shapes, or must be composed with non-aitestkit assertions.
