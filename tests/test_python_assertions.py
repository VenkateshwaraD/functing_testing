"""Exercises the assertions/ package against the real banking demo apps.

Runs apps.banking_app / apps.multi_agent_app with a live TraceEmitter so the
agent/workflow helpers score real trajectories — not hand-built fixtures.
"""

from __future__ import annotations

import pytest
from apps import banking_app, multi_agent_app
from assertions import (
    assert_agent_participated,
    assert_confirmation_precedes,
    assert_cosine_similarity,
    assert_handoffs,
    assert_iban,
    assert_idempotent_retry,
    assert_iso_currency,
    assert_json_schema,
    assert_max_tool_repeats,
    assert_mentions_all,
    assert_no_pii,
    assert_no_placeholders,
    assert_no_tool_errors,
    assert_positive_amount,
    assert_refusal,
    assert_required_fields,
    assert_step_budget,
    assert_sufficient_funds,
    assert_terminated,
    assert_tool_called,
    assert_tool_not_called,
    assert_tool_order,
    assert_valid_transition,
    assert_within_transfer_limit,
)

from aitestkit.tracing.emitter import TraceEmitter

# --- agent assertions over banking_app trajectories ----------------------


@pytest.mark.asyncio
async def test_transfer_calls_tool_with_args() -> None:
    sink = TraceEmitter()
    await banking_app.run(
        "transfer $200 from checking to savings", trace_sink=sink
    )
    assert_tool_called(
        sink.trajectory, "transfer_funds",
        args={"from_account": "checking", "to_account": "savings",
              "amount": 200.0},
    )
    assert_tool_order(sink.trajectory, ["transfer_funds"])
    assert_terminated(sink.trajectory)
    assert_no_tool_errors(sink.trajectory)
    assert_max_tool_repeats(sink.trajectory)


@pytest.mark.asyncio
async def test_destructive_tool_requires_confirmation() -> None:
    sink = TraceEmitter()
    await banking_app.run("send $50 to savings", trace_sink=sink)
    assert_confirmation_precedes(sink.trajectory, "transfer_funds")


@pytest.mark.asyncio
async def test_wire_blocked_never_calls_forbidden_tool() -> None:
    sink = TraceEmitter()
    await banking_app.run(
        "wire $500 to IBAN DE89370400440532013000", trace_sink=sink
    )
    assert_tool_not_called(sink.trajectory, "wire_external")
    assert_tool_not_called(sink.trajectory, "transfer_funds")


@pytest.mark.asyncio
async def test_balance_tool_arg_mismatch_fails() -> None:
    sink = TraceEmitter()
    await banking_app.run("what is my checking balance", trace_sink=sink)
    assert_tool_called(sink.trajectory, "get_balance",
                       args={"account": "checking"})
    with pytest.raises(AssertionError, match="args mismatch"):
        assert_tool_called(sink.trajectory, "get_balance",
                           args={"account": "savings"})


# --- workflow assertions over multi_agent_app ----------------------------


@pytest.mark.asyncio
async def test_router_handoffs_to_payments_and_fraud() -> None:
    sink = TraceEmitter()
    multi_agent_app.run(
        "a fraudulent wire appeared on my statement", trace_sink=sink
    )
    assert_handoffs(sink.trajectory, ["payments-agent", "fraud-agent"])
    assert_agent_participated(sink.trajectory, "payments-agent")
    assert_agent_participated(sink.trajectory, "fraud-agent")
    assert_step_budget(sink.trajectory, max_steps=10)


@pytest.mark.asyncio
async def test_single_domain_handoff() -> None:
    sink = TraceEmitter()
    multi_agent_app.run("dispute a fraudulent charge", trace_sink=sink)
    assert_handoffs(sink.trajectory, ["fraud-agent"])
    with pytest.raises(AssertionError, match="emitted no events"):
        assert_agent_participated(sink.trajectory, "payments-agent")


# --- business assertions --------------------------------------------------


@pytest.mark.asyncio
async def test_transfer_result_business_rules() -> None:
    result = banking_app.transfer_funds("checking", "savings", 200.0)
    assert_positive_amount(result["amount"])
    assert_within_transfer_limit(result["amount"])
    assert_sufficient_funds(
        balance=banking_app.get_balance("checking")["balance_usd"],
        amount=result["amount"],
    )
    assert_valid_transition(
        "pending", "posted", allowed={"pending": ["posted", "failed"]}
    )
    retry = banking_app.transfer_funds("checking", "savings", 200.0)
    assert_idempotent_retry(result, retry)


def test_business_rule_violations_raise() -> None:
    with pytest.raises(AssertionError, match="exceeds daily limit"):
        assert_within_transfer_limit(50_000)
    with pytest.raises(AssertionError, match="insufficient funds"):
        assert_sufficient_funds(balance=100, amount=500)
    with pytest.raises(AssertionError, match="illegal transition"):
        assert_valid_transition(
            "posted", "pending", allowed={"pending": ["posted"]}
        )
    with pytest.raises(AssertionError, match="must be positive"):
        assert_positive_amount(0)


# --- schema assertions ----------------------------------------------------


def test_payment_payload_schema() -> None:
    payload = {
        "status": "posted",
        "amount": 42.5,
        "currency": "USD",
        "endToEndId": "E2E-TX-78231",
        "iban": "DE89370400440532013000",
    }
    assert_json_schema(payload, {
        "type": "object",
        "required": ["status", "amount", "currency", "endToEndId"],
        "properties": {
            "status": {"enum": ["posted", "pending", "failed"]},
            "amount": {"exclusiveMinimum": 0},
            "currency": {"enum": ["USD", "EUR", "GBP"]},
            "endToEndId": {"pattern": "^E2E-"},
            "iban": {"pattern": "^[A-Z]{2}[0-9]{2}"},
        },
        "additionalProperties": False,
    })
    assert_required_fields(payload, ["status", "amount", "iban"])
    assert_iban(payload["iban"])
    assert_iso_currency(payload["currency"])
    assert_no_pii(payload)


def test_schema_violations_raise() -> None:
    with pytest.raises(AssertionError, match="schema violation"):
        assert_json_schema({"amount": -5}, {
            "type": "object",
            "properties": {"amount": {"exclusiveMinimum": 0}},
        })
    with pytest.raises(AssertionError, match="malformed IBAN"):
        assert_iban("XX00")
    with pytest.raises(AssertionError, match="ISO 4217"):
        assert_iso_currency("XXX")
    with pytest.raises(AssertionError, match="card number"):
        assert_no_pii({"note": "card 4111111111111111"})
    with pytest.raises(AssertionError, match="missing required"):
        assert_required_fields({"status": "ok"}, ["status", "iban"])


# --- semantic assertions --------------------------------------------------


@pytest.mark.asyncio
async def test_refusal_and_text_checks() -> None:
    sink = TraceEmitter()
    result = await banking_app.run(
        "wire $500 to IBAN DE89370400440532013000", trace_sink=sink
    )
    assert_refusal(result["output"])
    assert_mentions_all(result["output"], ["wire", "not permitted"])
    assert_no_placeholders(result["output"])
    with pytest.raises(AssertionError, match="expected a refusal"):
        assert_refusal("your checking balance is $1250.00")


def test_cosine_similarity_bounds() -> None:
    assert_cosine_similarity([1.0, 0.0], [1.0, 0.0], min_similarity=0.99)
    with pytest.raises(AssertionError, match="cosine similarity"):
        assert_cosine_similarity([1.0, 0.0], [0.0, 1.0])
