"""Demo funds-transfer agent — deterministic, fully offline.

Models a banking assistant with tool access: balance lookup, FX rates,
internal transfers (confirmation-gated) and external wires (forbidden by
policy). Emits trajectory events via trace_sink and routes every tool call
through the injected ToolRegistry, so agentic metrics score *behavior* —
not just the final answer. Swap ``function:`` in the suites for your real
agent module to test it the same way.
"""

from __future__ import annotations

import re
from typing import Any

_BALANCES = {"checking": 1250.00, "savings": 8300.50}
_RATES = {"eur": 0.92, "gbp": 0.79, "inr": 83.10}


def get_balance(account: str) -> dict:
    return {"account": account, "balance_usd": _BALANCES.get(account, 0.0)}


def get_exchange_rate(currency: str) -> dict:
    return {"currency": currency.upper(), "rate": _RATES.get(currency.lower(), 1.0)}


def transfer_funds(from_account: str, to_account: str, amount: float) -> dict:
    return {
        "status": "posted",
        "from": from_account,
        "to": to_account,
        "amount": amount,
        "confirmation_id": "TX-78231",
    }


def wire_external(iban: str, amount: float) -> dict:
    return {"status": "wired", "iban": iban, "amount": amount}


_TOOL_IMPLS = {
    "get_balance": get_balance,
    "get_exchange_rate": get_exchange_rate,
    "transfer_funds": transfer_funds,
    "wire_external": wire_external,
}


class _NullSink:
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: None


_AMOUNT = re.compile(r"\$(\d+(?:\.\d+)?)")


async def run(
    input: str,
    *,
    trace_sink: Any | None = None,
    tool_registry: Any | None = None,
    **_: Any,
) -> dict:
    """Classify intent -> (confirm if destructive) -> call tool -> answer."""
    sink = trace_sink or _NullSink()
    text = input.strip().lower()
    calls: list[dict] = []

    async def _use(name: str, args: dict) -> Any:
        sink.emit("action", text=f"use tool: {name}")
        if tool_registry is not None:
            result = await tool_registry.invoke(name, args)
        else:
            result = _TOOL_IMPLS[name](**args)
        sink.tool_call(name, args=args, result=result)
        calls.append({"name": name, "args": args})
        return result

    sink.thought(f"classifying intent: {input}")

    # External wires — forbidden by policy; refuse without calling the tool.
    if "wire" in text or "iban" in text:
        sink.thought("external wire requested — policy forbids it")
        answer = "External wire transfers are not permitted."
        sink.message("assistant", answer)
        return {"output": answer, "tool_calls": calls}

    # Internal transfer — destructive: confirmation precedes the tool call.
    if any(word in text for word in ("transfer", "send", "pay")):
        match = _AMOUNT.search(text)
        amount = float(match.group(1)) if match else 0.0
        src, dst = (
            ("savings", "checking") if "to checking" in text else ("checking", "savings")
        )
        sink.thought(f"internal transfer {src} -> {dst}, ${amount:.2f}")
        sink.message("confirmation", "customer approved transfer in-session")
        try:
            res = await _use(
                "transfer_funds",
                {"from_account": src, "to_account": dst, "amount": amount},
            )
            calls[-1]["requires_confirmation"] = True
        except Exception as e:  # noqa: BLE001 — degrade gracefully (resilience demo)
            sink.emit("observation", text=f"tool error: {e}")
            answer = "The payment service is unavailable — please retry shortly."
            sink.message("assistant", answer)
            return {"output": answer, "tool_calls": calls}
        answer = (
            f"Transferred ${res['amount']:.2f} from {res['from']} to "
            f"{res['to']}. Confirmation {res['confirmation_id']}."
        )
        sink.message("assistant", answer)
        return {"output": answer, "tool_calls": calls}

    # FX rate lookup.
    if "rate" in text or "exchange" in text:
        currency = "gbp" if "gbp" in text else "eur"
        res = await _use("get_exchange_rate", {"currency": currency})
        answer = f"1 USD = {res['rate']} {res['currency']}."
        sink.message("assistant", answer)
        return {"output": answer, "tool_calls": calls}

    # Balance lookup.
    if "balance" in text:
        account = "savings" if "savings" in text else "checking"
        res = await _use("get_balance", {"account": account})
        answer = f"Your {res['account']} balance is ${res['balance_usd']:.2f}."
        sink.message("assistant", answer)
        return {"output": answer, "tool_calls": calls}

    sink.thought("no banking intent matched")
    answer = f"I can't help with {input!r}."
    sink.message("assistant", answer)
    return {"output": answer, "tool_calls": calls}
