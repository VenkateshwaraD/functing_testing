"""Cost estimation: model price table (USD per 1M tokens, input/output).

Approximate and versioned — enough for budgets, cost reports, and nightly
spend caps. Unknown models return zero rather than guessing.
"""

from __future__ import annotations

from typing import Any

# (input $/1M tokens, output $/1M tokens) — public list prices, prefix-matched.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-ada-002": (0.10, 0.0),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-1": (15.00, 75.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "mistral-large": (2.00, 6.00),
    "mistral-small": (0.20, 0.60),
    "llama3": (0.0, 0.0),  # local/ollama — free
}


def estimate_cost(
    model: str, *, prompt_tokens: int, completion_tokens: int
) -> float:
    """USD cost for one call; 0.0 for unknown models (logs nothing — cheap)."""
    price = _price_for(model)
    if price is None:
        return 0.0
    input_rate, output_rate = price
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


def _price_for(model: str) -> tuple[float, float] | None:
    """Longest-prefix match so 'gpt-4o-2024-08-06' hits 'gpt-4o'."""
    best = _match(model, lambda m, p: m.startswith(p))
    return best if best is not None else _match(model, lambda m, p: p in m)


def _match(
    model: str, pred: Any
) -> tuple[float, float] | None:
    """Longest entry whose key satisfies ``pred`` — the fallback (substring)
    catches gateway/Bedrock names like 'global.anthropic.claude-sonnet-4-6'."""
    best: tuple[int, tuple[float, float]] | None = None
    for prefix, price in MODEL_PRICES.items():
        if pred(model, prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), price)
    return best[1] if best else None
