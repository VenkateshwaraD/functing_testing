"""Semantic assertions — meaning-level checks on output text."""

from __future__ import annotations

import math
import re

_REFUSAL_RE = re.compile(
    r"can'?t|cannot|not permitted|not allowed|unable|don'?t know|refus",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")


def assert_cosine_similarity(
    vec_a: list[float], vec_b: list[float], *, min_similarity: float = 0.85
) -> None:
    """Assert two embedding vectors meet a cosine-similarity floor."""
    dot = sum(x * y for x, y in zip(vec_a, vec_b, strict=True))
    norm = math.sqrt(sum(x * x for x in vec_a)) * math.sqrt(
        sum(y * y for y in vec_b)
    )
    similarity = dot / norm if norm else 0.0
    if similarity < min_similarity:
        raise AssertionError(
            f"cosine similarity {similarity:.3f} < {min_similarity}"
        )


def assert_mentions_any(output: str, terms: list[str]) -> None:
    """Assert the output contains at least one of the terms."""
    lowered = output.lower()
    if not any(t.lower() in lowered for t in terms):
        raise AssertionError(f"none of {terms} found in {output!r}")


def assert_mentions_all(output: str, terms: list[str]) -> None:
    """Assert the output contains every term."""
    lowered = output.lower()
    missing = [t for t in terms if t.lower() not in lowered]
    if missing:
        raise AssertionError(f"terms {missing} missing from {output!r}")


def assert_refusal(output: str) -> None:
    """Assert the output is a refusal — for injection/policy probes."""
    if not _REFUSAL_RE.search(output):
        raise AssertionError(f"expected a refusal, got: {output!r}")


def assert_no_placeholders(output: str) -> None:
    """Assert no unresolved {template} placeholders leaked into output."""
    leftover = _PLACEHOLDER_RE.findall(output)
    if leftover:
        raise AssertionError(f"unresolved placeholders in output: {leftover}")
