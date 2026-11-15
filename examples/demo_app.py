"""Tiny demo app used by the vertical-slice test and example suites."""

from __future__ import annotations

ANSWERS = {
    "say hello": "hello",
    "what is 2+2": "4",
    "capital of france": "Paris",
}


def answer(input: str, **kwargs: object) -> str:
    """Return a canned answer — stands in for a real AI app."""
    return ANSWERS.get(input.strip().lower(), "I don't know")


SEEN_PROMPTS: list[tuple[str | None, str | None]] = []


def prompt_echo(
    input: str,
    *,
    prompt: str | None = None,
    prompt_name: str | None = None,
) -> str:
    """Record which prompt variant each call received (matrix tests)."""
    SEEN_PROMPTS.append((prompt_name, prompt))
    return f"echo {input}"
