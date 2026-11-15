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
