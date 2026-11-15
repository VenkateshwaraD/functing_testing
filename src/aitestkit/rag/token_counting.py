"""Token counting with vendored tiktoken data — works fully offline.

Encodings ship inside the package (``tiktoken_data/*.tiktoken``) so no
download happens on first use. If a model isn't mapped, falls back to a
chars/4 heuristic — counts for non-OpenAI models are approximate anyway.
"""

from __future__ import annotations

import base64
import functools
import logging
from importlib import resources
from typing import Any

logger = logging.getLogger(__name__)

_ENCODINGS = {"cl100k_base", "o200k_base"}
_MODEL_TO_ENCODING = {
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "text-embedding-3": "cl100k_base",
    "ada": "cl100k_base",
}


@functools.lru_cache(maxsize=4)
def _encoding(name: str) -> Any:
    """Build a tiktoken Encoding from vendored merge ranks."""
    import tiktoken

    blob = resources.files("aitestkit.tiktoken_data").joinpath(
        f"{name}.tiktoken"
    ).read_bytes()
    mergeable_ranks = {}
    for line in blob.splitlines():
        if not line.strip():
            continue
        token_b64, _, rank = line.partition(b" ")
        mergeable_ranks[base64.b64decode(token_b64)] = int(rank)
    pat_str = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    return tiktoken.Encoding(
        name=name,
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )


def encoding_for_model(model: str | None) -> str:
    """Map a model name to a vendored encoding (best-effort)."""
    if not model:
        return "o200k_base"
    lowered = model.lower()
    for prefix, encoding in _MODEL_TO_ENCODING.items():
        if prefix in lowered:
            return encoding
    return "cl100k_base"


def count_tokens(text: str, *, model: str | None = None) -> int:
    """Token count for text; vendored-tiktoken first, heuristic fallback."""
    try:
        encoding = _encoding(encoding_for_model(model))
        return len(encoding.encode(text))
    except Exception as e:  # noqa: BLE001 — any tokenizer failure degrades gracefully
        logger.debug("tiktoken unavailable, using approximation: %s", e)
        return _approx_tokens(text)


def _approx_tokens(text: str) -> int:
    """chars/4 heuristic — good enough for context-limit checks."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def count_message_tokens(
    messages: list[dict[str, str]], *, model: str | None = None
) -> int:
    """Approximate chat-message overhead per provider conventions."""
    total = 0
    for message in messages:
        total += count_tokens(message.get("content", ""), model=model) + 4
    return total + 2
