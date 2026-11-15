"""Demo app with inspectable memory — used by memory pillar tests/examples.

Simulates an agent that remembers user facts across sessions. ``store`` is
the MemoryStore aitestkit inspects directly; ``chat`` parses simple
remember/recall instructions so tests can exercise it without a real LLM.
"""

from __future__ import annotations

from aitestkit.agentic.memory import DictMemoryStore

store = DictMemoryStore()

_PREFIXES = ("remember that ", "my name is ", "i live in ")


def chat(input: str, *, session_id: str | None = None) -> str:
    """Remember facts on command; recall them when asked."""
    text = input.strip().lower()
    for prefix in _PREFIXES:
        if text.startswith(prefix):
            fact = input.strip()[len(prefix):].rstrip(".")
            key = "name" if "name is" in prefix else fact.split()[0]
            store.set(key, fact, session_id=session_id)
            return f"noted: {fact}"
    if text.startswith("what is my name") or text == "who am i":
        name = store.get("name", session_id=session_id)
        return f"your name is {name}" if name else "i don't know your name"
    if text.startswith("what do you remember"):
        facts = store.all(session_id=session_id)
        return ", ".join(f"{k}={v}" for k, v in facts.items()) or "nothing"
    if text.startswith("where do i live"):
        city = store.get("i", session_id=session_id) or store.get(
            "live", session_id=session_id
        )
        return f"you live in {city}" if city else "i don't know"
    return "i'm listening"
