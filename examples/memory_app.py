"""Demo app with inspectable memory — used by memory pillar tests/examples.

Simulates an agent that remembers user facts across sessions. ``store`` is
the MemoryStore aitestkit inspects directly; ``chat`` parses simple
remember/recall instructions so tests can exercise it without a real LLM.
"""

from __future__ import annotations

from aitestkit.agentic.memory import DictMemoryStore

store = DictMemoryStore()

# Facts a real app would already know — pre-seeded so standalone cases
# don't race the seeding conversation.
store.set("name", "grace", session_id="sess-seeded")
store.set("live", "paris", session_id="sess-seeded")

_PREFIX_KEYS = {
    "my name is ": "name",
    "i live in ": "live",
    "remember that ": None,  # first word of the fact becomes the key
}


def chat(input: str, *, session_id: str | None = None) -> str:
    """Remember facts on command; recall them when asked."""
    text = input.strip().lower()
    for prefix, key in _PREFIX_KEYS.items():
        if text.startswith(prefix):
            fact = input.strip()[len(prefix):].rstrip(".")
            store.set(key or fact.split()[0], fact, session_id=session_id)
            return f"noted: {fact}"
    if text.startswith("what is my name") or text == "who am i":
        name = store.get("name", session_id=session_id)
        return f"your name is {name}" if name else "i don't know your name"
    if text.startswith("what do you remember"):
        facts = store.all(session_id=session_id)
        return ", ".join(f"{k}={v}" for k, v in facts.items()) or "nothing"
    if text.startswith("where do i live"):
        city = store.get("live", session_id=session_id)
        return f"you live in {city}" if city else "i don't know"
    return "i'm listening"
