"""MemoryStore protocol + reference implementations.

The SUT owns its memory; aitestkit inspects it *directly* through this
protocol — that's what separates real memory testing from re-asking the
model. Suites wire ``memory: {store: "module:store"}``; the object must
implement the protocol below.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryStore(Protocol):
    """Inspectable memory backend (short-term, long-term, episodic)."""

    def get(self, key: str, *, session_id: str | None = None) -> Any | None:
        """Fetch a stored value."""
        ...

    def set(self, key: str, value: Any, *, session_id: str | None = None) -> None:
        """Store a value."""
        ...

    def search(self, query: str, *, session_id: str | None = None) -> list[Any]:
        """Retrieve relevant entries for a query."""
        ...

    def all(self, *, session_id: str | None = None) -> dict[str, Any]:
        """Dump all entries for a session (or globally)."""
        ...

    def clear(self, *, session_id: str | None = None) -> None:
        """Forget everything (or one session)."""
        ...


class DictMemoryStore:
    """In-memory store — sessions partitioned by key prefix."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _bucket(self, session_id: str | None) -> dict[str, Any]:
        return self._data.setdefault(session_id or "__global__", {})

    def get(self, key: str, *, session_id: str | None = None) -> Any | None:
        return self._bucket(session_id).get(key)

    def set(self, key: str, value: Any, *, session_id: str | None = None) -> None:
        with self._lock:
            self._bucket(session_id)[key] = value

    def search(self, query: str, *, session_id: str | None = None) -> list[Any]:
        query_lower = query.lower()
        return [
            value
            for key, value in self._bucket(session_id).items()
            if query_lower in key.lower() or query_lower in str(value).lower()
        ]

    def all(self, *, session_id: str | None = None) -> dict[str, Any]:
        if session_id is not None:
            return dict(self._bucket(session_id))
        merged: dict[str, Any] = {}
        for bucket in self._data.values():
            merged.update(bucket)
        return merged

    def clear(self, *, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._data.clear()
            else:
                self._data.pop(session_id, None)


class JsonFileMemoryStore(DictMemoryStore):
    """File-backed store — survives process restart (persistence tests)."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        if path.exists():
            self._data = json.loads(path.read_text())

    def set(self, key: str, value: Any, *, session_id: str | None = None) -> None:
        super().set(key, value, session_id=session_id)
        self._save()

    def clear(self, *, session_id: str | None = None) -> None:
        super().clear(session_id=session_id)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, default=str))
