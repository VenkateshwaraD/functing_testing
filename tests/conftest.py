"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make example apps importable as python targets in tests.
REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
APPS_DIR = EXAMPLES_DIR / "apps"
for _p in (str(REPO_ROOT), str(EXAMPLES_DIR), str(APPS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def fake_provider():
    from aitestkit.providers.fake import FakeChatProvider

    return FakeChatProvider()
