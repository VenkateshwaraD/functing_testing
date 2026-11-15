"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make example apps importable as python targets in tests.
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))


@pytest.fixture
def fake_provider():
    from aitestkit.providers.fake import FakeChatProvider

    return FakeChatProvider()
