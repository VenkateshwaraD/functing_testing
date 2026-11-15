"""pytest collection plugin: *.aitestkit.yaml files become test items."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_aitestkit_yaml_collection(tmp_path: Path) -> None:
    """A passing and failing case become two pytest items."""
    suite_file = tmp_path / "demo.aitestkit.yaml"
    suite_file.write_text(
        """
target: {type: python, function: "demo_app:answer"}
metrics:
  - name: contains
    params: {substrings: ["hello"]}
cases:
  - id: greeting
    input: "say hello"
  - id: failing
    input: "unknown question"
"""
    )
    examples = Path(__file__).parent.parent / "examples"
    if str(examples) not in sys.path:
        sys.path.insert(0, str(examples))
    code = pytest.main(
        [str(suite_file), "-q", "--no-header", "-p", "no:cacheprovider"]
    )
    assert code == pytest.ExitCode.TESTS_FAILED


def test_aitestkit_yaml_all_pass(tmp_path: Path) -> None:
    suite_file = tmp_path / "ok.aitestkit.yaml"
    suite_file.write_text(
        """
target: {type: python, function: "demo_app:answer"}
metrics:
  - name: contains
    params: {substrings: ["hello"]}
cases:
  - id: greeting
    input: "say hello"
"""
    )
    examples = Path(__file__).parent.parent / "examples"
    if str(examples) not in sys.path:
        sys.path.insert(0, str(examples))
    code = pytest.main(
        [str(suite_file), "-q", "--no-header", "-p", "no:cacheprovider"]
    )
    assert code == pytest.ExitCode.OK
