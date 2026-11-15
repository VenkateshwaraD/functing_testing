"""Security regression tests: code-loading guards, env isolation, escaping,
redaction coverage, and stream truncation.

Each test maps to a reviewed defect:
- --no-allow-python-targets bypass via memory.store / tools.*.function /
  providers.*.impl / 'module:Class' provider names.
- unset ${VAR} forwarded as a literal header value.
- command/mcp child processes inheriting the full parent environment.
- GitHub workflow-command injection via unescaped case_id.
- IBAN/PAN/email/account PII + stale unredacted results.jsonl lines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aitestkit.config import ProviderEntry, SuiteConfig, TargetConfig
from aitestkit.exceptions import ConfigError
from aitestkit.models import CaseResult, CaseStatus, RunMetadata, TestRun
from aitestkit.redact import Redactor
from aitestkit.targets.base import child_env, expand_env
from aitestkit.targets.command_target import CommandTarget
from aitestkit.targets.factory import build_target
from aitestkit.targets.http_target import HttpTarget
from aitestkit.targets.tool_registry import build_registry

# ---- ${VAR} expansion ------------------------------------------------------


def test_expand_env_set_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TOKEN", "abc123")
    assert expand_env("Bearer ${MY_TOKEN}") == "Bearer abc123"


def test_expand_env_unset_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_VAR", raising=False)
    with pytest.raises(ConfigError, match="MISSING_VAR"):
        expand_env("Bearer ${MISSING_VAR}")


def test_http_target_unset_env_var_fails() -> None:
    with pytest.raises(ConfigError, match="NOT_SET_KEY"):
        HttpTarget(
            url="https://x.example.com",
            headers={"Authorization": "Bearer ${NOT_SET_KEY}"},
        )


# ---- child env isolation ---------------------------------------------------


def test_child_env_excludes_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value-1234567890")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = child_env()
    assert "OPENAI_API_KEY" not in env
    assert env.get("PATH") == "/usr/bin"


def test_child_env_explicit_extra_and_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value-1234567890")
    monkeypatch.setenv("MY_KEY", "passed-through")
    env = child_env({"MY_KEY": "${MY_KEY}", "OTHER": "static"})
    assert env["MY_KEY"] == "passed-through"
    assert env["OTHER"] == "static"
    assert "OPENAI_API_KEY" not in env


def test_child_env_unset_extra_fails() -> None:
    with pytest.raises(ConfigError, match="GHOST_VAR"):
        child_env({"X": "${GHOST_VAR}"})


async def test_command_target_does_not_leak_parent_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value-1234567890")
    target = CommandTarget(
        command=[
            sys.executable,
            "-c",
            "import os,json; "
            "print(json.dumps({'output': os.environ.get('OPENAI_API_KEY','-')}))",
        ],
        allowed=True,
    )
    from aitestkit.models import TestCase

    out = await target.invoke(TestCase(id="c", input="x"))
    assert out.output == "-"


# ---- python-target guard coverage ------------------------------------------


def _suite(**kwargs) -> SuiteConfig:
    base = {
        "name": "s",
        "target": {"type": "http", "url": "https://x.example.com"},
        "datasets": ["d.yaml"],
        "metrics": [{"name": "exact_match"}],
    }
    base.update(kwargs)
    return SuiteConfig.model_validate(base)


def test_no_python_targets_blocks_memory_store() -> None:
    from aitestkit.cli import _reject_python_code_paths

    suite = _suite(memory={"store": "demo_app:store"})
    with pytest.raises(ConfigError, match="memory.store"):
        _reject_python_code_paths(suite)


def test_no_python_targets_blocks_tool_functions() -> None:
    from aitestkit.cli import _reject_python_code_paths

    suite = _suite(tools={"lookup": {"function": "demo_app:lookup"}})
    with pytest.raises(ConfigError, match="tools.lookup.function"):
        _reject_python_code_paths(suite)


def test_no_python_targets_blocks_provider_impl() -> None:
    from aitestkit.cli import _reject_python_code_paths

    suite = _suite(
        providers={
            "mine": ProviderEntry(
                provider="mine", model="m", impl="mod:Cls"
            )
        },
        judge={"provider": "mine", "model": "m"},
    )
    with pytest.raises(ConfigError, match="providers.mine"):
        _reject_python_code_paths(suite)


def test_no_python_targets_blocks_module_class_judge() -> None:
    from aitestkit.cli import _reject_python_code_paths

    suite = _suite(judge={"provider": "mymod:MyJudge", "model": "m"})
    with pytest.raises(ConfigError, match="judge.provider"):
        _reject_python_code_paths(suite)


def test_no_python_targets_allows_plain_http_suite() -> None:
    from aitestkit.cli import _reject_python_code_paths

    _reject_python_code_paths(_suite())  # no code paths — must not raise


def test_build_registry_blocked_without_python() -> None:
    with pytest.raises(ConfigError, match="allow-python-targets"):
        build_registry({"t": {"function": "math:sqrt"}}, allow_python=False)


def test_custom_provider_blocked_without_flag() -> None:
    from aitestkit.providers.registry import make_chat_provider, make_embedder

    with pytest.raises(ConfigError, match="allow-python-targets"):
        make_chat_provider("math:AnyClass", allow_custom=False)
    with pytest.raises(ConfigError, match="allow-python-targets"):
        make_embedder("math:AnyClass", allow_custom=False)
    with pytest.raises(ConfigError, match="allow-python-targets"):
        make_chat_provider("fake", impl="mod:Cls", allow_custom=False)


def test_python_target_still_blocked() -> None:
    cfg = TargetConfig(type="python", function="demo_app:answer")
    with pytest.raises(ConfigError):
        build_target(cfg, allow_python_targets=False)


# ---- GitHub annotation escaping --------------------------------------------


def test_github_annotation_escapes_case_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from aitestkit.reporting import github_actions

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    run = TestRun(
        suite_name="s",
        metadata=RunMetadata(),
        results=[
            CaseResult(
                case_id="evil\n::warning file=x::pwned",
                status=CaseStatus.FAILED,
                input="i",
                metrics=[],
            )
        ],
    )
    github_actions.emit(run, redactor=Redactor(), suite_path=None)
    out = capsys.readouterr().out
    # every emitted line must be a single annotation — no injected commands
    assert out.count("::error") == 1
    assert "\n::" not in out.strip()
    assert "%0A" in out


def test_escape_property_escapes_separators() -> None:
    from aitestkit.reporting.github_actions import _escape_property

    assert _escape_property("a:b,c\nd") == "a%3Ab%2Cc%0Ad"


# ---- redaction --------------------------------------------------------------


def test_redactor_covers_banking_pii() -> None:
    r = Redactor()
    text = (
        "IBAN DE89370400440532013000 card 4111 1111 1111 1111 "
        "mail max@example.com account number is 45211234 "
        "acct: 9988776655"
    )
    out = r.redact_text(text)
    for leaked in (
        "DE89370400440532013000",
        "4111 1111 1111 1111",
        "max@example.com",
        "45211234",
        "9988776655",
    ):
        assert leaked not in out, leaked


def test_redactor_account_pattern_no_false_positive() -> None:
    r = Redactor()
    # unlabeled short digit runs stay — amounts/years are not PII
    assert r.redact_text("transfer 4521 euros") == "transfer 4521 euros"


# ---- results.jsonl truncation -----------------------------------------------


def test_stream_open_truncates_stale_lines(tmp_path: Path) -> None:
    from aitestkit.reporting.stream import ResultsStream

    path = tmp_path / "results.jsonl"
    path.write_text('{"type":"case_result","stale":true}\n', encoding="utf-8")
    stream = ResultsStream(path, redactor=Redactor())
    run = TestRun(suite_name="s", metadata=RunMetadata())
    stream.open(run)
    stream.close(run)
    lines = path.read_text().strip().splitlines()
    assert all(json.loads(line).get("type") != "case_result" for line in lines)
    assert json.loads(lines[0])["type"] == "run_header"


def test_stream_append_mode_still_available(tmp_path: Path) -> None:
    from aitestkit.reporting.stream import ResultsStream

    path = tmp_path / "results.jsonl"
    path.write_text('{"type":"kept"}\n', encoding="utf-8")
    stream = ResultsStream(path, redactor=Redactor())
    stream.open(TestRun(suite_name="s", metadata=RunMetadata()), mode="a")
    stream.close(TestRun(suite_name="s", metadata=RunMetadata()))
    assert json.loads(path.read_text().splitlines()[0])["type"] == "kept"
