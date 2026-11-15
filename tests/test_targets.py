"""Target tests: http (respx boundary), command (real subprocess), tools."""

from __future__ import annotations

import sys

import pytest

from aitestkit.exceptions import ConfigError, TargetError
from aitestkit.models import TestCase
from aitestkit.targets.command_target import CommandTarget
from aitestkit.targets.factory import build_target
from aitestkit.targets.http_target import HttpTarget
from aitestkit.targets.tool_registry import ToolFault, ToolRegistry


def _case() -> TestCase:
    return TestCase(id="c1", input="hello", expected_output="world")


async def test_http_target_extracts_fields(respx_mock) -> None:

    respx_mock.post("https://api.example.com/chat").respond(
        json={"data": {"answer": "paris", "docs": ["d1", "d2"]}}
    )
    target = HttpTarget(
        url="https://api.example.com/chat",
        request_template={"q": "{input}"},
        response_path="data.answer",
        context_path="data.docs",
    )
    output = await target.invoke(_case())
    assert output.output == "paris"
    assert output.retrieval_context == ["d1", "d2"]


async def test_http_target_4xx_is_target_error(respx_mock) -> None:
    respx_mock.post("https://api.example.com/chat").respond(status_code=500)
    target = HttpTarget(url="https://api.example.com/chat")
    with pytest.raises(TargetError, match="500"):
        await target.invoke(_case())


async def test_http_target_plain_text(respx_mock) -> None:
    respx_mock.post("https://api.example.com/chat").respond(text="raw answer")
    target = HttpTarget(url="https://api.example.com/chat")
    output = await target.invoke(_case())
    assert output.output == "raw answer"


async def test_command_target_requires_flag() -> None:
    with pytest.raises(ConfigError, match="--allow-command-targets"):
        CommandTarget(command=["echo", "hi"], allowed=False)


async def test_command_target_runs_argv() -> None:
    target = CommandTarget(
        command=[
            sys.executable,
            "-c",
            "import sys,json; c=json.load(sys.stdin); "
            "print(json.dumps({'output': 'echo:'+c['input']}))",
        ],
        allowed=True,
    )
    output = await target.invoke(_case())
    assert output.output == "echo:hello"


async def test_command_target_kills_on_timeout() -> None:
    target = CommandTarget(
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        allowed=True,
        timeout_seconds=0.2,
    )
    with pytest.raises(TargetError, match="killed"):
        await target.invoke(_case())


async def test_command_target_nonzero_exit() -> None:
    target = CommandTarget(
        command=[sys.executable, "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"],
        allowed=True,
    )
    with pytest.raises(TargetError, match="exited 3"):
        await target.invoke(_case())


def test_factory_gates_command_target() -> None:
    from aitestkit.config import TargetConfig

    config = TargetConfig(type="command", command=["echo", "hi"])
    with pytest.raises(ConfigError, match="--allow-command-targets"):
        build_target(config, allow_command_targets=False)
    target = build_target(config, allow_command_targets=True)
    assert target.name == "command"


def test_factory_http_and_python() -> None:
    from aitestkit.config import TargetConfig

    http_cfg = TargetConfig(type="http", url="https://x.example.com")
    assert build_target(http_cfg).name == "http"
    py_cfg = TargetConfig(type="python", function="demo_app:answer")
    assert build_target(py_cfg).name == "python"
    with pytest.raises(ConfigError, match="--allow-python-targets|not implemented|disabled"):
        build_target(py_cfg, allow_python_targets=False)


async def test_tool_registry_records_calls() -> None:
    registry = ToolRegistry()
    registry.register("search", lambda q: f"results for {q}")
    result = await registry.invoke("search", {"q": "paris"})
    assert result == "results for paris"
    assert registry.calls[0].name == "search"
    assert registry.calls[0].args == {"q": "paris"}


async def test_tool_registry_fault_error_then_ok() -> None:
    registry = ToolRegistry()
    registry.register("t", lambda: "fine")
    registry.inject_fault("t", ToolFault(error="db down", times=1))
    with pytest.raises(TargetError, match="db down"):
        await registry.invoke("t")
    assert await registry.invoke("t") == "fine"  # fault exhausted
    assert registry.calls[0].error is not None
    assert "db down" in registry.calls[0].error


async def test_tool_registry_fault_timeout() -> None:
    registry = ToolRegistry()
    registry.register("slow", lambda: "done")
    registry.inject_fault("slow", ToolFault(timeout_ms=10))
    with pytest.raises(TimeoutError, match="injected timeout"):
        await registry.invoke("slow")


async def test_tool_registry_mock_and_confirm() -> None:
    registry = ToolRegistry()
    registry.register("delete_all", lambda path: "deleted", requires_confirmation=True)
    registry.mock_result("delete_all", {"dry_run": True})
    assert registry.requires_confirmation("delete_all")
    result = await registry.invoke("delete_all", {"path": "/"})
    assert result == {"dry_run": True}
    assert registry.calls[0].requires_confirmation is True


async def test_tool_registry_unknown_tool() -> None:
    registry = ToolRegistry()
    with pytest.raises(ConfigError, match="not registered"):
        await registry.invoke("nope")
