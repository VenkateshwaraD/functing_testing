"""Async run engine: invoke targets, evaluate metrics, produce a TestRun.

Error taxonomy: ProviderError/TargetError -> case ``error`` (infrastructure,
not quality); missing required fields -> metric ``skipped``; a mix of
pass/fail across --repeat runs -> ``flaky``. Results stream to disk
per-case through ``on_case_result`` so a crash preserves progress.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aitestkit.exceptions import (
    AitestkitError,
    BudgetExceededError,
    MetricError,
    TargetError,
)
from aitestkit.logging_utils import get_logger
from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.models import (
    CaseResult,
    CaseStatus,
    ConversationCase,
    MetricResult,
    TargetOutput,
    TestCase,
    TestRun,
    Usage,
)
from aitestkit.targets.base import TargetAdapter

if TYPE_CHECKING:
    from aitestkit.config import PromptVariant

logger = get_logger(__name__)

OnCaseResult = Callable[[CaseResult], Awaitable[None] | None]


class RunEngine:
    """Executes cases through a target and scores them with metrics."""

    def __init__(
        self,
        *,
        target: TargetAdapter,
        metrics: list[BaseMetric],
        ctx: MetricContext,
        concurrency: int = 8,
        case_timeout_seconds: float = 300,
        repeat: int = 1,
        max_cost_usd: float | None = None,
        models: list[str] | None = None,
        tool_mocks: dict[str, Any] | None = None,
        conversations: list[ConversationCase] | None = None,
        tool_registry: Any | None = None,
        prompts: list[PromptVariant] | None = None,
    ) -> None:
        self.target = target
        self.metrics = metrics
        self.ctx = ctx
        self.concurrency = concurrency
        self.case_timeout = case_timeout_seconds
        self.repeat = max(1, repeat)
        self.max_cost_usd = max_cost_usd
        self.models: list[str | None] = list(models) if models else [None]
        self.tool_mocks = tool_mocks
        self.conversations = conversations or []
        self.tool_registry = tool_registry
        self.prompts: list[PromptVariant | None] = (
            list(prompts) if prompts else [None]
        )
        self._spent = Usage()
        self._budget_hit = False
        self._cancelled = False

    def cancel(self) -> None:
        """Stop scheduling new case work; in-flight cases finish normally."""
        self._cancelled = True

    async def run(
        self,
        cases: list[TestCase],
        *,
        on_case_result: OnCaseResult | None = None,
        skip_ids: set[str] | None = None,
    ) -> tuple[TestRun, bool]:
        """Run all cases; returns (run, budget_exceeded).

        ``skip_ids`` excludes already-completed case ids (``--resume``).
        """
        skip_ids = skip_ids or set()
        semaphore = asyncio.Semaphore(self.concurrency)
        test_run = TestRun(run_id="", results=[])

        async def worker(
            case: TestCase, model_spec: str | None, prompt: PromptVariant | None
        ) -> list[CaseResult]:
            async with semaphore:
                if self._cancelled:
                    return [self._skipped_result(case, "run cancelled (SIGINT)")]
                if self._budget_hit:
                    return [self._skipped_result(case, "budget cap reached earlier")]
                try:
                    return await self._run_case(
                        case, model_spec=model_spec, prompt=prompt
                    )
                except BudgetExceededError:
                    self._budget_hit = True
                    return [
                        self._skipped_result(
                            case, f"skipped: budget cap {self.max_cost_usd} USD hit"
                        )
                    ]

        tasks = [
            worker(case, model_spec, prompt)
            for case in cases
            for model_spec in self.models
            for prompt in self.prompts
            if _effective_id(case, model_spec, _prompt_name(prompt)) not in skip_ids
        ]
        for gathered in asyncio.as_completed(tasks):
            for result in await gathered:
                test_run.results.append(result)
                if on_case_result is not None:
                    maybe_awaitable = on_case_result(result)
                    if maybe_awaitable is not None:
                        await maybe_awaitable
        for conversation in self.conversations:
            for result in await self._run_conversation(conversation):
                test_run.results.append(result)
                if on_case_result is not None:
                    maybe_awaitable = on_case_result(result)
                    if maybe_awaitable is not None:
                        await maybe_awaitable
        return test_run, self._budget_hit

    async def _run_conversation(self, conversation: ConversationCase) -> list[CaseResult]:
        """Run a multi-turn case sequentially; turns share a session_id."""
        from aitestkit.tracing.emitter import TraceEmitter

        session = conversation.session_id or conversation.id
        results: list[CaseResult] = []
        for index, turn in enumerate(conversation.turns):
            case = TestCase(
                id=f"{conversation.id}/turn-{index}",
                input=turn.input,
                expected_output=turn.expected_output,
                tags=list(dict.fromkeys(turn.tags + conversation.tags)),
                metadata={
                    **conversation.metadata,
                    "_conversation": conversation.id,
                    "_turn": index,
                    "_final_turn": index == len(conversation.turns) - 1,
                    "session_id": session,
                },
            )
            emitter = TraceEmitter()
            case_id = case.id or "turn"
            try:
                output = await asyncio.wait_for(
                    self.target.invoke(
                        case,
                        session_id=session,
                        trace_sink=emitter,
                        tool_registry=self.tool_registry,
                    ),
                    timeout=self.case_timeout,
                )
            except TimeoutError:
                results.append(
                    self._error_result(case, case_id, "turn timed out")
                )
                continue
            except (TargetError, AitestkitError) as e:
                results.append(self._error_result(case, case_id, str(e)))
                continue
            if output.trajectory is None and len(emitter):
                output.trajectory = emitter.trajectory
            self._spend(output.usage)
            result = await self._evaluate_metrics(case, output, case_id)
            result.session_id = session
            results.append(result)
        return results

    def _skipped_result(self, case: TestCase, reason: str) -> CaseResult:
        return CaseResult(
            case_id=case.id or "case",
            status=CaseStatus.SKIPPED,
            input=case.input,
            error=reason,
            tags=case.tags,
        )

    async def _run_case(
        self,
        case: TestCase,
        *,
        model_spec: str | None,
        prompt: PromptVariant | None = None,
    ) -> list[CaseResult]:
        provider, model = (
            (model_spec.split(":", 1) + [None])[:2] if model_spec else (None, None)
        )
        outcomes: list[CaseResult] = []
        for _ in range(self.repeat):
            outcomes.append(
                await self._run_once(
                    case, provider=provider, model=model, prompt=prompt
                )
            )
        if self.repeat == 1:
            return outcomes
        return [self._collapse_repeats(case, outcomes, model_spec)]

    def _collapse_repeats(
        self, case: TestCase, outcomes: list[CaseResult], model_spec: str | None
    ) -> CaseResult:
        """Merge repeated outcomes; mixed pass/fail marks the case flaky."""
        passes = sum(o.status == CaseStatus.PASSED for o in outcomes)
        fails = sum(o.status == CaseStatus.FAILED for o in outcomes)
        errors = sum(o.status == CaseStatus.ERROR for o in outcomes)
        if errors == len(outcomes):
            status = CaseStatus.ERROR
        elif fails and passes:
            status = CaseStatus.FLAKY
        elif fails:
            status = CaseStatus.FAILED
        elif passes:
            status = CaseStatus.PASSED
        else:
            status = CaseStatus.SKIPPED
        base = outcomes[0]
        base.status = status
        base.repeat = {
            "runs": len(outcomes),
            "pass_rate": passes / len(outcomes),
            "statuses": [o.status.value for o in outcomes],
            "model": model_spec,
        }
        base.metrics = outcomes[0].metrics
        base.judge_usage = _sum_usage(o.judge_usage for o in outcomes)
        base.sut_usage = _sum_usage(o.sut_usage for o in outcomes)
        return base

    async def _run_once(
        self,
        case: TestCase,
        *,
        provider: str | None,
        model: str | None,
        prompt: PromptVariant | None = None,
    ) -> CaseResult:
        case_id = _effective_id(
            case,
            f"{provider}:{model}" if model else None,
            _prompt_name(prompt),
        )
        from aitestkit.tracing.emitter import TraceEmitter

        emitter = TraceEmitter()
        rendered_prompt = (
            _render_prompt(prompt, case.input) if prompt is not None else None
        )
        try:
            output = await asyncio.wait_for(
                self.target.invoke(
                    case,
                    model=model,
                    provider=provider,
                    tool_mocks=self.tool_mocks,
                    session_id=case.metadata.get("session_id"),
                    trace_sink=emitter,
                    tool_registry=self.tool_registry,
                    prompt=rendered_prompt,
                    prompt_name=_prompt_name(prompt),
                ),
                timeout=self.case_timeout,
            )
        except TimeoutError:
            return self._error_result(case, case_id, f"target timed out after {self.case_timeout}s")
        except (TargetError, AitestkitError) as e:
            return self._error_result(case, case_id, str(e))
        if output.error:
            return self._error_result(case, case_id, output.error)
        if output.trajectory is None and len(emitter):
            output.trajectory = emitter.trajectory
        self._spend(output.usage)
        result = await self._evaluate_metrics(case, output, case_id)
        if prompt is not None:
            result.metadata["prompt_name"] = _prompt_name(prompt)
            result.metadata["rendered_prompt"] = rendered_prompt
        self._spend(result.judge_usage)
        return result

    def _spend(self, usage: Usage) -> None:
        self._spent += usage
        if self.max_cost_usd is not None and self._spent.cost_usd > self.max_cost_usd:
            raise BudgetExceededError(
                f"run budget of ${self.max_cost_usd} exceeded "
                f"(spent ${self._spent.cost_usd:.4f})"
            )

    def _error_result(self, case: TestCase, case_id: str, error: str) -> CaseResult:
        return CaseResult(
            case_id=case_id,
            status=CaseStatus.ERROR,
            input=case.input,
            error=error,
            tags=case.tags,
        )

    async def _evaluate_metrics(
        self, case: TestCase, output: TargetOutput, case_id: str
    ) -> CaseResult:
        results: list[MetricResult] = []
        for metric in self.metrics:
            try:
                metric_result = await metric.evaluate(case, output, self.ctx)
            except MetricError as e:
                metric_result = MetricResult.error(metric=metric.name, reason=str(e))
            except Exception as e:  # noqa: BLE001 — metric bug must not kill the run
                logger.warning("metric %s crashed: %s", metric.name, e)
                metric_result = MetricResult.error(
                    metric=metric.name, reason=f"metric error: {e}"
                )
            results.append(metric_result)
        scored = [r for r in results if r.status.value == "scored"]
        if scored and all(r.passed for r in scored):
            status = CaseStatus.PASSED
        elif scored:
            status = CaseStatus.FAILED
        else:
            status = CaseStatus.SKIPPED
        judge_usage = _sum_usage(r.usage for r in results)
        return CaseResult(
            case_id=case_id,
            status=status,
            input=case.input,
            actual_output=output.output,
            metrics=results,
            trajectory=output.trajectory,
            latency_ms=output.latency_ms,
            judge_usage=judge_usage,
            sut_usage=output.usage,
            tags=case.tags,
        )


def _sum_usage(usages: Any) -> Usage:
    total = Usage()
    for usage in usages:
        total += usage
    return total


def _prompt_name(prompt: PromptVariant | None) -> str | None:
    return prompt.name if prompt is not None else None


def _render_prompt(prompt: PromptVariant, input_text: str) -> str:
    """Substitute ``{input}`` in a PromptVariant template."""
    try:
        return prompt.template.format(input=input_text)
    except (KeyError, IndexError) as e:
        raise TargetError(
            f"prompt {prompt.name!r}: bad template placeholder: {e}"
        ) from e


def _effective_id(
    case: TestCase, model_spec: str | None, prompt_name: str | None = None
) -> str:
    """The case_id used in results — matches what resume skips on."""
    base = case.id or "case"
    suffix = "".join(
        f"[{part}]" for part in (model_spec, prompt_name) if part
    )
    return f"{base}{suffix}"
