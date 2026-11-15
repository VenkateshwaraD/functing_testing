"""Prompt simulation metric — deterministic checks on rendered prompts.

Catches template bugs before tokens are spent: unresolved ``{placeholder}``
tokens, banned phrases, and prompt length over a token budget. Runs without
a judge or provider.
"""

from __future__ import annotations

import re

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.registry import register
from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase
from aitestkit.prompts import render_prompt, unresolved_placeholders
from aitestkit.rag.token_counting import count_tokens


@register
class PromptSim(BaseMetric):
    """Render ``template`` against the case and lint the result.

    Params:
        template: prompt template with ``{input}``/``{metadata_key}``
        banned: phrases that must not appear in the rendered prompt
            (word-boundary matched — 'dan' does not hit 'dangerous')
        max_tokens: token budget for the rendered prompt
    """

    name = "prompt_sim"
    default_threshold = 1.0

    def __init__(
        self,
        *,
        template: str = "{input}",
        banned: list[str] | None = None,
        max_tokens: int | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self.template = template
        self.banned = banned or []
        self.max_tokens = max_tokens

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        rendered = render_prompt(self.template, case)
        problems: list[str] = []
        leftovers = unresolved_placeholders(rendered)
        if leftovers:
            problems.append(f"unresolved placeholders: {leftovers}")
        hits = [
            b
            for b in self.banned
            if re.search(rf"(?<!\w){re.escape(b)}(?!\w)", rendered, re.IGNORECASE)
        ]
        if hits:
            problems.append(f"banned phrases present: {hits}")
        tokens = count_tokens(rendered)
        if self.max_tokens is not None and tokens > self.max_tokens:
            problems.append(
                f"prompt over budget: {tokens} > {self.max_tokens} tokens"
            )
        passed = not problems
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="; ".join(problems),
            details={
                "rendered_prompt": rendered,
                "unresolved": leftovers,
                "banned_hits": hits,
                "tokens": tokens,
            },
        )
