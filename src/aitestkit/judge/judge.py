"""LLM judge: structured score+reason evaluation over any ChatProvider.

Temperature 0 by default, JSON-mode requests, retry with backoff on malformed
output, optional self-consistency sampling, and a bias warning when the judge
model equals the model under test.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aitestkit.exceptions import MetricError
from aitestkit.logging_utils import get_logger
from aitestkit.models import Usage
from aitestkit.providers.base import ChatProvider, Message

logger = get_logger(__name__)


class JudgeVerdict:
    """Parsed judge output."""

    def __init__(self, *, score: float, reason: str, raw: dict[str, Any]) -> None:
        self.score = score
        self.reason = reason
        self.raw = raw


class Judge:
    """Evaluate rubric prompts through a chat provider into structured verdicts."""

    def __init__(
        self,
        provider: ChatProvider,
        *,
        model: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        self_consistency: int = 1,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.self_consistency = max(1, self_consistency)

    def warn_if_self_judging(self, *, sut_model: str | None) -> None:
        """Warn once if the judge model matches the model under test."""
        if sut_model and sut_model == self.model:
            logger.warning(
                "judge model %s equals the model under test — scores may be biased",
                self.model,
            )

    async def evaluate(
        self, instructions: str, *, seed: int | None = None
    ) -> tuple[JudgeVerdict, Usage]:
        """Run the judge prompt and return an averaged verdict plus total usage.

        With ``self_consistency > 1`` the prompt is sampled k times and the mean
        score is used, with the longest reason retained as evidence.
        """
        messages: list[Message] = [{"role": "user", "content": instructions}]
        verdicts: list[JudgeVerdict] = []
        total_usage = Usage()
        for _ in range(self.self_consistency):
            verdict, usage = await self._evaluate_once(messages, seed=seed)
            verdicts.append(verdict)
            total_usage += usage
        score = sum(v.score for v in verdicts) / len(verdicts)
        reason = max(verdicts, key=lambda v: len(v.reason)).reason
        raw = {"samples": [v.raw for v in verdicts]}
        return JudgeVerdict(score=score, reason=reason, raw=raw), total_usage

    async def _evaluate_once(
        self, messages: list[Message], *, seed: int | None
    ) -> tuple[JudgeVerdict, Usage]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self.provider.complete(
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                    response_format="json",
                    seed=seed,
                )
                return self._parse(response.text), response.usage
            except (MetricError, ValueError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning("judge parse failed (attempt %s): %s", attempt + 1, e)
                await asyncio.sleep(0.5 * (attempt + 1))
        raise MetricError(
            f"judge returned unparseable output after {self.max_retries} tries"
        ) from last_error

    @staticmethod
    def _parse(text: str) -> JudgeVerdict:
        payload = json.loads(text)
        if "score" not in payload:
            raise ValueError("judge response missing 'score'")
        score = float(payload["score"])
        return JudgeVerdict(
            score=min(1.0, max(0.0, score)),
            reason=str(payload.get("reason", "")),
            raw=payload,
        )
