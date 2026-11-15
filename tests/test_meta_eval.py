"""Meta-evaluation: do the metrics discriminate on labeled data?

Faithfulness is scored via real claim decomposition, so a scripted fake
judge can't validate discrimination — that needs a live model. These tests
run only when a provider key is present (nightly-live CI); locally they
verify the labeled set loads and the plumbing resolves.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from aitestkit.judge.judge import Judge
from aitestkit.metrics.base import MetricContext
from aitestkit.metrics.rag import Faithfulness
from aitestkit.models import TargetOutput, TestCase

META_EVAL = (
    Path(__file__).parent.parent
    / "examples"
    / "meta_eval"
    / "faithfulness_labeled.yaml"
)


def _load_samples() -> list[dict]:
    data = yaml.safe_load(META_EVAL.read_text())
    assert data["schema_version"] == 1
    return data["samples"]


def test_labeled_set_loads() -> None:
    samples = _load_samples()
    labels = {s["label"] for s in samples}
    assert labels == {"faithful", "unfaithful"}
    assert len(samples) >= 6


@pytest.mark.live
async def test_faithfulness_discriminates() -> None:
    """Known-faithful answers must outscore known-unfaithful ones.

    Requires OPENAI_API_KEY; skipped in offline/replay CI.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — live meta-eval only")
    from aitestkit.providers.registry import make_chat_provider

    provider = make_chat_provider("openai")
    ctx = MetricContext(
        judge=Judge(provider, model=os.environ.get("JUDGE_MODEL", "gpt-4o-mini"))
    )
    scores: dict[str, list[float]] = {"faithful": [], "unfaithful": []}
    for sample in _load_samples():
        case = TestCase(
            id=sample["id"],
            input="answer the question",
            context=[sample["context"]],
        )
        output = TargetOutput(output=sample["answer"])
        result = await Faithfulness().measure(case, output, ctx)
        assert result.score is not None, f"{sample['id']}: {result.reason}"
        scores[sample["label"]].append(result.score)
    faithful_mean = sum(scores["faithful"]) / len(scores["faithful"])
    unfaithful_mean = sum(scores["unfaithful"]) / len(scores["unfaithful"])
    assert faithful_mean > unfaithful_mean, (
        f"metric does not discriminate: faithful={faithful_mean:.2f} "
        f"vs unfaithful={unfaithful_mean:.2f}"
    )
    assert faithful_mean >= 0.8
    assert unfaithful_mean <= 0.5
