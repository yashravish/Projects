"""Unit test for `app.eval.run_evaluation`.

We bypass `GraphRunner` entirely by injecting a duck-typed object exposing
`.run(...)` and `.model`. This pins the runner's contract (calls .run once
per item, scores each, aggregates) without requiring a database.
"""
from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.agents.state import CritiqueResult, InquiryResult, TraceStep
from app.eval.dataset import GoldItem, GoldQuestionDataset
from app.eval.runner import run_evaluation


class _StubRunner:
    """Mimics `GraphRunner.run` interface."""

    model = "stub-model"

    def __init__(self, *, answer: str, passed: bool, latency_ms: int = 50) -> None:
        self._answer = answer
        self._passed = passed
        self._latency_ms = latency_ms
        self.calls: list[str] = []

    async def run(
        self,
        *,
        organization_id: uuid.UUID,
        user_id: uuid.UUID | None,
        question: str,
    ) -> InquiryResult:
        self.calls.append(question)
        return InquiryResult(
            question=question,
            answer_text=self._answer,
            citations=[],
            retrieved=[],
            critique=CritiqueResult(
                grounding_score=0.9 if self._passed else 0.3,
                hallucination_risk=0.1 if self._passed else 0.7,
                passed=self._passed,
                issues=[],
            ),
            trace=[TraceStep("plan", "Plan", "ok", self._latency_ms)],
            token_usages=[],
            model=self.model,
            error=None,
        )


def _dataset() -> GoldQuestionDataset:
    return GoldQuestionDataset(
        name="unit-test",
        description="unit",
        items=(
            GoldItem(
                id="a",
                question="will this answer contain alpha?",
                expected_doc_filenames=("any.pdf",),
                must_contain_any=(("alpha",),),
            ),
            GoldItem(
                id="b",
                question="will this answer contain beta?",
                expected_doc_filenames=("any.pdf",),
                must_contain_any=(("beta",),),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_runner_invokes_runner_per_item_and_aggregates() -> None:
    runner: Any = _StubRunner(answer="alpha and beta", passed=True)
    out = await run_evaluation(
        runner=runner,
        dataset=_dataset(),
        organization_id=uuid.uuid4(),
    )
    # Both questions ran, in order.
    assert runner.calls == [
        "will this answer contain alpha?",
        "will this answer contain beta?",
    ]
    # Aggregate is the average of two perfect-faithfulness items.
    assert out.aggregate.n_items == 2
    assert out.aggregate.faithfulness == 1.0
    # Both expected docs are absent from retrieved (stub returns []), so
    # retrieval recall is 0 and item_passed must be False even though critic passes.
    assert out.aggregate.retrieval_recall == 0.0
    assert out.aggregate.pass_rate == 0.0


@pytest.mark.asyncio
async def test_runner_reports_dataset_version_in_outcome() -> None:
    ds = _dataset()
    runner: Any = _StubRunner(answer="alpha and beta", passed=True)
    out = await run_evaluation(
        runner=runner,
        dataset=ds,
        organization_id=uuid.uuid4(),
    )
    assert out.dataset_name == ds.name
    assert out.dataset_version == ds.version
    assert out.model == runner.model
