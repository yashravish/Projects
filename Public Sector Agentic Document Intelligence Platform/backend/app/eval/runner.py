"""Drive a `GoldQuestionDataset` through the inquiry graph and score it.

Concerns:
  * Run each gold item through *the exact same* `GraphRunner` that serves
    `/query/inquiry`, so the regression measures the production path.
  * Items run sequentially. The graph is fast (~tens of ms with offline LLM,
    seconds with `gpt-4o-mini`) and the dataset is small. Sequential keeps
    the audit trail readable and avoids spurious DB contention.
  * Failures in a single item do not abort the run — `GraphRunner.run`
    already converts exceptions into a structured `InquiryResult(error=…)`,
    which the metrics layer treats as a failed item.
"""
from __future__ import annotations

import dataclasses
import time
import uuid

from app.agents.graph import GraphRunner
from app.agents.state import InquiryResult
from app.eval.dataset import GoldItem, GoldQuestionDataset
from app.eval.metrics import (
    AggregateMetrics,
    ItemMetrics,
    aggregate as aggregate_metrics,
    score_item,
)
from app.logging_config import get_logger

log = get_logger("eval.runner")


@dataclasses.dataclass(frozen=True)
class EvaluationItemResult:
    """One gold item's run + scores. Persisted on `EvaluationRun.per_item_results`."""

    gold: GoldItem
    inquiry: InquiryResult
    metrics: ItemMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "gold": self.gold.as_dict(),
            "metrics": self.metrics.as_dict(),
            "inquiry": {
                "question": self.inquiry.question,
                "answer_text": self.inquiry.answer_text,
                "model": self.inquiry.model,
                "error": self.inquiry.error,
                "citations": [c.as_dict() for c in self.inquiry.citations],
                "retrieved": [
                    {
                        "chunk_id": str(c.chunk_id),
                        "document_id": str(c.document_id),
                        "document_filename": c.document_filename,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "fused_score": c.fused_score,
                    }
                    for c in self.inquiry.retrieved
                ],
                "critique": self.inquiry.critique.as_dict(),
                "latency_ms": self.inquiry.total_latency_ms,
                "token_input": self.inquiry.total_input_tokens,
                "token_output": self.inquiry.total_output_tokens,
                "cost_usd": self.inquiry.total_cost_usd,
            },
        }


@dataclasses.dataclass(frozen=True)
class EvaluationOutcome:
    """The full deliverable of one evaluation run, before persistence."""

    dataset_name: str
    dataset_version: str
    model: str
    items: list[EvaluationItemResult]
    aggregate: AggregateMetrics
    wall_time_ms: int

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "model": self.model,
            "items": [it.as_dict() for it in self.items],
            "aggregate": self.aggregate.as_dict(),
            "wall_time_ms": self.wall_time_ms,
        }


async def run_evaluation(
    *,
    runner: GraphRunner,
    dataset: GoldQuestionDataset,
    organization_id: uuid.UUID,
    user_id: uuid.UUID | None = None,
) -> EvaluationOutcome:
    """Run every gold item through `runner.run`, score, and aggregate."""
    started = time.monotonic()
    item_results: list[EvaluationItemResult] = []

    for gold in dataset:
        log.info(
            "eval.item.start",
            dataset=dataset.name,
            dataset_version=dataset.version,
            item_id=gold.id,
            organization_id=str(organization_id),
        )
        result = await runner.run(
            organization_id=organization_id,
            user_id=user_id,
            question=gold.question,
        )
        metrics = score_item(gold=gold, result=result)
        item_results.append(
            EvaluationItemResult(gold=gold, inquiry=result, metrics=metrics)
        )
        log.info(
            "eval.item.done",
            item_id=gold.id,
            passed=metrics.item_passed,
            recall=metrics.retrieval_recall,
            faithfulness=metrics.faithfulness,
            grounding=metrics.grounding_score,
        )

    agg = aggregate_metrics([ir.metrics for ir in item_results])
    wall_ms = int((time.monotonic() - started) * 1000)

    log.info(
        "eval.run.complete",
        dataset=dataset.name,
        dataset_version=dataset.version,
        n_items=agg.n_items,
        pass_rate=agg.pass_rate,
        wall_time_ms=wall_ms,
    )
    return EvaluationOutcome(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        model=runner.model,
        items=item_results,
        aggregate=agg,
        wall_time_ms=wall_ms,
    )


__all__ = [
    "EvaluationItemResult",
    "EvaluationOutcome",
    "run_evaluation",
]
