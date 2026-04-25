"""Evaluation harness — gold question regression for the inquiry agent."""
from app.eval.dataset import (
    GOLD_DATASET,
    GoldItem,
    GoldQuestionDataset,
    get_dataset,
)
from app.eval.metrics import (
    AggregateMetrics,
    ItemMetrics,
    aggregate as aggregate_metrics,
    score_item,
)
from app.eval.runner import (
    EvaluationOutcome,
    EvaluationItemResult,
    run_evaluation,
)

__all__ = [
    "AggregateMetrics",
    "EvaluationItemResult",
    "EvaluationOutcome",
    "GOLD_DATASET",
    "GoldItem",
    "GoldQuestionDataset",
    "ItemMetrics",
    "aggregate_metrics",
    "get_dataset",
    "run_evaluation",
    "score_item",
]
