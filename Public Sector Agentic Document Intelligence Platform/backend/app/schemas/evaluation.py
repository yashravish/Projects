"""Pydantic schemas for the evaluation harness API.

Mirrors `app.eval.metrics` and `app.eval.runner` dataclasses on the wire,
preserving the same field names so the frontend Zod schemas can be a 1:1
translation. The internal dataclasses are kept frozen so they can flow
through MLflow / JSONB persistence without aliasing.
"""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.eval.dataset import GoldItem, GoldQuestionDataset
from app.eval.metrics import AggregateMetrics, ItemMetrics
from app.eval.runner import EvaluationItemResult, EvaluationOutcome


EvaluationStatus = Literal["pending", "running", "success", "failed"]


# ── Request --------------------------------------------------------------------


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str | None = Field(
        default=None,
        description="Registered dataset name; omit for the default gold set.",
    )
    top_k: int = Field(default=6, ge=1, le=20)
    candidate_k: int = Field(default=25, ge=5, le=100)


# ── Dataset peek ---------------------------------------------------------------


class GoldItemOut(BaseModel):
    id: str
    question: str
    expected_doc_filenames: list[str]
    must_contain_any: list[list[str]]
    forbidden_phrases: list[str]
    topic: str

    @classmethod
    def from_dataclass(cls, gi: GoldItem) -> GoldItemOut:
        return cls(
            id=gi.id,
            question=gi.question,
            expected_doc_filenames=list(gi.expected_doc_filenames),
            must_contain_any=[list(g) for g in gi.must_contain_any],
            forbidden_phrases=list(gi.forbidden_phrases),
            topic=gi.topic,
        )


class DatasetOut(BaseModel):
    name: str
    description: str
    version: str
    n_items: int
    items: list[GoldItemOut]

    @classmethod
    def from_dataset(cls, ds: GoldQuestionDataset) -> DatasetOut:
        return cls(
            name=ds.name,
            description=ds.description,
            version=ds.version,
            n_items=len(ds),
            items=[GoldItemOut.from_dataclass(it) for it in ds.items],
        )


# ── Metrics --------------------------------------------------------------------


class ItemMetricsOut(BaseModel):
    item_id: str
    retrieval_recall: float
    retrieval_precision: float
    citation_precision: float
    citation_recall: float
    faithfulness: float
    forbidden_phrase_rate: float
    grounding_score: float
    hallucination_risk: float
    answer_passed_critic: bool
    item_passed: bool
    latency_ms: int
    n_retrieved: int
    n_citations: int

    @classmethod
    def from_dataclass(cls, m: ItemMetrics) -> ItemMetricsOut:
        return cls(**m.as_dict())  # `as_dict` already names every field


class AggregateMetricsOut(BaseModel):
    n_items: int
    pass_rate: float
    retrieval_recall: float
    retrieval_precision: float
    citation_precision: float
    citation_recall: float
    faithfulness: float
    forbidden_phrase_rate: float
    grounding_score: float
    hallucination_risk: float
    latency_ms_p50: float
    latency_ms_p95: float
    n_failures: int

    @classmethod
    def from_dataclass(cls, a: AggregateMetrics) -> AggregateMetricsOut:
        return cls(**a.as_dict())


# ── Per-item drill-down --------------------------------------------------------


class EvalCitationOut(BaseModel):
    document_filename: str
    page_start: int
    page_end: int
    snippet: str


class EvalItemAnswerOut(BaseModel):
    """Compact view of the inquiry that produced this item's metrics.

    Intentionally smaller than `InquiryResponse` — the eval roll-up doesn't
    need every chunk score; if the user wants the full trace they can run
    the question via `/query/inquiry`.
    """

    question: str
    answer_text: str
    error: str | None
    citations: list[EvalCitationOut]
    grounding_score: float
    hallucination_risk: float
    passed: bool
    latency_ms: int
    cost_usd: float


class EvaluationItemOut(BaseModel):
    gold: GoldItemOut
    metrics: ItemMetricsOut
    inquiry: EvalItemAnswerOut

    @classmethod
    def from_runtime(cls, item: EvaluationItemResult) -> EvaluationItemOut:
        inq = item.inquiry
        return cls(
            gold=GoldItemOut.from_dataclass(item.gold),
            metrics=ItemMetricsOut.from_dataclass(item.metrics),
            inquiry=EvalItemAnswerOut(
                question=inq.question,
                answer_text=inq.answer_text,
                error=inq.error,
                citations=[
                    EvalCitationOut(
                        document_filename=c.document_filename,
                        page_start=c.page_start,
                        page_end=c.page_end,
                        snippet=c.snippet,
                    )
                    for c in inq.citations
                ],
                grounding_score=float(inq.critique.grounding_score),
                hallucination_risk=float(inq.critique.hallucination_risk),
                passed=bool(inq.critique.passed),
                latency_ms=int(inq.total_latency_ms),
                cost_usd=float(inq.total_cost_usd),
            ),
        )

    @classmethod
    def from_persisted(cls, raw: dict[str, Any]) -> EvaluationItemOut:
        """Rehydrate from the JSONB row stored on `EvaluationRun.per_item_results`."""
        gold = raw.get("gold") or {}
        metrics = raw.get("metrics") or {}
        inquiry = raw.get("inquiry") or {}
        critique = inquiry.get("critique") or {}
        return cls(
            gold=GoldItemOut(
                id=str(gold.get("id") or ""),
                question=str(gold.get("question") or ""),
                expected_doc_filenames=[
                    str(x) for x in (gold.get("expected_doc_filenames") or [])
                ],
                must_contain_any=[
                    [str(x) for x in g]
                    for g in (gold.get("must_contain_any") or [])
                ],
                forbidden_phrases=[
                    str(x) for x in (gold.get("forbidden_phrases") or [])
                ],
                topic=str(gold.get("topic") or ""),
            ),
            metrics=ItemMetricsOut(
                item_id=str(metrics.get("item_id") or ""),
                retrieval_recall=float(metrics.get("retrieval_recall") or 0.0),
                retrieval_precision=float(metrics.get("retrieval_precision") or 0.0),
                citation_precision=float(metrics.get("citation_precision") or 0.0),
                citation_recall=float(metrics.get("citation_recall") or 0.0),
                faithfulness=float(metrics.get("faithfulness") or 0.0),
                forbidden_phrase_rate=float(metrics.get("forbidden_phrase_rate") or 0.0),
                grounding_score=float(metrics.get("grounding_score") or 0.0),
                hallucination_risk=float(metrics.get("hallucination_risk") or 0.0),
                answer_passed_critic=bool(metrics.get("answer_passed_critic")),
                item_passed=bool(metrics.get("item_passed")),
                latency_ms=int(metrics.get("latency_ms") or 0),
                n_retrieved=int(metrics.get("n_retrieved") or 0),
                n_citations=int(metrics.get("n_citations") or 0),
            ),
            inquiry=EvalItemAnswerOut(
                question=str(inquiry.get("question") or ""),
                answer_text=str(inquiry.get("answer_text") or ""),
                error=inquiry.get("error"),
                citations=[
                    EvalCitationOut(
                        document_filename=str(c.get("document_filename") or ""),
                        page_start=int(c.get("page_start") or 0),
                        page_end=int(c.get("page_end") or 0),
                        snippet=str(c.get("snippet") or ""),
                    )
                    for c in (inquiry.get("citations") or [])
                ],
                grounding_score=float(critique.get("grounding_score") or 0.0),
                hallucination_risk=float(critique.get("hallucination_risk") or 0.0),
                passed=bool(critique.get("passed")),
                latency_ms=int(inquiry.get("latency_ms") or 0),
                cost_usd=float(inquiry.get("cost_usd") or 0.0),
            ),
        )


# ── Persisted run views --------------------------------------------------------


class EvaluationRunSummary(BaseModel):
    """Compact summary for the listing rail."""

    run_id: uuid.UUID
    dataset_name: str
    dataset_version: str
    model: str
    status: EvaluationStatus
    n_items: int
    pass_rate: float
    grounding_score: float
    faithfulness: float
    retrieval_recall: float
    latency_ms_p50: float
    mlflow_run_id: str | None
    created_at: dt.datetime


class EvaluationRunList(BaseModel):
    items: list[EvaluationRunSummary]
    total: int
    page: int
    page_size: int


class EvaluationRunDetail(BaseModel):
    """Full eval run, with aggregate metrics and per-item drill-down."""

    run_id: uuid.UUID
    dataset_name: str
    dataset_version: str
    model: str
    status: EvaluationStatus
    aggregate: AggregateMetricsOut
    items: list[EvaluationItemOut]
    prompt_versions: dict[str, Any]
    retrieval_config: dict[str, Any]
    wall_time_ms: int
    mlflow_run_id: str | None
    created_at: dt.datetime

    @classmethod
    def from_outcome(
        cls,
        *,
        run_id: uuid.UUID,
        outcome: EvaluationOutcome,
        prompt_versions: dict[str, Any],
        retrieval_config: dict[str, Any],
        mlflow_run_id: str | None,
        created_at: dt.datetime,
        status: EvaluationStatus = "success",
    ) -> EvaluationRunDetail:
        return cls(
            run_id=run_id,
            dataset_name=outcome.dataset_name,
            dataset_version=outcome.dataset_version,
            model=outcome.model,
            status=status,
            aggregate=AggregateMetricsOut.from_dataclass(outcome.aggregate),
            items=[EvaluationItemOut.from_runtime(it) for it in outcome.items],
            prompt_versions=prompt_versions,
            retrieval_config=retrieval_config,
            wall_time_ms=outcome.wall_time_ms,
            mlflow_run_id=mlflow_run_id,
            created_at=created_at,
        )


__all__ = [
    "AggregateMetricsOut",
    "DatasetOut",
    "EvalCitationOut",
    "EvalItemAnswerOut",
    "EvaluationItemOut",
    "EvaluationRunDetail",
    "EvaluationRunList",
    "EvaluationRunRequest",
    "EvaluationRunSummary",
    "EvaluationStatus",
    "GoldItemOut",
    "ItemMetricsOut",
]
