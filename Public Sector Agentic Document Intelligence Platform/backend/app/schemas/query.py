"""Pydantic schemas for the inquiry endpoint.

These shapes are the wire contract the frontend consumes (mirrored in
`frontend/src/api/schemas.ts` (all domains, including audit). Anything user-visible flows through here;
internal `dataclass` records (Citation, RetrievedChunk, TraceStep) are
adapted via `from_*` methods so service code never hand-builds dicts.
"""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agents.state import (
    Citation as CitationDC,
)
from app.agents.state import (
    CritiqueResult,
    InquiryResult,
    TraceStep,
)
from app.retrieval.hybrid import RetrievedChunk

QueryStatus = Literal["success", "failed"]


class InquiryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)
    candidate_k: int = Field(default=25, ge=5, le=100)


class CitationOut(BaseModel):
    index: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    page_start: int
    page_end: int
    snippet: str

    @classmethod
    def from_dataclass(cls, c: CitationDC) -> CitationOut:
        return cls(
            index=c.index,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_filename=c.document_filename,
            page_start=c.page_start,
            page_end=c.page_end,
            snippet=c.snippet,
        )


class RetrievedChunkOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    page_start: int
    page_end: int
    chunk_index: int
    fused_score: float
    bm25_rank: int
    bm25_score: float
    vector_rank: int
    vector_score: float
    snippet: str

    @classmethod
    def from_dataclass(cls, c: RetrievedChunk) -> RetrievedChunkOut:
        return cls(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_filename=c.document_filename,
            page_start=c.page_start,
            page_end=c.page_end,
            chunk_index=c.chunk_index,
            fused_score=c.fused_score,
            bm25_rank=c.bm25.rank,
            bm25_score=c.bm25.score,
            vector_rank=c.vector.rank,
            vector_score=c.vector.score,
            snippet=c.text[:600],
        )


class TraceStepOut(BaseModel):
    node: str
    label: str
    detail: str
    duration_ms: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dataclass(cls, t: TraceStep) -> TraceStepOut:
        return cls(
            node=t.node,
            label=t.label,
            detail=t.detail,
            duration_ms=t.duration_ms,
            metadata=t.metadata,
        )


class CritiqueOut(BaseModel):
    grounding_score: float
    hallucination_risk: float
    passed: bool
    issues: list[str]

    @classmethod
    def from_dataclass(cls, c: CritiqueResult) -> CritiqueOut:
        return cls(
            grounding_score=c.grounding_score,
            hallucination_risk=c.hallucination_risk,
            passed=c.passed,
            issues=list(c.issues),
        )


class InquiryResponse(BaseModel):
    """The full result of one /inquiry call. Persisted; safe to replay."""

    run_id: uuid.UUID
    status: QueryStatus
    question: str
    answer_text: str
    citations: list[CitationOut]
    retrieved: list[RetrievedChunkOut]
    critique: CritiqueOut
    trace: list[TraceStepOut]
    model: str
    latency_ms: int
    token_input: int
    token_output: int
    cost_usd: float
    mlflow_run_id: str | None = None
    error: str | None = None
    created_at: dt.datetime

    @classmethod
    def from_inquiry(
        cls,
        *,
        run_id: uuid.UUID,
        result: InquiryResult,
        mlflow_run_id: str | None,
        created_at: dt.datetime,
    ) -> InquiryResponse:
        return cls(
            run_id=run_id,
            status="failed" if result.error else "success",
            question=result.question,
            answer_text=result.answer_text,
            citations=[CitationOut.from_dataclass(c) for c in result.citations],
            retrieved=[
                RetrievedChunkOut.from_dataclass(c) for c in result.retrieved
            ],
            critique=CritiqueOut.from_dataclass(result.critique),
            trace=[TraceStepOut.from_dataclass(t) for t in result.trace],
            model=result.model,
            latency_ms=result.total_latency_ms,
            token_input=result.total_input_tokens,
            token_output=result.total_output_tokens,
            cost_usd=result.total_cost_usd,
            mlflow_run_id=mlflow_run_id,
            error=result.error,
            created_at=created_at,
        )


class QueryRunListItem(BaseModel):
    """Compact summary for the inquiry history rail."""

    run_id: uuid.UUID
    question: str
    status: QueryStatus
    grounding_score: float | None
    hallucination_risk: float | None
    n_citations: int
    latency_ms: int
    model: str
    created_at: dt.datetime


class QueryRunList(BaseModel):
    items: list[QueryRunListItem]
    total: int
    page: int
    page_size: int


__all__ = [
    "CitationOut",
    "CritiqueOut",
    "InquiryRequest",
    "InquiryResponse",
    "QueryRunList",
    "QueryRunListItem",
    "QueryStatus",
    "RetrievedChunkOut",
    "TraceStepOut",
]
