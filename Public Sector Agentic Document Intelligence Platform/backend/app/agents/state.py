"""Typed state and result records for the LangGraph inquiry agent.

`GraphState` is what flows between nodes inside LangGraph. The `InquiryResult`
is what the service layer hands back to the API and what gets persisted on
QueryRun.answer.

These objects are intentionally **pure data**. They contain no SQLAlchemy
models, no DB sessions, and no event-loop / Celery state, so they round-trip
cleanly through MLflow logging, JSONB persistence, and the wire.
"""
from __future__ import annotations

import dataclasses
import uuid
from typing import Any, TypedDict

from app.agents.llm_client import TokenUsage
from app.retrieval.hybrid import RetrievedChunk


@dataclasses.dataclass(frozen=True)
class Citation:
    """A single binding between an inline `[N]` marker and a chunk.

    `index` is the 1-based marker the synthesizer emitted in the answer text.
    """

    index: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    page_start: int
    page_end: int
    snippet: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "document_filename": self.document_filename,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "snippet": self.snippet,
        }


@dataclasses.dataclass(frozen=True)
class CritiqueResult:
    """Output of the critic node — one number per axis, plus issues found."""

    grounding_score: float  # in [0, 1]; 1 = every claim grounded
    hallucination_risk: float  # in [0, 1]; 0 = no risk
    passed: bool
    issues: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "grounding_score": self.grounding_score,
            "hallucination_risk": self.hallucination_risk,
            "passed": self.passed,
            "issues": list(self.issues),
        }


@dataclasses.dataclass(frozen=True)
class TraceStep:
    """One node's execution record for the audit trail and the UI timeline."""

    node: str  # plan | retrieve | synthesize | critique
    label: str  # human-friendly title
    detail: str  # one-line summary
    duration_ms: int
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "label": self.label,
            "detail": self.detail,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }


class GraphState(TypedDict, total=False):
    """LangGraph state mutated step-by-step by each node.

    `total=False` so each node only declares the keys it writes.
    """

    organization_id: uuid.UUID
    user_id: uuid.UUID | None
    question: str
    model: str

    # plan node
    sub_questions: list[str]

    # retrieve node
    retrieved: list[RetrievedChunk]

    # synthesize node
    answer_text: str
    citations: list[Citation]

    # critique node
    critique: CritiqueResult

    # cross-cutting
    trace: list[TraceStep]
    token_usages: list[TokenUsage]
    error: str | None


@dataclasses.dataclass(frozen=True)
class InquiryResult:
    """End-of-graph deliverable handed to the service layer."""

    question: str
    answer_text: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    critique: CritiqueResult
    trace: list[TraceStep]
    token_usages: list[TokenUsage]
    model: str
    error: str | None = None

    @property
    def total_input_tokens(self) -> int:
        return sum(u.prompt_tokens for u in self.token_usages)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.completion_tokens for u in self.token_usages)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(u.cost_usd for u in self.token_usages), 6)

    @property
    def total_latency_ms(self) -> int:
        return sum(s.duration_ms for s in self.trace)


__all__ = [
    "Citation",
    "CritiqueResult",
    "GraphState",
    "InquiryResult",
    "TraceStep",
]
