"""Inquiry orchestration.

`run_inquiry` is the only entry point the API uses. Responsibilities:

1. Build a per-request `HybridRetriever` (DB session is request-scoped).
2. Drive the LangGraph `GraphRunner`.
3. Persist a `QueryRun` row with the answer JSON, metrics, and config.
4. Open an MLflow run alongside, log metrics + params + the answer artifact.
5. Return the persisted run id alongside the in-memory `InquiryResult`.

The list/get methods enforce tenant isolation and never deserialise raw JSONB
columns into the response — they go through Pydantic schemas.
"""
from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import GraphRunner
from app.agents.llm_client import Embedder, LLMClient
from app.agents.prompts import all_prompt_versions
from app.agents.state import InquiryResult
from app.config import get_settings
from app.db.models import QueryRun
from app.logging_config import get_logger
from app.observability.mlflow_client import MLflowRunRecorder, get_mlflow_recorder
from app.retrieval.hybrid import HybridRetriever, RetrievalConfig
from app.schemas.query import (
    InquiryRequest,
    InquiryResponse,
    QueryRunListItem,
)
from app.security.tenant import apply_tenant_filter

log = get_logger("query_service")


class QueryRunNotFoundError(Exception):
    """Raised when a QueryRun does not exist for the supplied tenant."""


async def run_inquiry(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    user_id: uuid.UUID | None,
    request: InquiryRequest,
    llm: LLMClient,
    embedder: Embedder,
    recorder: MLflowRunRecorder | None = None,
) -> InquiryResponse:
    """End-to-end inquiry: graph → persist → MLflow → response."""
    settings = get_settings()
    cfg = RetrievalConfig(
        top_k=request.top_k,
        candidate_k=request.candidate_k,
    )
    retriever = HybridRetriever(session=session, embedder=embedder)
    runner = GraphRunner(
        llm=llm,
        retriever=retriever,
        retrieval_config=cfg,
        model=settings.openai_default_model,
    )

    rec = recorder or get_mlflow_recorder()
    mlflow_run_id: str | None = None
    with rec.record_inquiry(
        organization_id=str(organization_id),
        run_name=f"inquiry-{uuid.uuid4().hex[:8]}",
        tags={
            "model": runner.model,
            "embedder": embedder.backend_name,
            "llm": llm.backend_name,
        },
    ) as active:
        mlflow_run_id = active.run_id

        result: InquiryResult = await runner.run(
            organization_id=organization_id,
            user_id=user_id,
            question=request.question,
        )

        rec.log_params(
            {
                "model": result.model,
                "top_k": cfg.top_k,
                "candidate_k": cfg.candidate_k,
                "embedder": embedder.backend_name,
                "llm": llm.backend_name,
                **{f"prompt.{k}": v for k, v in all_prompt_versions().items()},
            }
        )
        rec.log_metrics(
            {
                "latency_ms": float(result.total_latency_ms),
                "n_retrieved": float(len(result.retrieved)),
                "n_citations": float(len(result.citations)),
                "token_input": float(result.total_input_tokens),
                "token_output": float(result.total_output_tokens),
                "cost_usd": float(result.total_cost_usd),
                "grounding_score": float(result.critique.grounding_score),
                "hallucination_risk": float(result.critique.hallucination_risk),
                "passed": 1.0 if result.critique.passed else 0.0,
            }
        )
        rec.log_dict(
            {
                "question": result.question,
                "answer": result.answer_text,
                "citations": [c.as_dict() for c in result.citations],
                "trace": [t.as_dict() for t in result.trace],
                "critique": result.critique.as_dict(),
            },
            artifact_file="inquiry.json",
        )

    answer_payload = {
        "answer_text": result.answer_text,
        "citations": [c.as_dict() for c in result.citations],
        "retrieved": [
            {
                "chunk_id": str(c.chunk_id),
                "document_id": str(c.document_id),
                "document_filename": c.document_filename,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "chunk_index": c.chunk_index,
                "fused_score": c.fused_score,
                "bm25_rank": c.bm25.rank,
                "bm25_score": c.bm25.score,
                "vector_rank": c.vector.rank,
                "vector_score": c.vector.score,
                "snippet": c.text[:600],
            }
            for c in result.retrieved
        ],
        "critique": result.critique.as_dict(),
        "trace": [t.as_dict() for t in result.trace],
        "error": result.error,
    }

    run_row = QueryRun(
        organization_id=organization_id,
        user_id=user_id,
        query_text=request.question,
        model_name=result.model,
        prompt_versions=all_prompt_versions(),
        retrieval_config={
            **cfg.as_dict(),
            "embedder": embedder.backend_name,
            "llm": llm.backend_name,
        },
        answer=answer_payload,
        latency_ms=result.total_latency_ms,
        token_input=result.total_input_tokens,
        token_output=result.total_output_tokens,
        cost_usd=Decimal(str(result.total_cost_usd)),
        grounding_score=result.critique.grounding_score,
        hallucination_risk=result.critique.hallucination_risk,
        status="failed" if result.error else "success",
        error_message=result.error,
        mlflow_run_id=mlflow_run_id,
    )
    session.add(run_row)
    await session.flush()
    await session.refresh(run_row)
    await session.commit()

    log.info(
        "query.run.persisted",
        run_id=str(run_row.id),
        organization_id=str(organization_id),
        n_citations=len(result.citations),
        n_retrieved=len(result.retrieved),
        latency_ms=result.total_latency_ms,
        grounding=result.critique.grounding_score,
        mlflow_run_id=mlflow_run_id,
    )

    return InquiryResponse.from_inquiry(
        run_id=run_row.id,
        result=result,
        mlflow_run_id=mlflow_run_id,
        created_at=run_row.created_at,
    )


# ---- list / get ---------------------------------------------------------------


async def list_query_runs(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    page: int = 1,
    page_size: int = 25,
) -> tuple[list[QueryRunListItem], int]:
    page = max(page, 1)
    page_size = max(min(page_size, 100), 1)
    offset = (page - 1) * page_size

    stmt = apply_tenant_filter(
        select(QueryRun).order_by(QueryRun.created_at.desc()),
        QueryRun,
        organization_id,
    ).limit(page_size).offset(offset)
    rows = (await session.execute(stmt)).scalars().all()

    count_stmt = apply_tenant_filter(
        select(func.count(QueryRun.id)),
        QueryRun,
        organization_id,
    )
    total = int((await session.execute(count_stmt)).scalar_one())

    items: list[QueryRunListItem] = []
    for r in rows:
        n_citations = 0
        if isinstance(r.answer, dict):
            citations = r.answer.get("citations") or []
            if isinstance(citations, list):
                n_citations = len(citations)
        items.append(
            QueryRunListItem(
                run_id=r.id,
                question=r.query_text,
                status="failed" if r.status != "success" else "success",
                grounding_score=r.grounding_score,
                hallucination_risk=r.hallucination_risk,
                n_citations=n_citations,
                latency_ms=r.latency_ms,
                model=r.model_name,
                created_at=r.created_at,
            )
        )
    return items, total


async def get_query_run(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    run_id: uuid.UUID,
) -> InquiryResponse:
    stmt = apply_tenant_filter(
        select(QueryRun).where(QueryRun.id == run_id),
        QueryRun,
        organization_id,
    )
    row: QueryRun | None = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise QueryRunNotFoundError(f"QueryRun {run_id} not found")

    return _row_to_response(row)


def _row_to_response(row: QueryRun) -> InquiryResponse:
    """Reconstruct an `InquiryResponse` from the persisted JSONB payload.

    Used by `get_query_run` so the detail page can replay a past inquiry
    without re-running the graph.
    """
    payload: dict[str, Any] = row.answer if isinstance(row.answer, dict) else {}
    citations_raw = payload.get("citations") or []
    retrieved_raw = payload.get("retrieved") or []
    trace_raw = payload.get("trace") or []
    critique_raw = payload.get("critique") or {}

    return InquiryResponse(
        run_id=row.id,
        status="failed" if row.status != "success" else "success",
        question=row.query_text,
        answer_text=str(payload.get("answer_text") or ""),
        citations=_coerce_list(citations_raw, _citation_from_dict),
        retrieved=_coerce_list(retrieved_raw, _retrieved_from_dict),
        critique={
            "grounding_score": float(critique_raw.get("grounding_score") or 0.0),
            "hallucination_risk": float(critique_raw.get("hallucination_risk") or 0.0),
            "passed": bool(critique_raw.get("passed")),
            "issues": [str(i) for i in (critique_raw.get("issues") or [])],
        },
        trace=_coerce_list(trace_raw, _trace_from_dict),
        model=row.model_name,
        latency_ms=row.latency_ms,
        token_input=row.token_input,
        token_output=row.token_output,
        cost_usd=float(row.cost_usd),
        mlflow_run_id=row.mlflow_run_id,
        error=row.error_message,
        created_at=row.created_at,
    )


def _coerce_list(
    raw: object,
    transform: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [transform(item) for item in raw if isinstance(item, dict)]


def _citation_from_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": int(d.get("index") or 0),
        "chunk_id": uuid.UUID(str(d["chunk_id"])),
        "document_id": uuid.UUID(str(d["document_id"])),
        "document_filename": str(d.get("document_filename") or ""),
        "page_start": int(d.get("page_start") or 0),
        "page_end": int(d.get("page_end") or 0),
        "snippet": str(d.get("snippet") or ""),
    }


def _retrieved_from_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": uuid.UUID(str(d["chunk_id"])),
        "document_id": uuid.UUID(str(d["document_id"])),
        "document_filename": str(d.get("document_filename") or ""),
        "page_start": int(d.get("page_start") or 0),
        "page_end": int(d.get("page_end") or 0),
        "chunk_index": int(d.get("chunk_index") or 0),
        "fused_score": float(d.get("fused_score") or 0.0),
        "bm25_rank": int(d.get("bm25_rank") or -1),
        "bm25_score": float(d.get("bm25_score") or 0.0),
        "vector_rank": int(d.get("vector_rank") or -1),
        "vector_score": float(d.get("vector_score") or 0.0),
        "snippet": str(d.get("snippet") or ""),
    }


def _trace_from_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "node": str(d.get("node") or ""),
        "label": str(d.get("label") or ""),
        "detail": str(d.get("detail") or ""),
        "duration_ms": int(d.get("duration_ms") or 0),
        "metadata": d.get("metadata") if isinstance(d.get("metadata"), dict) else {},
    }


# `dt` import kept for use in tests / future helpers (e.g. window queries).
_unused = dt  # noqa: F841


__all__ = [
    "QueryRunNotFoundError",
    "get_query_run",
    "list_query_runs",
    "run_inquiry",
]
