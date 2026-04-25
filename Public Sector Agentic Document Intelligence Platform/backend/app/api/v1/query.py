"""Query (inquiry) API.

Routes:
    POST /query/inquiry        — run one inquiry, persist a QueryRun, return answer + trace
    GET  /query/runs           — paginated history for the caller's org
    GET  /query/runs/{run_id}  — replay one persisted run

The POST handler is intentionally synchronous — clients want one round-trip
that returns the full answer + citations + trace + critique. The graph is
fast enough for human-scale latencies (typically <2s with offline LLM,
<6s with `gpt-4o-mini`). Streaming is a future enhancement.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status

from app.agents.llm_client import build_embedder, build_llm
from app.deps import CurrentUser, SessionDep
from app.governance import pii as pii_mod
from app.logging_config import get_logger
from app.observability import audit_emitter
from app.observability.mlflow_client import get_mlflow_recorder
from app.schemas.query import (
    InquiryRequest,
    InquiryResponse,
    QueryRunList,
)
from app.services import query_service

router = APIRouter(prefix="/query", tags=["query"])
log = get_logger("api.query")


@router.post(
    "/inquiry",
    response_model=InquiryResponse,
    status_code=status.HTTP_200_OK,
)
async def post_inquiry(
    payload: InquiryRequest,
    session: SessionDep,
    user: CurrentUser,
) -> InquiryResponse:
    llm = build_llm()
    embedder = build_embedder()
    recorder = get_mlflow_recorder()

    # PII pass over the inbound question. We redact before persistence so
    # the `query_runs.query_text` column never holds raw PII; the audit
    # event records *what kinds* of PII were detected (counts only, never
    # values) so an analyst can prove a leak was caught.
    redacted_question, findings = pii_mod.redact(payload.question)
    safe_payload = (
        payload
        if not findings
        else payload.model_copy(update={"question": redacted_question})
    )

    try:
        response = await query_service.run_inquiry(
            session=session,
            organization_id=user.organization_id,
            user_id=user.id,
            request=safe_payload,
            llm=llm,
            embedder=embedder,
            recorder=recorder,
        )
    except ValueError as exc:
        await audit_emitter.emit(
            session=session,
            organization_id=user.organization_id,
            actor_id=user.id,
            action="query.run",
            resource_type="query_run",
            resource_id=None,
            outcome="error",
            metadata={
                "reason": str(exc)[:200],
                "pii_kinds": pii_mod.summarize(findings),
                "had_pii": bool(findings),
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await audit_emitter.emit(
        session=session,
        organization_id=user.organization_id,
        actor_id=user.id,
        action="query.run",
        resource_type="query_run",
        resource_id=response.run_id,
        outcome="success" if response.status == "success" else "error",
        metadata={
            "model": response.model,
            "n_citations": len(response.citations),
            "n_retrieved": len(response.retrieved),
            "latency_ms": response.latency_ms,
            "pii_kinds": pii_mod.summarize(findings),
            "had_pii": bool(findings),
            "mlflow_run_id": response.mlflow_run_id,
        },
    )
    return response


@router.get("/runs", response_model=QueryRunList)
async def list_runs(
    session: SessionDep,
    user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
) -> QueryRunList:
    items, total = await query_service.list_query_runs(
        session=session,
        organization_id=user.organization_id,
        page=page,
        page_size=page_size,
    )
    return QueryRunList(items=items, total=total, page=page, page_size=page_size)


@router.get("/runs/{run_id}", response_model=InquiryResponse)
async def get_run(
    run_id: uuid.UUID,
    session: SessionDep,
    user: CurrentUser,
) -> InquiryResponse:
    try:
        return await query_service.get_query_run(
            session=session,
            organization_id=user.organization_id,
            run_id=run_id,
        )
    except query_service.QueryRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail="query run not found") from exc
