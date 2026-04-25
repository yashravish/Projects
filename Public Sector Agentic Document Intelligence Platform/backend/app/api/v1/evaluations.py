"""Evaluation harness API.

Routes:
    GET  /evaluations/dataset          — peek the active gold dataset
    POST /evaluations/run              — run the harness, persist a row, return detail
    GET  /evaluations                  — paginated history for the caller's tenant
    GET  /evaluations/{run_id}         — replay one persisted evaluation

The POST handler is synchronous because the gold set is small (~10 items)
and each item runs in <1s with the offline LLM, <10s with `gpt-4o-mini`.
For larger datasets a Celery task with polling would be the obvious path.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status

from app.agents.llm_client import build_embedder, build_llm
from app.deps import CurrentUser, SessionDep
from app.logging_config import get_logger
from app.observability import audit_emitter
from app.observability.mlflow_client import get_mlflow_recorder
from app.schemas.evaluation import (
    DatasetOut,
    EvaluationRunDetail,
    EvaluationRunList,
    EvaluationRunRequest,
)
from app.services import evaluation_service

router = APIRouter(prefix="/evaluations", tags=["evaluations"])
log = get_logger("api.evaluations")


@router.get("/dataset", response_model=DatasetOut)
async def get_dataset() -> DatasetOut:
    """Tenant-agnostic — the gold set is the same shape for every tenant."""
    return evaluation_service.get_default_dataset_view()


@router.post(
    "/run",
    response_model=EvaluationRunDetail,
    status_code=status.HTTP_200_OK,
)
async def post_run(
    payload: EvaluationRunRequest,
    session: SessionDep,
    user: CurrentUser,
) -> EvaluationRunDetail:
    llm = build_llm()
    embedder = build_embedder()
    recorder = get_mlflow_recorder()

    try:
        detail = await evaluation_service.run_evaluation_for_tenant(
            session=session,
            organization_id=user.organization_id,
            user_id=user.id,
            request=payload,
            llm=llm,
            embedder=embedder,
            recorder=recorder,
        )
    except KeyError as exc:
        await audit_emitter.emit(
            session=session,
            organization_id=user.organization_id,
            actor_id=user.id,
            action="evaluation.run",
            resource_type="evaluation_run",
            resource_id=None,
            outcome="error",
            metadata={"reason": "unknown_dataset", "dataset_name": payload.dataset_name},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await audit_emitter.emit(
        session=session,
        organization_id=user.organization_id,
        actor_id=user.id,
        action="evaluation.run",
        resource_type="evaluation_run",
        resource_id=detail.run_id,
        outcome="success" if detail.status == "success" else "error",
        metadata={
            "dataset_name": detail.dataset_name,
            "dataset_version": detail.dataset_version,
            "model": detail.model,
            "n_items": detail.aggregate.n_items,
            "n_failures": detail.aggregate.n_failures,
            "pass_rate": detail.aggregate.pass_rate,
            "mlflow_run_id": detail.mlflow_run_id,
        },
    )
    return detail


@router.get("", response_model=EvaluationRunList)
async def list_runs(
    session: SessionDep,
    user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
) -> EvaluationRunList:
    items, total = await evaluation_service.list_evaluations(
        session=session,
        organization_id=user.organization_id,
        page=page,
        page_size=page_size,
    )
    return EvaluationRunList(
        items=items, total=total, page=page, page_size=page_size
    )


@router.get("/{run_id}", response_model=EvaluationRunDetail)
async def get_run(
    run_id: uuid.UUID,
    session: SessionDep,
    user: CurrentUser,
) -> EvaluationRunDetail:
    try:
        return await evaluation_service.get_evaluation(
            session=session,
            organization_id=user.organization_id,
            run_id=run_id,
        )
    except evaluation_service.EvaluationRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail="evaluation run not found") from exc
