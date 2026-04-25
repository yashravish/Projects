"""Training & model registry API.

Routes:
    POST   /training/jobs                  — kick off a training job (sync)
    GET    /training/jobs                  — paginated history
    GET    /training/jobs/{job_id}         — one job's full detail

    GET    /models                         — paginated registry
    GET    /models/{model_id}              — model detail
    POST   /models/{model_id}/promote      — { stage: production | archived }
    POST   /models/{model_id}/predict      — score (query, passages[])

The POST /training/jobs handler is synchronous — the local backend trains in
under a second on the seeded corpus, and a SageMaker job is polled to
completion (the timeout matches the training-job stopping condition). For
larger trainings a Celery handoff would be the obvious extension.

Promotions and runs require the `analyst` or `admin` role; viewers can read
but not promote.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.db.models import User
from app.deps import CurrentUser, SessionDep, require_role
from app.observability import audit_emitter
from app.observability.mlflow_client import get_mlflow_recorder
from app.schemas.training import (
    PromoteModelRequest,
    RegisteredModelDetail,
    RegisteredModelList,
    RerankerPredictRequest,
    RerankerPredictResponse,
    TrainingJobDetail,
    TrainingJobList,
    TrainingJobRequest,
)
from app.services import training_service

router = APIRouter(prefix="", tags=["training"])


# ---- Training jobs ----------------------------------------------------------


@router.post(
    "/training/jobs",
    response_model=TrainingJobDetail,
    status_code=status.HTTP_200_OK,
)
async def post_training_job(
    payload: TrainingJobRequest,
    session: SessionDep,
    user: User = Depends(require_role("admin", "analyst")),
) -> TrainingJobDetail:
    try:
        detail = await training_service.submit_training_job_for_tenant(
            session=session,
            organization_id=user.organization_id,
            user_id=user.id,
            request=payload,
            recorder=get_mlflow_recorder(),
        )
    except training_service.TrainingError as exc:
        await audit_emitter.emit(
            session=session,
            organization_id=user.organization_id,
            actor_id=user.id,
            action="training.submit",
            resource_type="training_job",
            resource_id=None,
            outcome="error",
            metadata={
                "name": payload.name,
                "auto_promote": payload.auto_promote,
                "reason": str(exc)[:200],
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    await audit_emitter.emit(
        session=session,
        organization_id=user.organization_id,
        actor_id=user.id,
        action="training.submit",
        resource_type="training_job",
        resource_id=detail.job_id,
        outcome="success" if detail.status == "success" else "error",
        metadata={
            "name": detail.name,
            "version": detail.version,
            "backend": detail.backend,
            "auto_promote": payload.auto_promote,
            "registered_model_id": str(detail.registered_model_id)
            if detail.registered_model_id
            else None,
            "duration_s": detail.duration_s,
            "mlflow_run_id": detail.mlflow_run_id,
        },
    )
    if payload.auto_promote and detail.registered_model_id is not None:
        await audit_emitter.emit(
            session=session,
            organization_id=user.organization_id,
            actor_id=user.id,
            action="model.promote",
            resource_type="registered_model",
            resource_id=detail.registered_model_id,
            outcome="success",
            metadata={
                "name": detail.name,
                "version": detail.version,
                "stage": "production",
                "auto_promoted": True,
            },
        )
    return detail


@router.get("/training/jobs", response_model=TrainingJobList)
async def list_training_jobs(
    session: SessionDep,
    user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
) -> TrainingJobList:
    items, total = await training_service.list_training_jobs(
        session=session,
        organization_id=user.organization_id,
        page=page,
        page_size=page_size,
    )
    return TrainingJobList(
        items=items, total=total, page=page, page_size=page_size
    )


@router.get("/training/jobs/{job_id}", response_model=TrainingJobDetail)
async def get_training_job(
    job_id: uuid.UUID,
    session: SessionDep,
    user: CurrentUser,
) -> TrainingJobDetail:
    try:
        return await training_service.get_training_job(
            session=session,
            organization_id=user.organization_id,
            job_id=job_id,
        )
    except training_service.TrainingJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="training job not found") from exc


# ---- Registered models ------------------------------------------------------


@router.get("/models", response_model=RegisteredModelList)
async def list_models(
    session: SessionDep,
    user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
) -> RegisteredModelList:
    items, total = await training_service.list_registered_models(
        session=session,
        organization_id=user.organization_id,
        page=page,
        page_size=page_size,
    )
    return RegisteredModelList(
        items=items, total=total, page=page, page_size=page_size
    )


@router.get("/models/{model_id}", response_model=RegisteredModelDetail)
async def get_model(
    model_id: uuid.UUID,
    session: SessionDep,
    user: CurrentUser,
) -> RegisteredModelDetail:
    try:
        return await training_service.get_registered_model(
            session=session,
            organization_id=user.organization_id,
            model_id=model_id,
        )
    except training_service.RegisteredModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@router.post(
    "/models/{model_id}/promote",
    response_model=RegisteredModelDetail,
)
async def promote_model(
    model_id: uuid.UUID,
    payload: PromoteModelRequest,
    session: SessionDep,
    user: User = Depends(require_role("admin", "analyst")),
) -> RegisteredModelDetail:
    try:
        detail = await training_service.promote_model(
            session=session,
            organization_id=user.organization_id,
            user_id=user.id,
            model_id=model_id,
            request=payload,
        )
    except training_service.RegisteredModelNotFoundError as exc:
        await audit_emitter.emit(
            session=session,
            organization_id=user.organization_id,
            actor_id=user.id,
            action="model.promote",
            resource_type="registered_model",
            resource_id=model_id,
            outcome="denied",
            metadata={"reason": "not_found", "stage": payload.stage},
        )
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        await audit_emitter.emit(
            session=session,
            organization_id=user.organization_id,
            actor_id=user.id,
            action="model.promote",
            resource_type="registered_model",
            resource_id=model_id,
            outcome="error",
            metadata={"reason": str(exc)[:200], "stage": payload.stage},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await audit_emitter.emit(
        session=session,
        organization_id=user.organization_id,
        actor_id=user.id,
        action="model.promote" if payload.stage == "production" else "model.archive",
        resource_type="registered_model",
        resource_id=detail.model_id,
        outcome="success",
        metadata={
            "name": detail.name,
            "version": detail.version,
            "stage": detail.stage,
            "notes": payload.notes,
        },
    )
    return detail


@router.post(
    "/models/{model_id}/predict",
    response_model=RerankerPredictResponse,
)
async def predict_with_model(
    model_id: uuid.UUID,
    payload: RerankerPredictRequest,
    session: SessionDep,
    user: CurrentUser,
) -> RerankerPredictResponse:
    try:
        return await training_service.predict_with_model(
            session=session,
            organization_id=user.organization_id,
            model_id=model_id,
            request=payload,
        )
    except training_service.RegisteredModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except training_service.TrainingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
