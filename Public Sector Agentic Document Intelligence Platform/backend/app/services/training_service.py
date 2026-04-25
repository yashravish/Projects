"""Training & model registry orchestration.

Public surface:

  * `submit_training_job_for_tenant` — kicks off a training job, persists a
    `TrainingJob` row that tracks every transition (`pending` → `running` →
    `success` | `failed`), registers a successful artifact in the
    `RegisteredModel` table + on-disk registry, and (optionally) auto-promotes
    it to production. Logs the whole life-cycle to MLflow under the
    `kind=training` tag so the same observability dashboard shows training,
    inquiry, and evaluation side-by-side.

  * `list_training_jobs` / `get_training_job` — paginated, tenant-scoped.
  * `list_registered_models` / `get_registered_model` — paginated, tenant-scoped.
  * `promote_model` — transition a model to `production` (or `archived`).
                      Promotion atomically archives any prior production
                      model under the same name so there is exactly one
                      production-stage row per (org, name) at any time.
  * `predict_with_model` — load the artifact and rerank a list of passages.
                           Used by the SPA's "Test reranker" mini-tool.

Tenant isolation: every persistence + read goes through `apply_tenant_filter`,
preserving the static guarantee enforced by `tests/unit/test_tenant_isolation.py`.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.models import RegisteredModel, TrainingJob
from app.logging_config import get_logger
from app.ml.backends import (
    TrainingBackend,
    TrainingJobOutcome,
    TrainingJobSpec,
    build_training_backend,
)
from app.ml.classifier import MODEL_FILENAME
from app.ml.registry import ModelRegistry, build_model_registry
from app.ml.reranker import (
    RerankRequest,
    Reranker,
    build_reranker_from_handle,
    reset_reranker_cache_for_tests,
)
from app.observability.mlflow_client import MLflowRunRecorder, get_mlflow_recorder
from app.schemas.training import (
    PromoteModelRequest,
    RegisteredModelDetail,
    RegisteredModelSummary,
    RerankerPredictRequest,
    RerankerPredictResponse,
    ScoredPassage,
    TrainingJobDetail,
    TrainingJobMetricsOut,
    TrainingJobRequest,
    TrainingJobSummary,
)
from app.security.tenant import apply_tenant_filter

log = get_logger("training_service")


class TrainingError(Exception):
    """Raised when training cannot complete."""


class TrainingJobNotFoundError(Exception):
    """Raised when a job lookup misses for the calling tenant."""


class RegisteredModelNotFoundError(Exception):
    """Raised when a model lookup misses for the calling tenant."""


# ── Versioning ───────────────────────────────────────────────────────────────


def _next_version(*, model_name: str) -> str:
    """Generate a fresh model version string.

    Format: `vYYYYMMDD-HHMMSS-XXXXXX`. The 6-char suffix is from the start
    of a UUID4, so two simultaneous training calls for the same name in the
    same second still produce distinct versions — important because (org,
    name, version) is uniquely indexed.
    """
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"v{stamp}-{suffix}"


# ── Submission ───────────────────────────────────────────────────────────────


async def submit_training_job_for_tenant(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    user_id: uuid.UUID | None,
    request: TrainingJobRequest,
    backend: TrainingBackend | None = None,
    registry: ModelRegistry | None = None,
    recorder: MLflowRunRecorder | None = None,
    settings: Settings | None = None,
) -> TrainingJobDetail:
    """End-to-end: spec → backend → outcome → DB rows + (optional) promotion."""
    s = settings or get_settings()
    be: TrainingBackend = backend or build_training_backend(settings=s)
    reg: ModelRegistry = registry or build_model_registry(settings=s)
    rec = recorder or get_mlflow_recorder()

    name = request.name
    version = _next_version(model_name=name)
    output_dir = str(pathlib.Path(s.models_dir) / name / version)

    log.info(
        "ml.train.submit",
        organization_id=str(organization_id),
        name=name,
        version=version,
        backend=be.name,
    )

    # Persist a `running` row up front so the SPA can see it appear immediately.
    row = TrainingJob(
        organization_id=organization_id,
        triggered_by=user_id,
        name=name,
        version=version,
        backend=be.name,
        framework="sklearn-tfidf-logreg",
        status="running",
        config={
            "name": name,
            "version": version,
            "auto_promote": request.auto_promote,
            "notes": request.notes,
            "output_dir": output_dir,
        },
        metrics={},
        manifest={},
        duration_s=0.0,
        started_at=dt.datetime.now(dt.timezone.utc),
    )
    session.add(row)
    await session.flush()
    await session.refresh(row)
    job_id = row.id
    await session.commit()

    spec = TrainingJobSpec(
        name=name,
        version=version,
        output_dir=output_dir,
        extra_env={"PSDI_TRAINING_JOB_ID": str(job_id)},
    )

    outcome: TrainingJobOutcome
    mlflow_run_id: str | None = None

    try:
        with rec.record_training(
            organization_id=str(organization_id),
            run_name=f"train-{name}-{version}",
            tags={
                "name": name,
                "version": version,
                "backend": be.name,
            },
        ) as active:
            mlflow_run_id = active.run_id
            rec.log_params(
                {
                    "model_name": name,
                    "model_version": version,
                    "backend": be.name,
                    "auto_promote": request.auto_promote,
                }
            )
            outcome = be.run_training_job(spec)
            if outcome.metrics:
                rec.log_metrics(
                    {
                        f"train.{k}": float(v)
                        for k, v in outcome.metrics.items()
                        if isinstance(v, (int, float))
                    }
                )
            if outcome.manifest:
                rec.log_dict(outcome.manifest, artifact_file="manifest.json")
                if outcome.log_excerpt:
                    rec.log_dict(
                        {"log": outcome.log_excerpt},
                        artifact_file="training_log.txt.json",
                    )
    except Exception as exc:  # noqa: BLE001 — record + rethrow
        await _mark_failed(
            session=session,
            organization_id=organization_id,
            job_id=job_id,
            error=f"{type(exc).__name__}: {exc!s}",
        )
        log.exception(
            "ml.train.failed_pre_persist",
            organization_id=str(organization_id),
            name=name,
            version=version,
        )
        raise TrainingError(f"training pipeline failed: {exc!s}") from exc

    # ── Persist outcome ──────────────────────────────────────────────────
    job = await _load_job(
        session=session,
        organization_id=organization_id,
        job_id=job_id,
    )
    if job is None:
        # Should be impossible — we just inserted it. Defend anyway.
        raise TrainingError(f"training job {job_id} disappeared mid-flight")
    job.status = outcome.status
    job.framework = outcome.framework
    job.framework_version = outcome.framework_version
    job.artifact_uri = outcome.artifact_uri
    job.external_job_id = outcome.external_job_id
    job.metrics = dict(outcome.metrics)
    job.manifest = dict(outcome.manifest)
    job.log_excerpt = outcome.log_excerpt
    job.duration_s = float(outcome.duration_s)
    job.mlflow_run_id = mlflow_run_id
    job.error_message = outcome.error_message
    job.started_at = outcome.started_at
    job.finished_at = outcome.finished_at
    await session.flush()
    await session.commit()

    if outcome.status != "success":
        log.warning(
            "ml.train.completed_failure",
            job_id=str(job_id),
            error=outcome.error_message,
        )
        return _job_row_to_detail(job, registered_model_id=None)

    # Verify artifact exists locally before we register.
    local_dir = outcome.output_dir
    if not (pathlib.Path(local_dir) / MODEL_FILENAME).is_file():
        await _mark_failed(
            session=session,
            organization_id=organization_id,
            job_id=job_id,
            error=f"trained model artifact not found at {local_dir}/{MODEL_FILENAME}",
        )
        refreshed = await _load_job(
            session=session,
            organization_id=organization_id,
            job_id=job_id,
        )
        assert refreshed is not None  # noqa: S101 — invariant: row was just updated
        return _job_row_to_detail(refreshed, registered_model_id=None)

    handle = reg.register(
        name=name,
        version=version,
        local_dir=local_dir,
        artifact_uri=outcome.artifact_uri,
        manifest=outcome.manifest,
        metrics=outcome.metrics,
    )

    registered = RegisteredModel(
        organization_id=organization_id,
        training_job_id=job_id,
        name=name,
        version=version,
        framework=outcome.framework,
        framework_version=outcome.framework_version,
        backend=outcome.backend,
        artifact_uri=handle.artifact_uri,
        local_dir=handle.local_dir,
        stage="staging",
        metrics=dict(outcome.metrics),
        manifest=dict(outcome.manifest),
        notes=request.notes,
    )
    session.add(registered)
    await session.flush()
    await session.refresh(registered)
    registered_model_id = registered.id
    await session.commit()

    if request.auto_promote:
        await _promote_internal(
            session=session,
            organization_id=organization_id,
            user_id=user_id,
            model_id=registered_model_id,
            stage="production",
            notes=request.notes,
        )

    log.info(
        "ml.train.persisted",
        job_id=str(job_id),
        registered_model_id=str(registered_model_id),
        name=name,
        version=version,
        f1=outcome.metrics.get("holdout_f1"),
    )

    job = await _load_job(
        session=session,
        organization_id=organization_id,
        job_id=job_id,
    )
    assert job is not None  # noqa: S101 — invariant: just persisted
    return _job_row_to_detail(
        job, registered_model_id=registered_model_id
    )


# ── Listing & detail ─────────────────────────────────────────────────────────


async def list_training_jobs(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    page: int = 1,
    page_size: int = 25,
) -> tuple[list[TrainingJobSummary], int]:
    page = max(page, 1)
    page_size = max(min(page_size, 100), 1)
    offset = (page - 1) * page_size

    stmt = (
        apply_tenant_filter(
            select(TrainingJob).order_by(TrainingJob.created_at.desc()),
            TrainingJob,
            organization_id,
        )
        .limit(page_size)
        .offset(offset)
    )
    rows = (await session.execute(stmt)).scalars().all()

    count_stmt = apply_tenant_filter(
        select(func.count(TrainingJob.id)),
        TrainingJob,
        organization_id,
    )
    total = int((await session.execute(count_stmt)).scalar_one())

    items = [_job_row_to_summary(r) for r in rows]
    return items, total


async def get_training_job(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    job_id: uuid.UUID,
) -> TrainingJobDetail:
    row = await _load_job(
        session=session,
        organization_id=organization_id,
        job_id=job_id,
    )
    if row is None:
        raise TrainingJobNotFoundError(f"training job {job_id} not found")
    rm_stmt = apply_tenant_filter(
        select(RegisteredModel.id).where(
            RegisteredModel.training_job_id == job_id
        ),
        RegisteredModel,
        organization_id,
    )
    rm_id: uuid.UUID | None = (await session.execute(rm_stmt)).scalar_one_or_none()
    return _job_row_to_detail(row, registered_model_id=rm_id)


async def list_registered_models(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[RegisteredModelSummary], int]:
    page = max(page, 1)
    page_size = max(min(page_size, 100), 1)
    offset = (page - 1) * page_size

    stmt = (
        apply_tenant_filter(
            select(RegisteredModel).order_by(RegisteredModel.created_at.desc()),
            RegisteredModel,
            organization_id,
        )
        .limit(page_size)
        .offset(offset)
    )
    rows = (await session.execute(stmt)).scalars().all()

    count_stmt = apply_tenant_filter(
        select(func.count(RegisteredModel.id)),
        RegisteredModel,
        organization_id,
    )
    total = int((await session.execute(count_stmt)).scalar_one())

    items = [_model_row_to_summary(r) for r in rows]
    return items, total


async def get_registered_model(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    model_id: uuid.UUID,
) -> RegisteredModelDetail:
    row = await _load_model(
        session=session,
        organization_id=organization_id,
        model_id=model_id,
    )
    if row is None:
        raise RegisteredModelNotFoundError(f"registered model {model_id} not found")
    return _model_row_to_detail(row)


# ── Promotion ────────────────────────────────────────────────────────────────


async def promote_model(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    user_id: uuid.UUID | None,
    model_id: uuid.UUID,
    request: PromoteModelRequest,
) -> RegisteredModelDetail:
    return await _promote_internal(
        session=session,
        organization_id=organization_id,
        user_id=user_id,
        model_id=model_id,
        stage=request.stage,
        notes=request.notes,
    )


async def get_production_model(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    name: str,
) -> RegisteredModel | None:
    stmt = apply_tenant_filter(
        select(RegisteredModel)
        .where(RegisteredModel.name == name)
        .where(RegisteredModel.stage == "production"),
        RegisteredModel,
        organization_id,
    ).order_by(RegisteredModel.promoted_at.desc())
    return (await session.execute(stmt)).scalar_one_or_none()


# ── Inference (predict) ──────────────────────────────────────────────────────


async def predict_with_model(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    model_id: uuid.UUID,
    request: RerankerPredictRequest,
    settings: Settings | None = None,
) -> RerankerPredictResponse:
    s = settings or get_settings()
    row = await _load_model(
        session=session,
        organization_id=organization_id,
        model_id=model_id,
    )
    if row is None:
        raise RegisteredModelNotFoundError(f"registered model {model_id} not found")
    if not row.local_dir:
        raise TrainingError(
            "model has no local artifact directory; cannot run inference"
        )
    reranker: Reranker = build_reranker_from_handle(
        name=row.name,
        version=row.version,
        local_dir=row.local_dir,
        settings=s,
    )
    result = reranker.score(
        RerankRequest(query=request.query, passages=tuple(request.passages))
    )
    scored = [
        ScoredPassage(index=i, passage=p, score=float(s_))
        for i, (p, s_) in enumerate(zip(request.passages, result.scores))
    ]
    scored.sort(key=lambda r: r.score, reverse=True)
    return RerankerPredictResponse(
        model_id=row.id,
        model_name=row.name,
        model_version=row.version,
        backend=result.backend,
        scored=scored,
    )


# ── Internal helpers ─────────────────────────────────────────────────────────


async def _mark_failed(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    job_id: uuid.UUID,
    error: str,
) -> None:
    row = await _load_job(
        session=session,
        organization_id=organization_id,
        job_id=job_id,
    )
    if row is None:
        return
    row.status = "failed"
    row.error_message = error[:600]
    row.finished_at = dt.datetime.now(dt.timezone.utc)
    await session.flush()
    await session.commit()


async def _promote_internal(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    user_id: uuid.UUID | None,
    model_id: uuid.UUID,
    stage: str,
    notes: str | None,
) -> RegisteredModelDetail:
    row = await _load_model(
        session=session,
        organization_id=organization_id,
        model_id=model_id,
    )
    if row is None:
        raise RegisteredModelNotFoundError(f"registered model {model_id} not found")
    if stage not in ("production", "archived"):
        raise ValueError(f"invalid stage {stage!r}")

    now = dt.datetime.now(dt.timezone.utc)

    if stage == "production":
        # Atomically archive the existing production model under the same
        # name. (One production-stage row per (org, name) at all times.)
        existing_stmt = apply_tenant_filter(
            select(RegisteredModel)
            .where(RegisteredModel.name == row.name)
            .where(RegisteredModel.stage == "production")
            .where(RegisteredModel.id != model_id),
            RegisteredModel,
            organization_id,
        )
        existing = (await session.execute(existing_stmt)).scalars().all()
        for old in existing:
            old.stage = "archived"
            old.archived_at = now
            log.info(
                "ml.model.archived_for_production",
                model_id=str(old.id),
                name=old.name,
                version=old.version,
            )

        row.stage = "production"
        row.promoted_at = now
        row.promoted_by = user_id
        if notes:
            row.notes = notes
        # Drop the in-process reranker cache so the next inquiry / predict
        # call picks up the freshly promoted model.
        reset_reranker_cache_for_tests()
    else:  # archived
        row.stage = "archived"
        row.archived_at = now
        if notes:
            row.notes = notes

    await session.flush()
    await session.refresh(row)
    await session.commit()

    log.info(
        "ml.model.staged",
        model_id=str(row.id),
        name=row.name,
        version=row.version,
        stage=row.stage,
    )
    return _model_row_to_detail(row)


async def _load_job(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    job_id: uuid.UUID,
) -> TrainingJob | None:
    stmt = apply_tenant_filter(
        select(TrainingJob).where(TrainingJob.id == job_id),
        TrainingJob,
        organization_id,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def _load_model(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    model_id: uuid.UUID,
) -> RegisteredModel | None:
    stmt = apply_tenant_filter(
        select(RegisteredModel).where(RegisteredModel.id == model_id),
        RegisteredModel,
        organization_id,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


def _metrics_view(metrics: dict[str, Any]) -> TrainingJobMetricsOut:
    flat: dict[str, Any] = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[k] = v
    return TrainingJobMetricsOut(**flat)


def _job_row_to_summary(row: TrainingJob) -> TrainingJobSummary:
    metrics = row.metrics if isinstance(row.metrics, dict) else {}
    return TrainingJobSummary(
        job_id=row.id,
        name=row.name,
        version=row.version,
        backend=row.backend,
        framework=row.framework,
        status=_coerce_status(row.status),
        duration_s=float(row.duration_s or 0.0),
        holdout_f1=float(metrics.get("holdout_f1") or 0.0),
        holdout_roc_auc=float(metrics.get("holdout_roc_auc") or 0.0),
        score_separation=float(metrics.get("score_separation") or 0.0),
        n_train=int(metrics.get("n_train") or 0),
        error_message=row.error_message,
        mlflow_run_id=row.mlflow_run_id,
        created_at=row.created_at,
    )


def _job_row_to_detail(
    row: TrainingJob, *, registered_model_id: uuid.UUID | None
) -> TrainingJobDetail:
    return TrainingJobDetail(
        job_id=row.id,
        organization_id=row.organization_id,
        triggered_by=row.triggered_by,
        name=row.name,
        version=row.version,
        backend=row.backend,
        framework=row.framework,
        framework_version=row.framework_version,
        status=_coerce_status(row.status),
        artifact_uri=row.artifact_uri,
        external_job_id=row.external_job_id,
        config=row.config if isinstance(row.config, dict) else {},
        metrics=_metrics_view(
            row.metrics if isinstance(row.metrics, dict) else {}
        ),
        manifest=row.manifest if isinstance(row.manifest, dict) else {},
        log_excerpt=row.log_excerpt,
        duration_s=float(row.duration_s or 0.0),
        mlflow_run_id=row.mlflow_run_id,
        error_message=row.error_message,
        started_at=row.started_at,
        finished_at=row.finished_at,
        created_at=row.created_at,
        registered_model_id=registered_model_id,
    )


def _model_row_to_summary(row: RegisteredModel) -> RegisteredModelSummary:
    metrics = row.metrics if isinstance(row.metrics, dict) else {}
    return RegisteredModelSummary(
        model_id=row.id,
        name=row.name,
        version=row.version,
        framework=row.framework,
        backend=row.backend,
        stage=_coerce_stage(row.stage),
        holdout_f1=float(metrics.get("holdout_f1") or 0.0),
        holdout_roc_auc=float(metrics.get("holdout_roc_auc") or 0.0),
        score_separation=float(metrics.get("score_separation") or 0.0),
        n_train=int(metrics.get("n_train") or 0),
        artifact_uri=row.artifact_uri,
        training_job_id=row.training_job_id,
        created_at=row.created_at,
        promoted_at=row.promoted_at,
        archived_at=row.archived_at,
    )


def _model_row_to_detail(row: RegisteredModel) -> RegisteredModelDetail:
    return RegisteredModelDetail(
        model_id=row.id,
        organization_id=row.organization_id,
        name=row.name,
        version=row.version,
        framework=row.framework,
        framework_version=row.framework_version,
        backend=row.backend,
        artifact_uri=row.artifact_uri,
        local_dir=row.local_dir,
        stage=_coerce_stage(row.stage),
        metrics=_metrics_view(
            row.metrics if isinstance(row.metrics, dict) else {}
        ),
        manifest=row.manifest if isinstance(row.manifest, dict) else {},
        training_job_id=row.training_job_id,
        promoted_by=row.promoted_by,
        notes=row.notes,
        promoted_at=row.promoted_at,
        archived_at=row.archived_at,
        created_at=row.created_at,
    )


def _coerce_status(status: str) -> str:
    if status in ("pending", "running", "success", "failed"):
        return status
    return "pending"


def _coerce_stage(stage: str) -> str:
    if stage in ("staging", "production", "archived"):
        return stage
    return "staging"


__all__ = [
    "RegisteredModelNotFoundError",
    "TrainingError",
    "TrainingJobNotFoundError",
    "get_production_model",
    "get_registered_model",
    "get_training_job",
    "list_registered_models",
    "list_training_jobs",
    "predict_with_model",
    "promote_model",
    "submit_training_job_for_tenant",
]
