"""Evaluation orchestration.

Exposes three operations the API uses:

* `run_evaluation` — drive the gold dataset through `GraphRunner`, score, log
  to MLflow, persist an `EvaluationRun`, return a fully-shaped detail.
* `list_evaluations` — paginated, tenant-scoped summary list.
* `get_evaluation` — single-row detail rebuild from JSONB.

Tenant isolation: every persistence and read goes through `apply_tenant_filter`,
matching the audit invariant enforced by `tests/unit/test_tenant_isolation.py`.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import GraphRunner
from app.agents.llm_client import Embedder, LLMClient
from app.agents.prompts import all_prompt_versions
from app.config import get_settings
from app.db.models import EvaluationRun
from app.eval import dataset as ds_mod
from app.eval import run_evaluation as run_dataset
from app.logging_config import get_logger
from app.observability.mlflow_client import MLflowRunRecorder, get_mlflow_recorder
from app.retrieval.hybrid import HybridRetriever, RetrievalConfig
from app.schemas.evaluation import (
    AggregateMetricsOut,
    DatasetOut,
    EvaluationItemOut,
    EvaluationRunDetail,
    EvaluationRunRequest,
    EvaluationRunSummary,
)
from app.security.tenant import apply_tenant_filter

log = get_logger("evaluation_service")


class EvaluationRunNotFoundError(Exception):
    """Raised when an evaluation run does not exist for the supplied tenant."""


def get_default_dataset_view() -> DatasetOut:
    """Public peek at the dataset (tenant-agnostic)."""
    return DatasetOut.from_dataset(ds_mod.GOLD_DATASET)


async def run_evaluation_for_tenant(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    user_id: uuid.UUID | None,
    request: EvaluationRunRequest,
    llm: LLMClient,
    embedder: Embedder,
    recorder: MLflowRunRecorder | None = None,
) -> EvaluationRunDetail:
    """End-to-end evaluation: dataset → graph → score → MLflow → persist."""
    settings = get_settings()
    cfg = RetrievalConfig(top_k=request.top_k, candidate_k=request.candidate_k)
    retriever = HybridRetriever(session=session, embedder=embedder)
    runner = GraphRunner(
        llm=llm,
        retriever=retriever,
        retrieval_config=cfg,
        model=settings.openai_default_model,
    )

    dataset = ds_mod.get_dataset(request.dataset_name)
    rec = recorder or get_mlflow_recorder()
    mlflow_run_id: str | None = None

    # Insert a `pending` row up front so a long eval is observable in the
    # listing while it runs. We update aggregate + status in place when done.
    pending = EvaluationRun(
        organization_id=organization_id,
        triggered_by=user_id,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        model_name=runner.model,
        prompt_versions=all_prompt_versions(),
        retrieval_config={
            **cfg.as_dict(),
            "embedder": embedder.backend_name,
            "llm": llm.backend_name,
        },
        aggregate_metrics={},
        per_item_results=[],
        status="running",
    )
    session.add(pending)
    await session.flush()
    await session.refresh(pending)
    pending_id = pending.id
    await session.commit()

    log.info(
        "eval.run.started",
        run_id=str(pending_id),
        dataset=dataset.name,
        dataset_version=dataset.version,
        organization_id=str(organization_id),
        n_items=len(dataset),
    )

    try:
        with rec.record_evaluation(
            organization_id=str(organization_id),
            run_name=f"eval-{dataset.name}-{uuid.uuid4().hex[:8]}",
            tags={
                "dataset_name": dataset.name,
                "dataset_version": dataset.version,
                "model": runner.model,
                "embedder": embedder.backend_name,
                "llm": llm.backend_name,
            },
        ) as active:
            mlflow_run_id = active.run_id

            outcome = await run_dataset(
                runner=runner,
                dataset=dataset,
                organization_id=organization_id,
                user_id=user_id,
            )

            rec.log_params(
                {
                    "model": runner.model,
                    "dataset_name": dataset.name,
                    "dataset_version": dataset.version,
                    "n_items": len(dataset),
                    "top_k": cfg.top_k,
                    "candidate_k": cfg.candidate_k,
                    "embedder": embedder.backend_name,
                    "llm": llm.backend_name,
                    **{f"prompt.{k}": v for k, v in all_prompt_versions().items()},
                }
            )
            rec.log_metrics(
                {f"agg.{k}": float(v) for k, v in _flat_numeric(outcome.aggregate.as_dict()).items()}
            )
            rec.log_dict(
                outcome.as_dict(),
                artifact_file="evaluation.json",
            )
    except Exception as exc:  # noqa: BLE001 — record failure on the row
        log.exception(
            "eval.run.failed",
            run_id=str(pending_id),
            organization_id=str(organization_id),
        )
        # Mark the row as failed and surface the error.
        await _mark_failed(
            session=session,
            organization_id=organization_id,
            run_id=pending_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    # ── Persist the completed run ─────────────────────────────────────────
    aggregate_payload = outcome.aggregate.as_dict()
    aggregate_payload["wall_time_ms"] = outcome.wall_time_ms
    per_item_payload = [it.as_dict() for it in outcome.items]

    stmt = apply_tenant_filter(
        select(EvaluationRun).where(EvaluationRun.id == pending_id),
        EvaluationRun,
        organization_id,
    )
    row: EvaluationRun | None = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        # Should be impossible — we just inserted it. Defend anyway.
        raise RuntimeError(f"evaluation run {pending_id} disappeared mid-flight")

    row.aggregate_metrics = aggregate_payload
    row.per_item_results = per_item_payload
    row.mlflow_run_id = mlflow_run_id
    row.status = "success"
    await session.flush()
    await session.refresh(row)
    await session.commit()

    log.info(
        "eval.run.persisted",
        run_id=str(row.id),
        organization_id=str(organization_id),
        pass_rate=outcome.aggregate.pass_rate,
        n_items=outcome.aggregate.n_items,
        n_failures=outcome.aggregate.n_failures,
        mlflow_run_id=mlflow_run_id,
    )

    return EvaluationRunDetail.from_outcome(
        run_id=row.id,
        outcome=outcome,
        prompt_versions=row.prompt_versions,
        retrieval_config=row.retrieval_config,
        mlflow_run_id=mlflow_run_id,
        created_at=row.created_at,
        status="success",
    )


# ── List / get ───────────────────────────────────────────────────────────────


async def list_evaluations(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    page: int = 1,
    page_size: int = 25,
) -> tuple[list[EvaluationRunSummary], int]:
    page = max(page, 1)
    page_size = max(min(page_size, 100), 1)
    offset = (page - 1) * page_size

    stmt = (
        apply_tenant_filter(
            select(EvaluationRun).order_by(EvaluationRun.created_at.desc()),
            EvaluationRun,
            organization_id,
        )
        .limit(page_size)
        .offset(offset)
    )
    rows = (await session.execute(stmt)).scalars().all()

    count_stmt = apply_tenant_filter(
        select(func.count(EvaluationRun.id)),
        EvaluationRun,
        organization_id,
    )
    total = int((await session.execute(count_stmt)).scalar_one())

    items: list[EvaluationRunSummary] = []
    for r in rows:
        agg = r.aggregate_metrics if isinstance(r.aggregate_metrics, dict) else {}
        items.append(
            EvaluationRunSummary(
                run_id=r.id,
                dataset_name=r.dataset_name,
                dataset_version=r.dataset_version,
                model=r.model_name,
                status=_coerce_status(r.status),
                n_items=int(agg.get("n_items") or 0),
                pass_rate=float(agg.get("pass_rate") or 0.0),
                grounding_score=float(agg.get("grounding_score") or 0.0),
                faithfulness=float(agg.get("faithfulness") or 0.0),
                retrieval_recall=float(agg.get("retrieval_recall") or 0.0),
                latency_ms_p50=float(agg.get("latency_ms_p50") or 0.0),
                mlflow_run_id=r.mlflow_run_id,
                created_at=r.created_at,
            )
        )
    return items, total


async def get_evaluation(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    run_id: uuid.UUID,
) -> EvaluationRunDetail:
    stmt = apply_tenant_filter(
        select(EvaluationRun).where(EvaluationRun.id == run_id),
        EvaluationRun,
        organization_id,
    )
    row: EvaluationRun | None = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise EvaluationRunNotFoundError(f"evaluation run {run_id} not found")

    return _row_to_detail(row)


# ── Internal ─────────────────────────────────────────────────────────────────


async def _mark_failed(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    run_id: uuid.UUID,
    error: str,
) -> None:
    stmt = apply_tenant_filter(
        select(EvaluationRun).where(EvaluationRun.id == run_id),
        EvaluationRun,
        organization_id,
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        return
    row.status = "failed"
    agg = row.aggregate_metrics if isinstance(row.aggregate_metrics, dict) else {}
    row.aggregate_metrics = {**agg, "error": error}
    await session.flush()
    await session.commit()


def _row_to_detail(row: EvaluationRun) -> EvaluationRunDetail:
    agg_raw = row.aggregate_metrics if isinstance(row.aggregate_metrics, dict) else {}
    items_raw = row.per_item_results if isinstance(row.per_item_results, list) else []

    aggregate = AggregateMetricsOut(
        n_items=int(agg_raw.get("n_items") or 0),
        pass_rate=float(agg_raw.get("pass_rate") or 0.0),
        retrieval_recall=float(agg_raw.get("retrieval_recall") or 0.0),
        retrieval_precision=float(agg_raw.get("retrieval_precision") or 0.0),
        citation_precision=float(agg_raw.get("citation_precision") or 0.0),
        citation_recall=float(agg_raw.get("citation_recall") or 0.0),
        faithfulness=float(agg_raw.get("faithfulness") or 0.0),
        forbidden_phrase_rate=float(agg_raw.get("forbidden_phrase_rate") or 0.0),
        grounding_score=float(agg_raw.get("grounding_score") or 0.0),
        hallucination_risk=float(agg_raw.get("hallucination_risk") or 0.0),
        latency_ms_p50=float(agg_raw.get("latency_ms_p50") or 0.0),
        latency_ms_p95=float(agg_raw.get("latency_ms_p95") or 0.0),
        n_failures=int(agg_raw.get("n_failures") or 0),
    )
    items = [EvaluationItemOut.from_persisted(it) for it in items_raw if isinstance(it, dict)]

    return EvaluationRunDetail(
        run_id=row.id,
        dataset_name=row.dataset_name,
        dataset_version=row.dataset_version,
        model=row.model_name,
        status=_coerce_status(row.status),
        aggregate=aggregate,
        items=items,
        prompt_versions=row.prompt_versions if isinstance(row.prompt_versions, dict) else {},
        retrieval_config=row.retrieval_config if isinstance(row.retrieval_config, dict) else {},
        wall_time_ms=int(agg_raw.get("wall_time_ms") or 0),
        mlflow_run_id=row.mlflow_run_id,
        created_at=row.created_at,
    )


def _coerce_status(status: str) -> str:
    if status in ("pending", "running", "success", "failed"):
        return status
    return "success"


def _flat_numeric(d: dict[str, Any]) -> dict[str, float]:
    """Filter a dict down to its numeric leaves so MLflow.log_metrics is happy."""
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, bool):
            continue  # booleans surface as 0/1 elsewhere; skip here
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


__all__ = [
    "EvaluationRunNotFoundError",
    "get_default_dataset_view",
    "get_evaluation",
    "list_evaluations",
    "run_evaluation_for_tenant",
]
