"""Audit ledger & retention API.

Routes:

    GET    /audit/events                    — paginated event list (filterable)
    GET    /audit/events/{event_id}         — single event detail
    GET    /audit/events.csv                — streaming CSV export

    GET    /audit/integrity                 — chain verification report
    GET    /audit/stats                     — header strip metrics

    GET    /audit/policies                  — list retention policies
    PUT    /audit/policies/{resource_type}  — upsert (admin)

    GET    /audit/retention/runs            — sweep history
    POST   /audit/retention/runs            — execute a sweep (admin)

The list / detail / integrity / stats endpoints are open to any
authenticated user; mutations (policy upserts, retention sweeps) require
the `admin` role. Every mutation also emits an audit event of its own
under `action="audit.*"` so the page that renders the ledger is itself
audited.
"""
from __future__ import annotations

import datetime as dt
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.db.models import User
from app.deps import CurrentUser, SessionDep, require_role
from app.observability import audit_emitter
from app.schemas.audit import (
    AuditEventFilters,
    AuditEventList,
    AuditEventOut,
    AuditOutcome,
    IntegrityReport,
    LedgerStats,
    RetentionPolicyList,
    RetentionPolicyOut,
    RetentionPolicyUpsert,
    RetentionRunList,
    RetentionRunOut,
)
from app.services import audit_service

router = APIRouter(prefix="/audit", tags=["audit"])


def _audit_event_filters(
    *,
    actions: list[str] | None,
    resource_types: list[str] | None,
    outcomes: list[AuditOutcome] | None,
    actor_ids: list[uuid.UUID] | None,
    since: dt.datetime | None,
    until: dt.datetime | None,
    search: str | None,
    page: int,
    page_size: int,
) -> AuditEventFilters:
    """Shared by paginated list and CSV export — same filter surface."""
    return AuditEventFilters(
        actions=actions,
        resource_types=resource_types,
        outcomes=outcomes,
        actor_ids=actor_ids,
        since=since,
        until=until,
        search=search,
        page=page,
        page_size=page_size,
    )


# ── Events ──────────────────────────────────────────────────────────────────


@router.get("/events", response_model=AuditEventList)
async def list_events(
    session: SessionDep,
    user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    actions: list[str] | None = Query(default=None),
    resource_types: list[str] | None = Query(default=None),
    outcomes: list[AuditOutcome] | None = Query(default=None),
    actor_ids: list[uuid.UUID] | None = Query(default=None),
    since: dt.datetime | None = Query(default=None),
    until: dt.datetime | None = Query(default=None),
    search: str | None = Query(default=None, max_length=200),
) -> AuditEventList:
    filters = _audit_event_filters(
        actions=actions,
        resource_types=resource_types,
        outcomes=outcomes,
        actor_ids=actor_ids,
        since=since,
        until=until,
        search=search,
        page=page,
        page_size=page_size,
    )
    return await audit_service.list_events(
        session=session,
        organization_id=user.organization_id,
        filters=filters,
    )


@router.get("/events/{event_id}", response_model=AuditEventOut)
async def get_event(
    event_id: uuid.UUID,
    session: SessionDep,
    user: CurrentUser,
) -> AuditEventOut:
    try:
        return await audit_service.get_event(
            session=session,
            organization_id=user.organization_id,
            event_id=event_id,
        )
    except audit_service.AuditEventNotFoundError as exc:
        raise HTTPException(status_code=404, detail="audit event not found") from exc


@router.get("/events.csv")
async def export_events_csv(
    session: SessionDep,
    user: CurrentUser,
    actions: list[str] | None = Query(default=None),
    resource_types: list[str] | None = Query(default=None),
    outcomes: list[AuditOutcome] | None = Query(default=None),
    actor_ids: list[uuid.UUID] | None = Query(default=None),
    since: dt.datetime | None = Query(default=None),
    until: dt.datetime | None = Query(default=None),
    search: str | None = Query(default=None, max_length=200),
) -> Response:
    filters = _audit_event_filters(
        actions=actions,
        resource_types=resource_types,
        outcomes=outcomes,
        actor_ids=actor_ids,
        since=since,
        until=until,
        search=search,
        page=1,
        page_size=200,  # ignored by exporter; row cap is enforced server-side
    )
    body = await audit_service.export_events_csv(
        session=session,
        organization_id=user.organization_id,
        filters=filters,
    )
    await audit_emitter.emit(
        session=session,
        organization_id=user.organization_id,
        actor_id=user.id,
        action="audit.export",
        resource_type="audit_export",
        resource_id=None,
        outcome="success",
        metadata={"size_bytes": len(body)},
    )
    return Response(
        content=body,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="audit-ledger.csv"',
        },
    )


# ── Stats & integrity ───────────────────────────────────────────────────────


@router.get("/stats", response_model=LedgerStats)
async def stats(session: SessionDep, user: CurrentUser) -> LedgerStats:
    return await audit_service.compute_ledger_stats(
        session=session,
        organization_id=user.organization_id,
    )


@router.get("/integrity", response_model=IntegrityReport)
async def integrity(session: SessionDep, user: CurrentUser) -> IntegrityReport:
    report = await audit_service.verify_chain(
        session=session,
        organization_id=user.organization_id,
    )
    await audit_emitter.emit(
        session=session,
        organization_id=user.organization_id,
        actor_id=user.id,
        action="audit.integrity.verify",
        resource_type="audit_log",
        resource_id=None,
        outcome="success" if report.chain_ok else "error",
        metadata={
            "total_events": report.total_events,
            "chain_ok": report.chain_ok,
            "n_breaks": len(report.breaks),
            "head_hash": report.head_hash,
            "tail_hash": report.tail_hash,
        },
    )
    return report


# ── Retention policies ──────────────────────────────────────────────────────


@router.get("/policies", response_model=RetentionPolicyList)
async def list_policies(
    session: SessionDep, user: CurrentUser
) -> RetentionPolicyList:
    items = await audit_service.list_policies(
        session=session, organization_id=user.organization_id
    )
    return RetentionPolicyList(items=items)


@router.put(
    "/policies/{resource_type}",
    response_model=RetentionPolicyOut,
)
async def upsert_policy(
    resource_type: str,
    payload: RetentionPolicyUpsert,
    session: SessionDep,
    user: User = Depends(require_role("admin")),
) -> RetentionPolicyOut:
    try:
        out = await audit_service.upsert_policy(
            session=session,
            organization_id=user.organization_id,
            resource_type=resource_type,
            request=payload,
        )
    except audit_service.AuditError as exc:
        await audit_emitter.emit(
            session=session,
            organization_id=user.organization_id,
            actor_id=user.id,
            action="retention.policy.upsert",
            resource_type="retention_policy",
            resource_id=None,
            outcome="error",
            metadata={
                "resource_type": resource_type,
                "ttl_days": payload.ttl_days,
                "reason": exc.message[:200],
            },
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    await session.commit()

    await audit_emitter.emit(
        session=session,
        organization_id=user.organization_id,
        actor_id=user.id,
        action="retention.policy.upsert",
        resource_type="retention_policy",
        resource_id=out.policy_id,
        outcome="success",
        metadata={
            "resource_type": out.resource_type,
            "ttl_days": out.ttl_days,
            "is_active": out.is_active,
        },
    )
    return out


# ── Retention runs ──────────────────────────────────────────────────────────


@router.get("/retention/runs", response_model=RetentionRunList)
async def list_retention_runs(
    session: SessionDep,
    user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
) -> RetentionRunList:
    return await audit_service.list_retention_runs(
        session=session,
        organization_id=user.organization_id,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/retention/runs",
    response_model=RetentionRunOut,
    status_code=status.HTTP_200_OK,
)
async def run_retention(
    session: SessionDep,
    user: User = Depends(require_role("admin")),
) -> RetentionRunOut:
    try:
        out = await audit_service.run_retention(
            session=session,
            organization_id=user.organization_id,
            user_id=user.id,
        )
    except audit_service.AuditError as exc:
        await audit_emitter.emit(
            session=session,
            organization_id=user.organization_id,
            actor_id=user.id,
            action="retention.run",
            resource_type="retention_run",
            resource_id=None,
            outcome="error",
            metadata={"reason": exc.message[:200]},
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    await audit_emitter.emit(
        session=session,
        organization_id=user.organization_id,
        actor_id=user.id,
        action="retention.run",
        resource_type="retention_run",
        resource_id=out.run_id,
        outcome="success" if out.status == "success" else "error",
        metadata={
            "status": out.status,
            "purged_counts": out.purged_counts,
        },
    )
    return out
