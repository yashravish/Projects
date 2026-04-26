from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.defect import Defect, DefectPriority, DefectSeverity, DefectStatus
from app.schemas.defects import DefectCreate, DefectRead, DefectStatsRead, DefectUpdate

router = APIRouter(prefix="/api/defects", tags=["defects"])


@router.get("", response_model=list[DefectRead])
async def list_defects(
    session: AsyncSession = Depends(get_db),
    project_id: int | None = None,
    status: str | None = None,
    limit: int = Query(100, le=500),
) -> list[DefectRead]:
    q = select(Defect)
    if project_id is not None:
        q = q.where(Defect.project_id == project_id)
    if status is not None:
        q = q.where(Defect.status == status)
    r = await session.execute(
        q.order_by(Defect.id.desc())
        .limit(limit)
        .options(selectinload(Defect.linked_articles))
    )
    return [_to_read(d) for d in r.scalars().all()]


@router.get("/stats", response_model=DefectStatsRead)
async def stats(
    session: AsyncSession = Depends(get_db), project_id: int | None = None
) -> DefectStatsRead:
    q = select(Defect) if project_id is None else select(Defect).where(Defect.project_id == project_id)
    all_rows = (await session.execute(q)).scalars().all()
    open_c = sum(1 for d in all_rows if d.status in (DefectStatus.open, DefectStatus.in_progress))
    res_c = sum(1 for d in all_rows if d.status in (DefectStatus.resolved, DefectStatus.closed))
    total = len(all_rows) or 1
    by_sev: dict[str, int] = {}
    for d in all_rows:
        by_sev[d.severity.value] = by_sev.get(d.severity.value, 0) + 1
    return DefectStatsRead(
        open=open_c,
        resolved=res_c,
        defect_rate=round(open_c / total, 4),
        by_severity=by_sev,
    )


@router.get("/{defect_id}", response_model=DefectRead)
async def get_defect(
    defect_id: int, session: AsyncSession = Depends(get_db)
) -> DefectRead:
    d = (await session.execute(
        select(Defect).options(selectinload(Defect.linked_articles)).where(Defect.id == defect_id)
    )).scalar_one_or_none()
    if not d:
        raise HTTPException(404, "Defect not found")
    return _to_read(d)


@router.post("", response_model=DefectRead, status_code=201)
async def create_defect(
    body: DefectCreate, session: AsyncSession = Depends(get_db)
) -> DefectRead:
    d = Defect(
        project_id=body.project_id,
        title=body.title,
        description=body.description,
        severity=DefectSeverity(body.severity),
        priority=DefectPriority(body.priority),
        status=DefectStatus(body.status),
        owner=body.owner,
        root_cause=body.root_cause,
        suggested_fix=body.suggested_fix,
        linked_pipeline_run_id=body.linked_pipeline_run_id,
        ai_report_id=body.ai_report_id,
    )
    if body.linked_kb_article_ids:
        from app.models.knowledge_base import KnowledgeBaseArticle
        for kid in set(body.linked_kb_article_ids):
            a = await session.get(KnowledgeBaseArticle, kid)
            if a:
                d.linked_articles.append(a)
    session.add(d)
    await session.flush()
    d2 = (await session.execute(
        select(Defect).options(selectinload(Defect.linked_articles)).where(Defect.id == d.id)
    )).scalar_one()
    return _to_read(d2)


@router.patch("/{defect_id}", response_model=DefectRead)
async def update_defect(
    defect_id: int, body: DefectUpdate, session: AsyncSession = Depends(get_db)
) -> DefectRead:
    d = (await session.execute(
        select(Defect).options(selectinload(Defect.linked_articles)).where(Defect.id == defect_id)
    )).scalar_one_or_none()
    if not d:
        raise HTTPException(404, "Defect not found")
    u = body.model_dump(exclude_unset=True)
    if "linked_kb_article_ids" in u:
        new_ids = u.pop("linked_kb_article_ids") or []
        d.linked_articles.clear()
        from app.models.knowledge_base import KnowledgeBaseArticle
        for kid in new_ids:
            a = await session.get(KnowledgeBaseArticle, kid)
            if a:
                d.linked_articles.append(a)
    for k, v in u.items():
        if hasattr(d, k):
            setattr(d, k, v)
    await session.flush()
    return _to_read(d)


def _to_read(d: Defect) -> DefectRead:
    ids = [a.id for a in (d.linked_articles or [])]
    return DefectRead(
        id=d.id,
        project_id=d.project_id,
        title=d.title,
        description=d.description,
        severity=d.severity.value,
        priority=d.priority.value,
        status=d.status.value,
        owner=d.owner,
        root_cause=d.root_cause,
        suggested_fix=d.suggested_fix,
        linked_pipeline_run_id=d.linked_pipeline_run_id,
        ai_report_id=d.ai_report_id,
        created_at=d.created_at,
        updated_at=d.updated_at,
        resolved_at=d.resolved_at,
        linked_kb_article_ids=ids,
    )
