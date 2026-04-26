from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.project import Project
from app.schemas.projects import ProjectCreate, ProjectList, ProjectRead

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=ProjectList)
async def list_projects(
    session: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> ProjectList:
    count_q = await session.execute(select(func.count()).select_from(Project))
    total = int(count_q.scalar_one())
    res = await session.execute(select(Project).offset(skip).limit(limit).order_by(Project.id))
    items = [ProjectRead.model_validate(p) for p in res.scalars().all()]
    return ProjectList(items=items, total=total)


@router.post("", response_model=ProjectRead, status_code=201)
async def create_project(
    body: ProjectCreate, session: AsyncSession = Depends(get_db)
) -> ProjectRead:
    ex = await session.execute(select(Project).where(Project.slug == body.slug))
    if ex.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Project slug already exists")
    p = Project(name=body.name, slug=body.slug, description=body.description)
    session.add(p)
    await session.flush()
    await session.refresh(p)
    return ProjectRead.model_validate(p)
