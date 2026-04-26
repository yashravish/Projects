from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.knowledge_base import ArticleType, KnowledgeBaseArticle
from app.schemas.knowledge import (
    KnowledgeArticleCreate,
    KnowledgeArticleRead,
    KnowledgeArticleUpdate,
)

router = APIRouter(prefix="/api/kb", tags=["knowledge"])


@router.get("", response_model=list[KnowledgeArticleRead])
async def list_articles(
    session: AsyncSession = Depends(get_db),
    q: str | None = Query(None, description="search title"),
    article_type: str | None = Query(None, alias="type"),
) -> list[KnowledgeArticleRead]:
    stmt = select(KnowledgeBaseArticle)
    if article_type:
        stmt = stmt.where(KnowledgeBaseArticle.type == article_type)
    if q:
        stmt = stmt.where(KnowledgeBaseArticle.title.ilike(f"%{q}%"))
    r = await session.execute(stmt.order_by(KnowledgeBaseArticle.id.desc()).limit(200))
    return [KnowledgeArticleRead.model_validate(x) for x in r.scalars().all()]


@router.get("/{article_id}", response_model=KnowledgeArticleRead)
async def get_article(article_id: int, session: AsyncSession = Depends(get_db)) -> KnowledgeArticleRead:
    a = await session.get(KnowledgeBaseArticle, article_id)
    if not a:
        raise HTTPException(404, "Article not found")
    return KnowledgeArticleRead.model_validate(a)


@router.post("", response_model=KnowledgeArticleRead, status_code=201)
async def create_article(
    body: KnowledgeArticleCreate, session: AsyncSession = Depends(get_db)
) -> KnowledgeArticleRead:
    try:
        at = ArticleType(body.type)
    except Exception as exc:
        raise HTTPException(400, f"Invalid type: {body.type}") from exc
    a = KnowledgeBaseArticle(
        title=body.title,
        slug=body.slug,
        type=at,
        content=body.content,
        tags=body.tags,
    )
    session.add(a)
    await session.flush()
    return KnowledgeArticleRead.model_validate(a)


@router.patch("/{article_id}", response_model=KnowledgeArticleRead)
async def update_article(
    article_id: int, body: KnowledgeArticleUpdate, session: AsyncSession = Depends(get_db)
) -> KnowledgeArticleRead:
    a = await session.get(KnowledgeBaseArticle, article_id)
    if not a:
        raise HTTPException(404, "not found")
    u = body.model_dump(exclude_unset=True)
    if "type" in u and u["type"] is not None:
        a.type = ArticleType(u["type"])
        del u["type"]
    for k, v in u.items():
        setattr(a, k, v)
    await session.flush()
    return KnowledgeArticleRead.model_validate(a)
