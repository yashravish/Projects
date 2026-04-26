from __future__ import annotations

import datetime as dt
import enum
from typing import List, TYPE_CHECKING

from sqlalchemy import Enum as PgEnum, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.associations import analysis_kb_article, defect_kb_article

if TYPE_CHECKING:
    from app.models.defect import Defect
    from app.models.ai_report import AIAnalysisReport


class ArticleType(str, enum.Enum):
    runbook = "runbook"
    postmortem = "postmortem"
    pipeline_setup = "pipeline_setup"
    troubleshooting = "troubleshooting"
    other = "other"


def _e(e: type[enum.Enum], name: str) -> PgEnum:
    return PgEnum(e, name=name, values_callable=lambda x: [m.value for m in x], native_enum=False)


class KnowledgeBaseArticle(Base):
    __tablename__ = "knowledge_base_articles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(300), index=True)
    slug: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    type: Mapped[ArticleType] = mapped_column(
        _e(ArticleType, "kbatype"), default=ArticleType.troubleshooting
    )
    content: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
    )

    linked_defects: Mapped[List["Defect"]] = relationship(
        secondary=defect_kb_article,
        back_populates="linked_articles",
    )
    linked_analyses: Mapped[List["AIAnalysisReport"]] = relationship(
        secondary=analysis_kb_article,
        back_populates="linked_articles",
    )
