from __future__ import annotations

import datetime as dt
import enum
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, String, Text, Enum as PgEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.associations import analysis_kb_article
from app.core.database import Base

if TYPE_CHECKING:
    from app.models.defect import Defect
    from app.models.knowledge_base import KnowledgeBaseArticle


class ReportSeverity(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


def _e(e: type[enum.Enum], name: str) -> PgEnum:
    return PgEnum(e, name=name, values_callable=lambda x: [m.value for m in x], native_enum=False)


class AIAnalysisReport(Base):
    __tablename__ = "ai_analysis_reports"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    log_snippet: Mapped[str] = mapped_column(Text)
    log_hash: Mapped[str] = mapped_column(String(64), index=True)
    root_cause_summary: Mapped[str] = mapped_column(Text)
    likely_file_or_component: Mapped[str] = mapped_column(String(512), default="unknown")
    suggested_fix: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[ReportSeverity] = mapped_column(
        _e(ReportSeverity, "reportsev"), default=ReportSeverity.medium
    )
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    project_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )

    created_defect: Mapped[Optional["Defect"]] = relationship(
        "Defect", back_populates="source_analysis", uselist=False
    )
    linked_articles: Mapped[List["KnowledgeBaseArticle"]] = relationship(
        secondary=analysis_kb_article,
        back_populates="linked_analyses",
    )
