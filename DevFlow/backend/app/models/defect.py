from __future__ import annotations

import datetime as dt
import enum
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import Enum as PgEnum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.associations import defect_kb_article

if TYPE_CHECKING:
    from app.models.knowledge_base import KnowledgeBaseArticle
    from app.models.pipeline import PipelineRun


class DefectSeverity(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class DefectPriority(str, enum.Enum):
    p0 = "p0"
    p1 = "p1"
    p2 = "p2"
    p3 = "p3"


class DefectStatus(str, enum.Enum):
    open = "open"
    in_progress = "in_progress"
    resolved = "resolved"
    closed = "closed"


def _e(e: type[enum.Enum], name: str) -> PgEnum:
    return PgEnum(e, name=name, values_callable=lambda x: [m.value for m in x], native_enum=False)


class Defect(Base):
    __tablename__ = "defects"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(400))
    description: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[DefectSeverity] = mapped_column(
        _e(DefectSeverity, "defsev"), default=DefectSeverity.medium
    )
    priority: Mapped[DefectPriority] = mapped_column(
        _e(DefectPriority, "defpri"), default=DefectPriority.p2
    )
    status: Mapped[DefectStatus] = mapped_column(
        _e(DefectStatus, "defstat"), default=DefectStatus.open
    )
    owner: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    root_cause: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    suggested_fix: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    linked_pipeline_run_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    ai_report_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("ai_analysis_reports.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )
    resolved_at: Mapped[Optional[dt.datetime]] = mapped_column(nullable=True)

    project = relationship("Project", back_populates="defects")
    pipeline_run = relationship(
        "PipelineRun", backref="linked_defects", foreign_keys=[linked_pipeline_run_id]
    )
    source_analysis: Mapped[Optional["AIAnalysisReport"]] = relationship(
        "AIAnalysisReport",
        back_populates="created_defect",
    )
    linked_articles: Mapped[List["KnowledgeBaseArticle"]] = relationship(
        secondary=defect_kb_article,
        back_populates="linked_defects",
    )
