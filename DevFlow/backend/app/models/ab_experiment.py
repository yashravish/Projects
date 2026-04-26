from __future__ import annotations

import datetime as dt
import enum
from typing import List

from sqlalchemy import Enum as PgEnum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ExperimentStatus(str, enum.Enum):
    draft = "draft"
    running = "running"
    paused = "paused"
    completed = "completed"


def _e(e: type[enum.Enum], name: str) -> PgEnum:
    return PgEnum(e, name=name, values_callable=lambda x: [m.value for m in x], native_enum=False)


class ABExperiment(Base):
    __tablename__ = "ab_experiments"
    __table_args__ = (UniqueConstraint("project_id", "key", name="uq_ab_experiment_key_per_project"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(120), index=True)
    name: Mapped[str] = mapped_column(String(200))
    variant_a_name: Mapped[str] = mapped_column(String(64), default="A")
    variant_b_name: Mapped[str] = mapped_column(String(64), default="B")
    traffic_a_percent: Mapped[int] = mapped_column(Integer, default=50)
    status: Mapped[ExperimentStatus] = mapped_column(
        _e(ExperimentStatus, "expstatus"), default=ExperimentStatus.draft
    )
    key_metric: Mapped[str] = mapped_column(String(120), default="conversion")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
    )

    project = relationship("Project", back_populates="experiments")
    rollups: Mapped[List["ABMetricRollup"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class ABMetricRollup(Base):
    __tablename__ = "ab_metric_rollups"
    __table_args__ = (UniqueConstraint("experiment_id", "variant", name="uq_ab_rollup_variant"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("ab_experiments.id", ondelete="CASCADE"), index=True
    )
    variant: Mapped[str] = mapped_column(String(8), index=True)  # "A" or "B"
    assignments: Mapped[int] = mapped_column(Integer, default=0)
    conversion_count: Mapped[int] = mapped_column(Integer, default=0)
    sum_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )

    experiment = relationship("ABExperiment", back_populates="rollups")
