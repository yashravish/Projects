from __future__ import annotations

import datetime as dt
import enum
from typing import List

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, Enum as PgEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class PipelineStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"


class StageName(str, enum.Enum):
    lint = "lint"
    unit_tests = "unit_tests"
    integration_tests = "integration_tests"
    build = "build"
    deploy = "deploy"


class StageStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"
    skipped = "skipped"


class TestStatus(str, enum.Enum):
    passed = "passed"
    failed = "failed"
    skipped = "skipped"


def _pg_enum(
    e: type[enum.Enum], name: str, native: str = "VARCHAR(64)"
) -> PgEnum:
    return PgEnum(e, name=name, values_callable=lambda x: [m.value for m in x], native_enum=False)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    status: Mapped[PipelineStatus] = mapped_column(
        _pg_enum(PipelineStatus, "pipelinestatus"),
        default=PipelineStatus.pending,
    )
    branch: Mapped[str] = mapped_column(String(256), default="main")
    commit_sha: Mapped[str] = mapped_column(String(64), default="0000000")
    started_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)
    total_duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    external_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)

    project = relationship("Project", back_populates="runs")
    stages: Mapped[List["PipelineStage"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="PipelineStage.sort_order",
    )
    test_results: Mapped[List["TestResult"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class PipelineStage(Base):
    __tablename__ = "pipeline_stages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"), index=True)
    name: Mapped[StageName] = mapped_column(_pg_enum(StageName, "stagename"))
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[StageStatus] = mapped_column(
        _pg_enum(StageStatus, "stagestatus"),
        default=StageStatus.pending,
    )
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    logs: Mapped[str] = mapped_column(Text, default="")
    passed: Mapped[bool] = mapped_column(Boolean, default=True)

    run = relationship("PipelineRun", back_populates="stages")


class TestResult(Base):
    __tablename__ = "test_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(512))
    suite: Mapped[str] = mapped_column(String(256), default="default")
    status: Mapped[TestStatus] = mapped_column(
        _pg_enum(TestStatus, "teststatus"),
        default=TestStatus.passed,
    )
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )

    run = relationship("PipelineRun", back_populates="test_results")
