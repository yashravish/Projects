from __future__ import annotations

import datetime as dt
import enum
from typing import Optional

from sqlalchemy import Enum as PgEnum, ForeignKey, Integer, String, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class DeploymentStatus(str, enum.Enum):
    pending = "pending"
    healthy = "healthy"
    degraded = "degraded"
    failed = "failed"
    rolled_back = "rolled_back"


def _e(e: type[enum.Enum], name: str) -> PgEnum:
    return PgEnum(e, name=name, values_callable=lambda x: [m.value for m in x], native_enum=False)


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    version: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[DeploymentStatus] = mapped_column(
        _e(DeploymentStatus, "depstatus"), default=DeploymentStatus.pending
    )
    environment: Mapped[str] = mapped_column(String(64), default="production", index=True)
    canary_percent: Mapped[int] = mapped_column(Integer, default=0)
    error_rate: Mapped[float] = mapped_column(Float, default=0.0)
    rolled_back_from_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("deployments.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
    )

    project = relationship("Project", back_populates="deployments")
    rolled_back_from: Mapped[Optional["Deployment"]] = relationship(
        remote_side="Deployment.id", backref="rollback_children"
    )
