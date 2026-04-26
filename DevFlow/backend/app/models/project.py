from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, List

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.ab_experiment import ABExperiment
    from app.models.defect import Defect
    from app.models.deployment import Deployment
    from app.models.pipeline import PipelineRun


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    slug: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )

    runs: Mapped[List["PipelineRun"]] = relationship(back_populates="project")
    deployments: Mapped[List["Deployment"]] = relationship(back_populates="project")
    experiments: Mapped[List["ABExperiment"]] = relationship(back_populates="project")
    defects: Mapped[List["Defect"]] = relationship(back_populates="project")
