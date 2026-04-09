from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class SyncJob(Base):
    """
    Tracks a single integration sync run.

    lifecycle: pending → running → success | partial_success | failed
    """

    __tablename__ = "sync_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    correlation_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # pending | running | success | partial_success | failed
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    triggered_by: Mapped[str] = mapped_column(String(50), nullable=False, default="scheduler")
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    records_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_inserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_updated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    job_metadata: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)

    failed_records: Mapped[list["FailedRecord"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "FailedRecord", back_populates="sync_job", lazy="select"
    )

    __table_args__ = (
        Index("ix_sync_jobs_job_type_status", "job_type", "status"),
        Index("ix_sync_jobs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<SyncJob id={self.id} type={self.job_type!r} "
            f"status={self.status!r} corr={self.correlation_id!r}>"
        )
