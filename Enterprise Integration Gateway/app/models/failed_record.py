from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class FailedRecord(Base):
    """
    Dead-letter queue for records that could not be transformed or persisted.

    status lifecycle: pending_retry → retrying → resolved | abandoned
    """

    __tablename__ = "failed_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    sync_job_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("sync_jobs.id"), nullable=True
    )
    source: Mapped[str] = mapped_column(String(50), nullable=False)       # 'crm', 'vendor'
    record_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'customer', 'order', 'shipment'
    external_id: Mapped[str | None] = mapped_column(String(100), index=True)
    raw_data: Mapped[str | None] = mapped_column(Text)  # original payload as JSON/XML string
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # pending_retry | retrying | resolved | abandoned
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending_retry")
    last_retried_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    sync_job: Mapped["SyncJob"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "SyncJob", back_populates="failed_records", lazy="select"
    )

    __table_args__ = (
        Index("ix_failed_records_source_status", "source", "status"),
        Index("ix_failed_records_sync_job_id", "sync_job_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<FailedRecord id={self.id} source={self.source!r} "
            f"type={self.record_type!r} status={self.status!r}>"
        )
