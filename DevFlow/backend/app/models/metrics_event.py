import datetime as dt
from typing import Any

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class MetricsEvent(Base):
    __tablename__ = "metrics_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    value: Mapped[float] = mapped_column()
    namespace: Mapped[str] = mapped_column(String(64), default="devflow", index=True)
    labels: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ts: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None), index=True
    )
