from fastapi import APIRouter, Depends, Response
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.metrics_event import MetricsEvent
from app.schemas.metrics import DashboardMetrics
from app.services.metrics_state import global_metrics

router = APIRouter(prefix="/api", tags=["observability"])


@router.get("/metrics/prom", response_class=Response)
async def prom_metrics() -> Response:
    data = global_metrics.to_prometheus_text()
    return Response(content=data, media_type="text/plain; version=0.0.4")


@router.get("/dashboard/metrics", response_model=DashboardMetrics)
async def dashboard_json(session: AsyncSession = Depends(get_db)) -> DashboardMetrics:
    ev = await session.execute(
        select(MetricsEvent).order_by(desc(MetricsEvent.id)).limit(50)
    )
    sample = [
        {"name": m.name, "value": m.value, "ts": m.ts.isoformat(), "labels": m.labels}
        for m in ev.scalars().all()
    ]
    return DashboardMetrics(
        from_metrics_state=global_metrics.to_dashboard(),
        from_metrics_events_sample=sample,
    )
