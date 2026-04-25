"""Health endpoint that probes each external dependency."""
from __future__ import annotations

import asyncio
from typing import Any, Literal

import httpx
import redis.asyncio as redis
from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.deps import SessionDep
from app.logging_config import get_logger

router = APIRouter(tags=["health"])
log = get_logger("health")

ComponentStatus = Literal["ok", "degraded", "down", "not_configured"]


async def _check_db(session: AsyncSession) -> ComponentStatus:
    try:
        result = await session.execute(text("SELECT 1"))
        return "ok" if result.scalar() == 1 else "degraded"
    except Exception as exc:  # pragma: no cover - failure path
        log.warning("health.db.failed", error=str(exc))
        return "down"


async def _check_redis() -> ComponentStatus:
    settings = get_settings()
    client: Any | None = None
    try:
        client = redis.from_url(settings.redis_url, socket_timeout=2.0)
        pong = await client.ping()
        return "ok" if pong else "degraded"
    except Exception as exc:  # pragma: no cover - failure path
        log.warning("health.redis.failed", error=str(exc))
        return "down"
    finally:
        if client is not None:
            await client.aclose()


async def _check_mlflow() -> ComponentStatus:
    settings = get_settings()
    url = f"{settings.mlflow_tracking_uri.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=2.0) as http:
            resp = await http.get(url)
        return "ok" if resp.status_code == 200 else "degraded"
    except Exception as exc:  # pragma: no cover - failure path
        log.warning("health.mlflow.failed", error=str(exc))
        return "down"


def _check_openai() -> ComponentStatus:
    settings = get_settings()
    if not settings.openai_api_key:
        return "not_configured"
    return "ok"


@router.get("/health")
async def health(session: SessionDep) -> dict[str, Any]:
    db_status, redis_status, mlflow_status = await asyncio.gather(
        _check_db(session),
        _check_redis(),
        _check_mlflow(),
    )
    openai_status = _check_openai()

    components = {
        "db": db_status,
        "redis": redis_status,
        "mlflow": mlflow_status,
        "openai": openai_status,
    }
    overall = (
        "ok"
        if all(s in ("ok", "not_configured") for s in components.values())
        else "degraded"
    )
    return {"status": overall, **components}
