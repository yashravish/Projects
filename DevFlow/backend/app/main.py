from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.database import Base, engine
from app.routers import ab, ai, defects, deployments, flags, health, knowledge, metrics, pipelines, projects
from app.services.metrics_state import global_metrics

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if "sqlite" in settings.database_url:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(  # type: ignore[override]
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        global_metrics.record_api_request()
        return await call_next(request)


app.add_middleware(MetricsMiddleware)

app.include_router(health.router)
app.include_router(projects.router)
app.include_router(pipelines.router)
app.include_router(deployments.router)
app.include_router(flags.router)
app.include_router(ab.router)
app.include_router(metrics.router)
app.include_router(ai.router)
app.include_router(defects.router)
app.include_router(knowledge.router)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics_root() -> str:
    return global_metrics.to_prometheus_text().strip() + "\n"
