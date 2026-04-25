"""FastAPI application factory and ASGI entrypoint."""
from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from structlog.contextvars import bind_contextvars, clear_contextvars

from app import __version__
from app.api.router import api_v1_router, health_routes
from app.config import Settings, get_settings
from app.logging_config import configure_logging, get_logger


def _validate_production_config(settings: Settings) -> None:
    if settings.env == "production":
        if "*" in settings.allowed_origins_list:
            raise RuntimeError("ALLOWED_ORIGINS=* is forbidden in production")
        if settings.jwt_secret == "change-me" and not settings.jwt_uses_rs256:
            raise RuntimeError("JWT_SECRET must be set (or RS256 keys) in production")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    log = get_logger("startup")
    log.info("psdi.startup", version=__version__, env=get_settings().env)
    yield
    log.info("psdi.shutdown")


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    _validate_production_config(settings)

    app = FastAPI(
        title="PublicSector Agentic Document Intelligence Platform",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        clear_contextvars()
        bind_contextvars(request_id=request_id, route=request.url.path, method=request.method)
        log = get_logger("http")
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.exception("http.error", latency_ms=elapsed_ms)
            return JSONResponse(
                status_code=500,
                content={
                    "error_code": "INTERNAL_ERROR",
                    "detail": "internal server error",
                    "request_id": request_id,
                },
                headers={"x-request-id": request_id},
            )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        response.headers["x-request-id"] = request_id
        log.info(
            "http.request",
            status=response.status_code,
            latency_ms=elapsed_ms,
        )
        return response

    app.include_router(health_routes.router)
    app.include_router(api_v1_router)

    return app


app = create_app()
