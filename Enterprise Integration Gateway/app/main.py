"""
Enterprise Integration Gateway — main application entry point.

Wires together:
  - FastAPI app with versioned router
  - Database initialization on startup
  - Redis connection pool for caching and rate limiting
  - Kafka producer/consumer for event-driven integration
  - APScheduler background sync jobs
  - Request-ID middleware
  - CORS middleware
  - Structured exception handlers
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import IntegrationError
from app.core.logging_config import setup_logging
from app.core.redis_client import init_redis, shutdown_redis
from app.core.kafka_client import init_kafka_producer, shutdown_kafka
from app.db.init_db import init_db
from app.jobs.scheduler import shutdown_scheduler, start_scheduler
from app.jobs.event_consumer import start_event_consumer, stop_event_consumer

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    logger.info(
        "application_starting",
        extra={"app_name": settings.APP_NAME, "version": settings.APP_VERSION, "env": settings.APP_ENV},
    )
    init_db()
    init_redis()
    init_kafka_producer()
    if settings.SCHEDULER_ENABLED:
        start_scheduler()
    if settings.KAFKA_ENABLED:
        start_event_consumer()
    logger.info("application_ready")

    yield

    logger.info("application_shutting_down")
    if settings.SCHEDULER_ENABLED:
        shutdown_scheduler()
    stop_event_consumer()
    shutdown_kafka()
    shutdown_redis()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Enterprise Integration Gateway — REST API for cross-system data synchronization.\n\n"
        "Integrates a mock CRM (JSON) and a mock Vendor (XML) feed into a unified "
        "normalized PostgreSQL datastore with full job tracking, retry support, "
        "Redis caching, Kafka event streaming, and AWS deployment support."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique X-Request-ID header to every response."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "http_transaction",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ── Exception handlers ─────────────────────────────────────────────────────────


@app.exception_handler(IntegrationError)
async def integration_error_handler(request: Request, exc: IntegrationError):
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "error": "integration_error",
            "source": exc.source,
            "message": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Check server logs.",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

app.include_router(api_router, prefix=settings.API_V1_PREFIX)
