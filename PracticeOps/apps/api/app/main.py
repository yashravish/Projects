"""FastAPI application entry point."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.errors import AppException
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestIDMiddleware, setup_rate_limiting
from app.routes import health
from app.routes.admin import router as admin_router
from app.routes.assignments import assignment_router
from app.routes.assignments import cycle_router as assignments_cycle_router
from app.routes.auth import me_router
from app.routes.auth import router as auth_router
from app.routes.cycles import router as cycles_router
from app.routes.dashboards import router as dashboards_router
from app.routes.invites import router as invites_router
from app.routes.notification_preferences import router as notification_preferences_router
from app.routes.practice_logs import cycle_router as practice_logs_cycle_router
from app.routes.practice_logs import practice_log_router
from app.routes.teams import router as teams_router
from app.routes.tickets import cycle_router as tickets_cycle_router
from app.routes.tickets import ticket_router
from app.services.scheduler import scheduler_lifespan

# Configure structured logging
# Use JSON format in production, human-readable in development
configure_logging(json_logs=settings.environment != "development")

logger = get_logger(__name__)

app = FastAPI(
    title="PracticeOps API",
    description="Practice management API for musical teams",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=scheduler_lifespan,
)

# Setup rate limiting
setup_rate_limiting(app)

# Request ID middleware (must be added first to capture all requests)
app.add_middleware(RequestIDMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle application exceptions with standard error format.

    Logs errors with request ID for correlation. No stack traces exposed to clients.
    """
    # Get request ID if available
    request_id = getattr(request.state, "request_id", None)

    # Log the error with context
    logger.warning(
        "app_exception",
        error_code=exc.code.value,
        message=exc.message,
        field=exc.field,
        status_code=exc.status_code,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code.value,
                "message": exc.message,
                "field": exc.field,
            }
        },
    )


# Routes
app.include_router(health.router)
app.include_router(auth_router)
app.include_router(me_router)
app.include_router(teams_router)
app.include_router(cycles_router)
app.include_router(assignments_cycle_router)
app.include_router(assignment_router)
app.include_router(practice_logs_cycle_router)
app.include_router(practice_log_router)
app.include_router(tickets_cycle_router)
app.include_router(ticket_router)
app.include_router(dashboards_router)
app.include_router(invites_router)
app.include_router(notification_preferences_router)
app.include_router(admin_router)
