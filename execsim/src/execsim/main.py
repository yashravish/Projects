"""FastAPI application factory."""

from fastapi import FastAPI

from execsim import __version__
from execsim.api.health import router as health_router
from execsim.api.runs import router as runs_router
from execsim.api.opportunities import router as opportunities_router
from execsim.api.auctions import router as auctions_router
from execsim.api.validation import router as validation_router
from execsim.logging import setup_logging


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    setup_logging()

    application = FastAPI(
        title="execsim",
        version=__version__,
        description="Execution Simulator & Auction Optimizer",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.include_router(health_router, prefix="/api/v1")
    application.include_router(runs_router, prefix="/api/v1")
    application.include_router(opportunities_router, prefix="/api/v1")
    application.include_router(auctions_router, prefix="/api/v1")
    application.include_router(validation_router, prefix="/api/v1")

    return application


app = create_app()
