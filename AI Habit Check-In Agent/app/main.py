from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.db.database import init_db
from app.utils.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and clean up on shutdown."""
    logger.info("Starting AI Habit Check-In Agent")
    await init_db()
    yield
    logger.info("Shutting down AI Habit Check-In Agent")


app = FastAPI(
    title="AI Habit Check-In Agent",
    description=(
        "A health behavior check-in API powered by AI coaching. "
        "Submit your health goals, daily actions, and mood to receive "
        "personalized coaching feedback with quality evaluation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
