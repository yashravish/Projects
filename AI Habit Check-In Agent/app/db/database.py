import aiosqlite
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from app.config import get_settings
from app.utils.logging import logger

# Extract path from SQLAlchemy-style URL for aiosqlite
_DB_PATH: str | None = None


def _get_db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        url = get_settings().database_url
        # "sqlite+aiosqlite:///./checkins.db" -> "./checkins.db"
        _DB_PATH = url.split("///")[-1]
    return _DB_PATH


def set_db_path(path: str) -> None:
    """Override the database path (used for testing)."""
    global _DB_PATH
    _DB_PATH = path


async def init_db() -> None:
    """Create the check-ins table if it doesn't exist."""
    db_path = _get_db_path()
    logger.info(f"Initializing database at {db_path}")

    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                health_goal TEXT NOT NULL,
                todays_actions TEXT NOT NULL,
                current_mood TEXT NOT NULL,
                summary TEXT NOT NULL,
                habit_risk TEXT NOT NULL,
                next_action TEXT NOT NULL,
                motivational_message TEXT NOT NULL,
                actionability_score INTEGER NOT NULL,
                empathy_score INTEGER NOT NULL,
                specificity_score INTEGER NOT NULL,
                safety_score INTEGER NOT NULL,
                evaluation_notes TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        await db.commit()
    logger.info("Database initialized successfully")


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Yield an aiosqlite connection."""
    db_path = _get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        yield db
