"""Shared fixtures for all test suites."""
import os
import sys
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://ciqalab:ciqalab_pass@localhost:5432/ciqalab"
)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEVICE_SIMULATOR_URL = os.getenv("DEVICE_SIMULATOR_URL", "http://localhost:8001")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8080")


@pytest.fixture(scope="session")
def db_engine():
    """SQLAlchemy engine for direct database assertions."""
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Scoped database session for test assertions."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="session")
def backend_url():
    return BACKEND_URL


@pytest.fixture(scope="session")
def device_url():
    return DEVICE_SIMULATOR_URL


@pytest.fixture(scope="session")
def frontend_url():
    return FRONTEND_URL


def run_sql(db_session, query: str, params: dict = None):
    """Execute a raw SQL query and return the result rows."""
    result = db_session.execute(text(query), params or {})
    return result.fetchall()


def count_rows(db_session, table: str, where: str = "1=1", params: dict = None):
    """Count rows in a table with an optional WHERE clause."""
    result = db_session.execute(
        text(f"SELECT COUNT(*) FROM {table} WHERE {where}"), params or {}
    )
    return result.scalar()
