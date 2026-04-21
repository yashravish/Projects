"""Test fixtures for execsim."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from execsim.config import Settings
from execsim.db.models import Base
from execsim.dependencies import get_db, get_settings
from execsim.main import create_app


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Application settings from environment."""
    return Settings()


@pytest.fixture(scope="session")
def engine(settings):
    """SQLAlchemy engine for the test database."""
    return create_engine(settings.database_url)


@pytest.fixture(scope="session")
def tables(engine):
    """Create all tables before tests, drop after."""
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture()
def db_session(engine, tables):
    """Yield a transactional session that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def client(db_session, settings):
    """FastAPI test client with overridden dependencies."""
    application = create_app()

    def _override_db():
        yield db_session

    application.dependency_overrides[get_db] = _override_db
    application.dependency_overrides[get_settings] = lambda: settings

    with TestClient(application) as c:
        yield c
