"""Pytest fixtures - SQLite in-memory database test setup."""
from __future__ import annotations

import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")

from app.core.security import hash_password  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models.user import User, UserRole  # noqa: E402


TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def engine():
    eng = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )

    @event.listens_for(eng, "connect")
    def _enable_sqlite_fk(dbapi_connection, connection_record):  # noqa: ARG001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=eng)
    try:
        yield eng
    finally:
        Base.metadata.drop_all(bind=eng)
        eng.dispose()


@pytest.fixture(scope="function")
def db_session(engine) -> Generator[Session, None, None]:
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(engine, db_session: Session) -> Generator[TestClient, None, None]:
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def seeded_users(db_session: Session) -> dict[str, User]:
    manager = User(
        name="Test Manager",
        email="manager@test.com",
        hashed_password=hash_password("password123"),
        role=UserRole.MANAGER,
    )
    analyst = User(
        name="Test Analyst",
        email="analyst@test.com",
        hashed_password=hash_password("password123"),
        role=UserRole.ANALYST,
    )
    other_analyst = User(
        name="Other Analyst",
        email="other@test.com",
        hashed_password=hash_password("password123"),
        role=UserRole.ANALYST,
    )
    db_session.add_all([manager, analyst, other_analyst])
    db_session.commit()
    db_session.refresh(manager)
    db_session.refresh(analyst)
    db_session.refresh(other_analyst)
    return {"manager": manager, "analyst": analyst, "other_analyst": other_analyst}


def login(client: TestClient, email: str, password: str = "password123") -> str:
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


@pytest.fixture(scope="function")
def manager_token(client: TestClient, seeded_users) -> str:
    return login(client, seeded_users["manager"].email)


@pytest.fixture(scope="function")
def analyst_token(client: TestClient, seeded_users) -> str:
    return login(client, seeded_users["analyst"].email)


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
