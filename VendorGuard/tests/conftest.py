"""
Shared test fixtures using an in-memory SQLite database for speed.
"""

import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test_vendorguard.db"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["AI_ENABLED"] = "false"

from backend.database import Base, get_db
from backend.main import app
from backend.auth import get_password_hash
from backend.models import User, ControlDomain
from backend.engine.domain_mapping import CONTROL_DOMAINS

TEST_DB_URL = "sqlite:///./test_vendorguard.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    db = TestSession()
    if not db.query(User).first():
        db.add(User(
            username="testadmin",
            email="testadmin@test.local",
            hashed_password=get_password_hash("testpass"),
            full_name="Test Admin",
            role="admin",
        ))
        db.add(User(
            username="testanalyst",
            email="testanalyst@test.local",
            hashed_password=get_password_hash("testpass"),
            full_name="Test Analyst",
            role="analyst",
        ))
        for d in CONTROL_DOMAINS:
            db.add(ControlDomain(**d))
        db.commit()
    db.close()
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test_vendorguard.db"):
        os.remove("./test_vendorguard.db")


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def db_session():
    db = TestSession()
    yield db
    db.close()


@pytest.fixture()
def auth_headers(client):
    response = client.post("/api/auth/login", json={"username": "testadmin", "password": "testpass"})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def analyst_headers(client):
    response = client.post("/api/auth/login", json={"username": "testanalyst", "password": "testpass"})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
