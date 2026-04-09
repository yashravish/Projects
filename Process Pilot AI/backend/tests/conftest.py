import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.jwt_handler import create_access_token, get_password_hash
from app.database import Base, get_db
from app.main import app
from app.models.user import User

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def create_tables():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def employee_user(db_session):
    user = db_session.query(User).filter(User.email == "testemployee@acme.com").first()
    if not user:
        user = User(
            email="testemployee@acme.com",
            full_name="Test Employee",
            department="Finance",
            role="employee",
            hashed_password=get_password_hash("testpass"),
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    return user


@pytest.fixture()
def manager_user(db_session):
    user = db_session.query(User).filter(User.email == "testmanager@acme.com").first()
    if not user:
        user = User(
            email="testmanager@acme.com",
            full_name="Test Manager",
            department="Operations",
            role="manager",
            hashed_password=get_password_hash("testpass"),
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    return user


@pytest.fixture()
def employee_headers(employee_user):
    token = create_access_token({"sub": str(employee_user.id)})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def manager_headers(manager_user):
    token = create_access_token({"sub": str(manager_user.id)})
    return {"Authorization": f"Bearer {token}"}
