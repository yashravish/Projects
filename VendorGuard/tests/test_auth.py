"""Tests for authentication endpoints."""


def test_login_success(client):
    res = client.post("/api/auth/login", json={"username": "testadmin", "password": "testpass"})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_invalid_password(client):
    res = client.post("/api/auth/login", json={"username": "testadmin", "password": "wrong"})
    assert res.status_code == 401


def test_login_nonexistent_user(client):
    res = client.post("/api/auth/login", json={"username": "nobody", "password": "pass"})
    assert res.status_code == 401


def test_get_me(client, auth_headers):
    res = client.get("/api/auth/me", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["username"] == "testadmin"
    assert data["role"] == "admin"


def test_get_me_unauthorized(client):
    res = client.get("/api/auth/me")
    assert res.status_code == 401


def test_logout(client, auth_headers):
    res = client.post("/api/auth/logout")
    assert res.status_code == 200
