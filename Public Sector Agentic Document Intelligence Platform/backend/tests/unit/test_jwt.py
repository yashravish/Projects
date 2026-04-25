"""Unit tests for JWT issuance, verification, and type-mismatch rejection."""
from __future__ import annotations

import uuid

import pytest
from app.security import jwt as jwt_module


def test_access_token_round_trip() -> None:
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    token = jwt_module.create_access_token(
        user_id=user_id, organization_id=org_id, role="admin"
    )
    payload = jwt_module.decode_token(token, expected_type="access")
    assert payload["sub"] == str(user_id)
    assert payload["org"] == str(org_id)
    assert payload["role"] == "admin"
    assert payload["type"] == "access"


def test_refresh_token_round_trip() -> None:
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    token = jwt_module.create_refresh_token(
        user_id=user_id, organization_id=org_id, role="analyst"
    )
    payload = jwt_module.decode_token(token, expected_type="refresh")
    assert payload["type"] == "refresh"


def test_decode_rejects_wrong_type() -> None:
    token = jwt_module.create_access_token(
        user_id=uuid.uuid4(), organization_id=uuid.uuid4(), role="admin"
    )
    with pytest.raises(jwt_module.TokenError):
        jwt_module.decode_token(token, expected_type="refresh")


def test_decode_rejects_garbage() -> None:
    with pytest.raises(jwt_module.TokenError):
        jwt_module.decode_token("not.a.jwt", expected_type="access")
