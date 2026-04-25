"""Unit tests for configuration parsing."""
from __future__ import annotations

import importlib

from app import config as config_module


def test_allowed_origins_list_parses_csv(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://a.test, http://b.test ,http://c.test")
    importlib.reload(config_module)
    settings = config_module.Settings()
    assert settings.allowed_origins_list == [
        "http://a.test",
        "http://b.test",
        "http://c.test",
    ]


def test_jwt_algorithm_defaults_to_hs256(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("JWT_PRIVATE_KEY_PATH", "")
    importlib.reload(config_module)
    settings = config_module.Settings()
    assert settings.jwt_algorithm == "HS256"
    assert settings.jwt_uses_rs256 is False
