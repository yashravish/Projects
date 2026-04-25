"""Unit tests for password hashing."""
from __future__ import annotations

from app.security.passwords import hash_password, verify_password


def test_hash_is_deterministically_verifiable() -> None:
    hashed = hash_password("CorrectHorseBatteryStaple!7")
    assert hashed != "CorrectHorseBatteryStaple!7"
    assert verify_password("CorrectHorseBatteryStaple!7", hashed)


def test_wrong_password_fails() -> None:
    hashed = hash_password("CorrectHorseBatteryStaple!7")
    assert not verify_password("WrongPassword!7", hashed)


def test_garbage_hash_does_not_raise() -> None:
    assert verify_password("anything", "not-a-valid-bcrypt-hash") is False
