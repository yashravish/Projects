"""Password hashing.

Bcrypt via passlib with sane defaults (cost factor 12). Hash format embeds
the algorithm + cost so verification is parameter-free.
"""
from __future__ import annotations

from typing import cast

from passlib.context import CryptContext

_pwd_context: CryptContext = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)


def hash_password(password: str) -> str:
    return cast(str, _pwd_context.hash(password))


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return cast(bool, _pwd_context.verify(password, password_hash))
    except ValueError:
        return False
