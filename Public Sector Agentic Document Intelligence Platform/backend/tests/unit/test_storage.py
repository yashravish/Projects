"""Local filesystem backend + magic-byte validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.storage_service import (
    InvalidPDFError,
    LocalFilesystemStorage,
    StorageError,
    hash_bytes,
    make_object_key,
    validate_pdf_bytes,
)


def test_validate_pdf_accepts_real_magic() -> None:
    validate_pdf_bytes(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", content_type="application/pdf")


def test_validate_pdf_rejects_non_pdf() -> None:
    with pytest.raises(InvalidPDFError) as exc:
        validate_pdf_bytes(b"GIF89a", content_type="application/pdf")
    assert exc.value.code == "NOT_A_PDF"


def test_validate_pdf_rejects_wrong_content_type() -> None:
    with pytest.raises(InvalidPDFError) as exc:
        validate_pdf_bytes(b"%PDF-1.4\n", content_type="image/png")
    assert exc.value.code == "INVALID_CONTENT_TYPE"


def test_validate_pdf_rejects_empty() -> None:
    with pytest.raises(InvalidPDFError) as exc:
        validate_pdf_bytes(b"", content_type="application/pdf")
    assert exc.value.code == "EMPTY_UPLOAD"


def test_validate_pdf_rejects_too_large() -> None:
    big = b"%PDF-1.4\n" + b"x" * (50 * 1024 * 1024 + 10)
    with pytest.raises(InvalidPDFError) as exc:
        validate_pdf_bytes(big, content_type="application/pdf")
    assert exc.value.code == "FILE_TOO_LARGE"


def test_hash_bytes_is_sha256() -> None:
    assert hash_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_make_object_key_is_deterministic() -> None:
    import uuid

    org = uuid.UUID("00000000-0000-0000-0000-000000000001")
    a = make_object_key(organization_id=org, sha256="abc123", filename="my file.pdf")
    b = make_object_key(organization_id=org, sha256="abc123", filename="my file.pdf")
    assert a == b
    assert "/" in a
    assert "my_file.pdf" in a
    assert "abc123" in a


@pytest.mark.asyncio
async def test_local_storage_put_get_delete_exists(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(root=str(tmp_path))
    obj = await storage.put("a/b.bin", b"hello world")
    assert obj.bytes_written == 11
    assert obj.sha256 == hash_bytes(b"hello world")
    assert await storage.exists("a/b.bin")
    assert await storage.get("a/b.bin") == b"hello world"

    chunks: list[bytes] = []
    async for chunk in storage.stream("a/b.bin", chunk_size=4):
        chunks.append(chunk)
    assert b"".join(chunks) == b"hello world"

    await storage.delete("a/b.bin")
    assert not await storage.exists("a/b.bin")


@pytest.mark.asyncio
async def test_local_storage_blocks_path_traversal(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(root=str(tmp_path))
    with pytest.raises(StorageError):
        await storage.put("../escape.bin", b"x")


@pytest.mark.asyncio
async def test_local_storage_get_missing_raises(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(root=str(tmp_path))
    with pytest.raises(StorageError):
        await storage.get("nope.bin")
