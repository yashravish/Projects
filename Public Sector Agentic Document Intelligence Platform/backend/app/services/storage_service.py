"""Storage abstraction over local filesystem and S3.

Two concrete backends behind a `Storage` type alias (`S3Storage | LocalFilesystemStorage`)
so callers don't care which is in use. Selection is driven by `Settings.storage_backend`.

PDF *content* validation (magic bytes, max size, max page count) is in this
module too so it lives next to the bytes themselves.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import os
import uuid
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings
from app.logging_config import get_logger

log = get_logger("storage")

PDF_MAGIC = b"%PDF-"
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 500


class StorageError(Exception):
    """Raised when an upload, download, or delete cannot complete."""


class InvalidPDFError(Exception):
    """Raised when an upload fails magic-byte / size / page validation."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class StoredObject:
    """The result of a successful put — what we record on `Document.s3_key`."""

    key: str
    bytes_written: int
    sha256: str


# ---------- magic-byte / size / page validation ---------------------------------


def validate_pdf_bytes(data: bytes, *, content_type: str | None) -> None:
    """Reject anything that isn't a real PDF. Raises `InvalidPDFError`."""
    if content_type and content_type.lower() not in {"application/pdf", "application/x-pdf"}:
        raise InvalidPDFError(
            f"unsupported content-type: {content_type}",
            code="INVALID_CONTENT_TYPE",
        )
    if len(data) == 0:
        raise InvalidPDFError("empty upload", code="EMPTY_UPLOAD")
    if len(data) > MAX_PDF_BYTES:
        raise InvalidPDFError(
            f"file exceeds maximum {MAX_PDF_BYTES} bytes ({len(data)} given)",
            code="FILE_TOO_LARGE",
        )
    if not data[:8].lstrip().startswith(PDF_MAGIC):
        raise InvalidPDFError(
            "file is not a PDF (missing %PDF- magic bytes)",
            code="NOT_A_PDF",
        )


def validate_page_count(page_count: int) -> None:
    if page_count <= 0:
        raise InvalidPDFError("PDF has no pages", code="ZERO_PAGES")
    if page_count > MAX_PDF_PAGES:
        raise InvalidPDFError(
            f"PDF exceeds maximum {MAX_PDF_PAGES} pages ({page_count} given)",
            code="TOO_MANY_PAGES",
        )


# ---------- virus-scan stub interface --------------------------------------------


@runtime_checkable
class VirusScanner(Protocol):
    """Pluggable virus scanner. Real implementation (e.g. ClamAV) plugs in here."""

    async def scan(self, data: bytes) -> bool:
        """Return True if the bytes are safe; False otherwise. Never raise on a
        clean file. Implementations should return False (not raise) on infection."""


class NullVirusScanner:
    """Default scanner: passes everything through, but logs that it ran.

    This is a deliberate, named no-op so callers can wire a real scanner in
    later without code changes elsewhere. The interface is real; the impl is
    stubbed.
    """

    backend_name = "null"

    async def scan(self, data: bytes) -> bool:
        log.debug("virus_scan.null", bytes=len(data))
        return True


def build_default_scanner() -> VirusScanner:
    return NullVirusScanner()


# ---------- key generation -------------------------------------------------------


def make_object_key(*, organization_id: uuid.UUID, sha256: str, filename: str) -> str:
    """Deterministic object key. SHA-based so duplicates collide."""
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    return f"orgs/{organization_id}/documents/{sha256}-{safe_name}"


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------- local filesystem backend ---------------------------------------------


class LocalFilesystemStorage:
    """Filesystem-backed storage. Default for local dev and tests."""

    backend_name = "local"

    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Block path traversal explicitly: refuse keys that escape the root.
        target = (self.root / key).resolve()
        if self.root not in target.parents and target != self.root:
            raise StorageError(f"key escapes storage root: {key}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    async def put(self, key: str, data: bytes) -> StoredObject:
        target = self._resolve(key)
        await asyncio.to_thread(target.write_bytes, data)
        sha = hash_bytes(data)
        log.info(
            "storage.put",
            backend=self.backend_name,
            key=key,
            bytes=len(data),
            sha256=sha[:12],
        )
        return StoredObject(key=key, bytes_written=len(data), sha256=sha)

    async def get(self, key: str) -> bytes:
        target = self._resolve(key)
        if not target.is_file():
            raise StorageError(f"object not found: {key}")
        return await asyncio.to_thread(target.read_bytes)

    async def stream(
        self, key: str, chunk_size: int = 64 * 1024
    ) -> AsyncGenerator[bytes, None]:
        target = self._resolve(key)
        if not target.is_file():
            raise StorageError(f"object not found: {key}")

        def _read_chunks() -> Iterable[bytes]:
            with target.open("rb") as fh:
                while True:
                    chunk = fh.read(chunk_size)
                    if not chunk:
                        return
                    yield chunk

        for chunk in await asyncio.to_thread(lambda: list(_read_chunks())):
            yield chunk

    async def delete(self, key: str) -> None:
        target = self._resolve(key)
        if target.is_file():
            await asyncio.to_thread(target.unlink)
            log.info("storage.delete", backend=self.backend_name, key=key)

    async def exists(self, key: str) -> bool:
        target = self._resolve(key)
        return await asyncio.to_thread(target.is_file)


# ---------- S3 backend -----------------------------------------------------------


class S3Storage:
    """Boto3 S3 backend. Synchronous boto3 calls dispatched via `to_thread`."""

    backend_name = "s3"

    def __init__(self, *, bucket: str, region: str, endpoint_url: str | None) -> None:
        if not bucket:
            raise StorageError("S3 backend requires a bucket name")
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url or None
        self._client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=self.endpoint_url,
        )

    @retry(
        retry=retry_if_exception_type((BotoCoreError, ClientError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _put_blocking(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    async def put(self, key: str, data: bytes) -> StoredObject:
        try:
            await asyncio.to_thread(self._put_blocking, key, data)
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"S3 put failed: {exc}") from exc
        sha = hash_bytes(data)
        log.info(
            "storage.put",
            backend=self.backend_name,
            key=key,
            bytes=len(data),
            sha256=sha[:12],
        )
        return StoredObject(key=key, bytes_written=len(data), sha256=sha)

    async def get(self, key: str) -> bytes:
        def _blocking() -> bytes:
            obj = self._client.get_object(Bucket=self.bucket, Key=key)
            body = obj["Body"]
            try:
                buf = io.BytesIO()
                for chunk in iter(lambda: body.read(64 * 1024), b""):
                    buf.write(chunk)
                return buf.getvalue()
            finally:
                body.close()

        try:
            return await asyncio.to_thread(_blocking)
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"S3 get failed: {exc}") from exc

    async def stream(
        self, key: str, chunk_size: int = 64 * 1024
    ) -> AsyncGenerator[bytes, None]:
        # S3 streaming via to_thread; chunks read on the worker thread, yielded here.
        def _open_body() -> object:
            try:
                obj = self._client.get_object(Bucket=self.bucket, Key=key)
            except (BotoCoreError, ClientError) as exc:
                raise StorageError(f"S3 get failed: {exc}") from exc
            return obj["Body"]

        body = await asyncio.to_thread(_open_body)
        try:
            while True:
                chunk = await asyncio.to_thread(body.read, chunk_size)  # type: ignore[attr-defined]
                if not chunk:
                    return
                yield chunk
        finally:
            await asyncio.to_thread(body.close)  # type: ignore[attr-defined]

    async def delete(self, key: str) -> None:
        def _blocking() -> None:
            self._client.delete_object(Bucket=self.bucket, Key=key)

        try:
            await asyncio.to_thread(_blocking)
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"S3 delete failed: {exc}") from exc
        log.info("storage.delete", backend=self.backend_name, key=key)

    async def exists(self, key: str) -> bool:
        def _blocking() -> bool:
            try:
                self._client.head_object(Bucket=self.bucket, Key=key)
                return True
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                    return False
                raise

        try:
            return await asyncio.to_thread(_blocking)
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"S3 head failed: {exc}") from exc


type Storage = S3Storage | LocalFilesystemStorage


# ---------- factory --------------------------------------------------------------


_storage_singleton: Storage | None = None


def build_storage(settings: Settings | None = None) -> Storage:
    settings = settings or get_settings()
    if settings.storage_backend == "s3":
        return S3Storage(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url or None,
        )
    os.makedirs(settings.local_upload_dir, exist_ok=True)
    return LocalFilesystemStorage(root=settings.local_upload_dir)


def get_storage() -> Storage:
    """Return the process-wide Storage singleton."""
    global _storage_singleton
    if _storage_singleton is None:
        _storage_singleton = build_storage()
    return _storage_singleton


def reset_storage_for_tests() -> None:
    """Reset the singleton; only used by tests that swap backends."""
    global _storage_singleton
    _storage_singleton = None
