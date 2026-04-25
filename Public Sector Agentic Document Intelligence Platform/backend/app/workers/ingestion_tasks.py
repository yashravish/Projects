"""Celery tasks that drive ingestion.

Tasks are thin: they create a fresh async DB session, build the storage and
embedder instances appropriate for this process, and delegate to
`app.services.ingestion_service.ingest_document`.

Async work runs on a per-task event loop spun up here — Celery is sync and
the rest of the codebase is async, so we bridge with `asyncio.run`.
"""
from __future__ import annotations

import asyncio
import uuid

from app.agents.llm_client import build_embedder
from app.db.session import async_session_factory
from app.logging_config import get_logger
from app.services.ingestion_service import ingest_document
from app.services.storage_service import build_storage
from app.workers.celery_app import celery_app

log = get_logger("worker.ingestion")


async def _run(organization_id: uuid.UUID, document_id: uuid.UUID) -> int:
    storage = build_storage()
    embedder = build_embedder()
    async with async_session_factory() as session:
        return await ingest_document(
            session,
            organization_id=organization_id,
            document_id=document_id,
            storage=storage,
            embedder=embedder,
        )


@celery_app.task(
    name="app.workers.ingestion_tasks.process_document",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    max_retries=2,
    retry_jitter=True,
)
def process_document(self: object, organization_id: str, document_id: str) -> int:
    """Entry point invoked by FastAPI on upload."""
    log.info(
        "worker.ingestion.start",
        organization_id=organization_id,
        document_id=document_id,
    )
    return asyncio.run(_run(uuid.UUID(organization_id), uuid.UUID(document_id)))
