"""Celery application factory.

Broker / result backend: Redis (the same instance used elsewhere).
Concurrency: pre-fork worker. One pool per task class is enough for the size
of corpus this platform serves.

The `task_acks_late=True` + `task_reject_on_worker_lost=True` pair ensures we
don't lose work if a worker crashes mid-task. `worker_prefetch_multiplier=1`
keeps long-running ingest tasks fairly distributed.
"""
from __future__ import annotations

from celery import Celery

from app.config import get_settings
from app.logging_config import configure_logging

configure_logging()
settings = get_settings()

celery_app = Celery(
    "publicsector_adip",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.ingestion_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=600,
    task_soft_time_limit=540,
    broker_connection_retry_on_startup=True,
)
