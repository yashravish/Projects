"""
Unit tests for the Kafka event publisher service.

Mocks the Kafka producer to verify event serialization and publish calls.
"""
import os
from datetime import datetime, timezone

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SCHEDULER_ENABLED", "false")
os.environ.setdefault("REDIS_ENABLED", "false")
os.environ.setdefault("KAFKA_ENABLED", "true")

import pytest
from unittest.mock import MagicMock, patch

from app.schemas.events import (
    SyncStartedEvent,
    SyncCompletedEvent,
    RecordFailedEvent,
    InboundSyncRequestEvent,
)
from app.services.event_publisher import publish_event
from app.core import kafka_client as kafka_module


# ── Event schema tests ───────────────────────────────────────────────────────


class TestEventSchemas:
    def test_sync_started_event(self):
        event = SyncStartedEvent(
            job_id=1,
            job_type="crm_sync",
            triggered_by="api",
            correlation_id="abc-123",
        )
        d = event.to_dict()
        assert d["event_type"] == "sync.started"
        assert d["job_id"] == 1
        assert d["job_type"] == "crm_sync"
        assert d["correlation_id"] == "abc-123"
        assert "timestamp" in d

    def test_sync_completed_event(self):
        event = SyncCompletedEvent(
            job_id=2,
            job_type="vendor_sync",
            status="success",
            records_processed=10,
            records_inserted=8,
            records_updated=2,
            records_failed=0,
            correlation_id="def-456",
        )
        d = event.to_dict()
        assert d["event_type"] == "sync.completed"
        assert d["records_processed"] == 10
        assert d["status"] == "success"

    def test_record_failed_event(self):
        event = RecordFailedEvent(
            job_id=3,
            source="vendor",
            record_type="order",
            external_id="VND-001",
            error_message="Missing required field",
            correlation_id="ghi-789",
        )
        d = event.to_dict()
        assert d["event_type"] == "record.failed"
        assert d["source"] == "vendor"
        assert d["external_id"] == "VND-001"

    def test_inbound_sync_request_event(self):
        event = InboundSyncRequestEvent(
            sync_type="crm",
            requested_by="external-system",
        )
        d = event.to_dict()
        assert d["event_type"] == "sync.request"
        assert d["sync_type"] == "crm"


# ── Event publisher tests ────────────────────────────────────────────────────


class TestPublishEvent:
    def test_publishes_event_when_producer_available(self, monkeypatch):
        mock_producer = MagicMock()
        mock_producer.is_available = True
        mock_producer.publish.return_value = True
        monkeypatch.setattr(kafka_module, "_producer", mock_producer)

        event = SyncStartedEvent(
            job_id=1,
            job_type="crm_sync",
            triggered_by="api",
            correlation_id="test-123",
        )
        result = publish_event(event, key="test-key")

        assert result is True
        mock_producer.publish.assert_called_once()
        call_args = mock_producer.publish.call_args
        assert call_args[0][0] == "eig.integration.events"  # topic
        assert call_args[0][1]["event_type"] == "sync.started"

    def test_skips_when_producer_unavailable(self, monkeypatch):
        monkeypatch.setattr(kafka_module, "_producer", None)

        event = SyncStartedEvent(
            job_id=1,
            job_type="crm_sync",
            triggered_by="api",
        )
        result = publish_event(event)

        assert result is False

    def test_uses_correlation_id_as_default_key(self, monkeypatch):
        mock_producer = MagicMock()
        mock_producer.is_available = True
        mock_producer.publish.return_value = True
        monkeypatch.setattr(kafka_module, "_producer", mock_producer)

        event = SyncCompletedEvent(
            job_id=2,
            job_type="vendor_sync",
            status="success",
            correlation_id="corr-abc",
        )
        publish_event(event)

        call_args = mock_producer.publish.call_args
        assert call_args[0][2] == "corr-abc"  # key parameter
