"""
Kafka producer and consumer wrappers.

Provides ``KafkaEventProducer`` for publishing integration events and
``KafkaEventConsumer`` for processing inbound sync requests. Both are
designed for graceful degradation — the application continues to operate
normally if Kafka is unavailable.
"""
import json
import logging
import threading
from typing import Any, Callable, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Lazy imports — confluent_kafka may not be installed in all environments ───

_Producer = None
_Consumer = None


def _load_confluent_kafka():
    global _Producer, _Consumer
    try:
        from confluent_kafka import Producer, Consumer
        _Producer = Producer
        _Consumer = Consumer
        return True
    except ImportError:
        logger.warning("confluent_kafka_not_installed", extra={
            "message": "confluent-kafka is not installed. Kafka features are disabled."
        })
        return False


# ── Producer ──────────────────────────────────────────────────────────────────


class KafkaEventProducer:
    """
    Thread-safe Kafka event producer.

    Serializes Python dicts to JSON and publishes them to configured topics.
    All errors are caught and logged — never fatal to the application.
    """

    def __init__(self) -> None:
        self._producer: Optional[Any] = None
        self._available = False

        if not settings.KAFKA_ENABLED:
            logger.info("kafka_producer_disabled", extra={"reason": "KAFKA_ENABLED=false"})
            return

        if not _load_confluent_kafka():
            return

        try:
            self._producer = _Producer({
                "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
                "client.id": "eig-gateway-producer",
                "acks": "all",
                "retries": 3,
                "retry.backoff.ms": 500,
                "linger.ms": 5,
            })
            self._available = True
            logger.info("kafka_producer_initialized", extra={
                "bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            })
        except Exception as exc:
            logger.warning("kafka_producer_init_failed", extra={"error": str(exc)})

    @property
    def is_available(self) -> bool:
        return self._available

    def publish(self, topic: str, event: dict, key: Optional[str] = None) -> bool:
        """
        Publish a JSON event to the specified Kafka topic.

        Returns True if the message was enqueued successfully, False otherwise.
        """
        if not self._available or self._producer is None:
            logger.debug("kafka_publish_skipped", extra={"topic": topic, "reason": "unavailable"})
            return False

        try:
            value = json.dumps(event, default=str).encode("utf-8")
            encoded_key = key.encode("utf-8") if key else None
            self._producer.produce(
                topic=topic,
                value=value,
                key=encoded_key,
                callback=self._delivery_callback,
            )
            self._producer.poll(0)  # trigger delivery callbacks
            return True
        except Exception as exc:
            logger.warning("kafka_publish_failed", extra={
                "topic": topic,
                "error": str(exc),
            })
            return False

    def flush(self, timeout: float = 5.0) -> None:
        """Flush pending messages. Called during shutdown."""
        if self._producer is not None:
            self._producer.flush(timeout=timeout)

    @staticmethod
    def _delivery_callback(err, msg) -> None:
        if err is not None:
            logger.warning("kafka_delivery_failed", extra={
                "topic": msg.topic(),
                "error": str(err),
            })
        else:
            logger.debug("kafka_delivery_success", extra={
                "topic": msg.topic(),
                "partition": msg.partition(),
                "offset": msg.offset(),
            })


# ── Consumer ──────────────────────────────────────────────────────────────────


class KafkaEventConsumer:
    """
    Background Kafka consumer that processes messages from a subscribed topic.

    Runs in a daemon thread and invokes a user-supplied callback for each
    message. Designed for fire-and-forget consumption of inbound sync requests.
    """

    def __init__(
        self,
        topics: list[str],
        handler: Callable[[dict], None],
        group_id: Optional[str] = None,
    ) -> None:
        self._consumer: Optional[Any] = None
        self._handler = handler
        self._topics = topics
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._available = False

        if not settings.KAFKA_ENABLED:
            logger.info("kafka_consumer_disabled", extra={"reason": "KAFKA_ENABLED=false"})
            return

        if not _load_confluent_kafka():
            return

        try:
            self._consumer = _Consumer({
                "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
                "group.id": group_id or settings.KAFKA_CONSUMER_GROUP,
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
                "session.timeout.ms": 30000,
            })
            self._available = True
            logger.info("kafka_consumer_initialized", extra={
                "topics": topics,
                "group_id": group_id or settings.KAFKA_CONSUMER_GROUP,
            })
        except Exception as exc:
            logger.warning("kafka_consumer_init_failed", extra={"error": str(exc)})

    @property
    def is_available(self) -> bool:
        return self._available

    def start(self) -> None:
        """Start the consumer in a background daemon thread."""
        if not self._available or self._consumer is None:
            return

        self._running = True
        self._consumer.subscribe(self._topics)
        self._thread = threading.Thread(
            target=self._consume_loop,
            name="kafka-consumer",
            daemon=True,
        )
        self._thread.start()
        logger.info("kafka_consumer_started", extra={"topics": self._topics})

    def stop(self) -> None:
        """Signal the consumer loop to stop and wait for the thread to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception:
                pass
        logger.info("kafka_consumer_stopped")

    def _consume_loop(self) -> None:
        """Main polling loop — runs in the daemon thread."""
        while self._running:
            try:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.warning("kafka_consume_error", extra={"error": str(msg.error())})
                    continue

                value = json.loads(msg.value().decode("utf-8"))
                logger.info("kafka_message_received", extra={
                    "topic": msg.topic(),
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                })
                self._handler(value)

            except json.JSONDecodeError as exc:
                logger.warning("kafka_message_parse_error", extra={"error": str(exc)})
            except Exception as exc:
                logger.error("kafka_consume_handler_error", extra={"error": str(exc)})


# ── Singleton instances ───────────────────────────────────────────────────────

_producer: Optional[KafkaEventProducer] = None
_consumer: Optional[KafkaEventConsumer] = None


def init_kafka_producer() -> KafkaEventProducer:
    """Initialize the global Kafka producer. Called at app startup."""
    global _producer
    _producer = KafkaEventProducer()
    return _producer


def get_kafka_producer() -> Optional[KafkaEventProducer]:
    """Return the global Kafka producer instance."""
    return _producer


def get_kafka_status() -> dict:
    """Return Kafka connectivity status for health checks."""
    if not settings.KAFKA_ENABLED:
        return {"enabled": False, "status": "disabled"}
    if _producer is None or not _producer.is_available:
        return {"enabled": True, "status": "disconnected"}
    return {
        "enabled": True,
        "status": "ok",
        "bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS,
        "events_topic": settings.KAFKA_EVENTS_TOPIC,
    }


def shutdown_kafka() -> None:
    """Flush producer and stop consumer. Called at app shutdown."""
    global _producer, _consumer
    if _producer is not None:
        _producer.flush()
        _producer = None
    if _consumer is not None:
        _consumer.stop()
        _consumer = None
