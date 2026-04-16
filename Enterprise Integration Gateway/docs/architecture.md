# Architecture Overview

## System Architecture

```
                         ┌──────────────────────────────────────────────────────┐
                         │          Enterprise Integration Gateway              │
                         │                                                      │
  ┌──────────────┐       │  ┌─────────────────────────────────────────────┐     │
  │  Mock CRM    │       │  │            FastAPI App                      │     │
  │  (JSON API)  │◄──────┼──│  /api/v1/sync/crm        (rate limited)    │     │
  │  port 8001   │       │  │  /api/v1/sync/vendor     (rate limited)    │     │
  └──────────────┘       │  │  /api/v1/sync/all        (rate limited)    │     │
                         │  │  /api/v1/customers       (cached)          │     │
  ┌──────────────┐       │  │  /api/v1/orders          (cached)          │     │
  │  Mock Vendor │       │  │  /api/v1/shipments       (cached)          │     │
  │  (XML Feed)  │◄──────┼──│  /api/v1/integration-jobs                  │     │
  │  port 8001   │       │  │  /api/v1/failed-records                    │     │
  └──────────────┘       │  │  /api/v1/health                            │     │
                         │  │  /api/v1/metrics          (cached)         │     │
  ┌──────────────┐       │  │  /api/v1/admin/status                      │     │
  │  Postman /   │       │  │  /api/v1/events/publish                    │     │
  │  curl /      │──────►│  │  /api/v1/events/status                     │     │
  │  any client  │       │  └──────────┬─────────────────┬───────────────┘     │
  └──────────────┘       │             │                 │                      │
                         │  ┌──────────▼──────────┐ ┌────▼────────────────┐     │
                         │  │  APScheduler        │ │ Kafka Event        │     │
                         │  │  Background Jobs    │ │ Consumer (inbound) │     │
                         │  │  (15-min intervals) │ │ (daemon thread)    │     │
                         │  └──────────┬──────────┘ └────────────────────┘     │
                         │             │                                        │
                         │  ┌──────────▼──────────────────────────────────┐     │
                         │  │         PostgreSQL Database                 │     │
                         │  │  customers │ orders │ shipments            │     │
                         │  │  sync_jobs │ failed_records                │     │
                         │  └─────────────────────────────────────────────┘     │
                         │                                                      │
                         │  ┌──────────────────┐  ┌──────────────────────┐      │
                         │  │    Redis 7       │  │    Kafka (KRaft)    │      │
                         │  │  Response cache  │  │  Event producer     │      │
                         │  │  Rate limiting   │  │  Event consumer     │      │
                         │  │  (sorted sets)   │  │  (async sync reqs) │      │
                         │  └──────────────────┘  └──────────────────────┘      │
                         └──────────────────────────────────────────────────────┘
```

## AWS Production Architecture

```
                    Internet
                       │
                ┌──────▼──────┐
                │     ALB     │ ◄── Public subnets
                └──────┬──────┘
                       │
        ┌──────────────┼──────────────┐
  ┌─────▼─────┐  ┌─────▼─────┐       │    Private subnets
  │ ECS Task  │  │ ECS Task  │       │
  │ (Fargate) │  │ (Fargate) │       │
  └─────┬─────┘  └─────┬─────┘       │
        │              │              │
   ┌────┴──────────────┴──────────────┴────┐
   │  ┌──────────┐  ┌────────┐  ┌───────┐  │
   │  │ RDS      │  │ Elasti │  │ MSK   │  │
   │  │ Postgres │  │ Cache  │  │ Kafka │  │
   │  └──────────┘  └────────┘  └───────┘  │
   └────────────────────────────────────────┘
```

See [`docs/aws-deployment.md`](aws-deployment.md) for full deployment instructions.

## Component Responsibilities

### API Layer (`app/api/v1/`)
- Receives HTTP requests via FastAPI
- Validates path/query parameters via Pydantic
- Delegates to service layer, returns typed responses
- Applies `@cached` decorator on GET endpoints for Redis caching
- Applies `RateLimiter` dependency on POST sync endpoints
- No business logic — thin controllers only

### Service Layer (`app/services/`)
- Owns business logic for sync orchestration
- Calls HTTP clients, transformers, and entity services
- Manages `SyncJob` lifecycle (pending → running → terminal)
- Handles per-record failures and writes to `failed_records`
- Publishes Kafka events at sync boundaries and on record failures
- All functions accept a `db: Session` parameter for testability

### HTTP Clients (`app/clients/`)
- Thin wrappers over httpx with retry logic
- `CrmClient`: fetches JSON from mock CRM
- `VendorClient`: fetches XML from mock vendor
- `BaseHttpClient`: shared retry + structured logging

### Transformation Utilities (`app/utils/transformers.py`)
- Pure functions: raw dict → Pydantic schema
- No DB access — fully unit-testable
- Raise `TransformationError` on unrecoverable issues

### XML Parser (`app/utils/xml_parser.py`)
- Safe XML parsing using stdlib `ElementTree`
- Returns `(valid_records[], malformed_records[])`
- Never raises — captures errors into the malformed list

### Redis Layer (`app/core/`)
- `redis_client.py`: singleton connection pool with graceful degradation
- `cache.py`: `@cached` decorator with TTL, `invalidate_cache()` via SCAN
- `rate_limiter.py`: sliding-window limiter using Redis sorted sets

### Kafka Layer (`app/core/` + `app/jobs/`)
- `kafka_client.py`: `KafkaEventProducer` / `KafkaEventConsumer` wrappers
- `event_publisher.py`: `publish_event()` helper used by sync service
- `event_consumer.py`: background consumer for inbound sync requests

### Background Scheduler (`app/jobs/`)
- APScheduler `BackgroundScheduler` with `IntervalTrigger`
- Each job opens its own DB session, calls sync_service, closes session
- Runs in a daemon thread alongside the FastAPI event loop

### Mock Providers (`mock_providers/`)
- Separate FastAPI app on port 8001
- CRM endpoints return JSON with camelCase fields
- Vendor endpoints return XML with PascalCase tags
- One intentionally malformed vendor order to test error handling

## Technology Rationale

| Technology | Why |
|-----------|-----|
| FastAPI | Type-annotated, OpenAPI auto-docs, fast dev cycle |
| SQLAlchemy 2.0 | Industry-standard ORM, migration-friendly via Alembic |
| PostgreSQL JSONB | Store raw payloads for audit without schema migrations |
| Redis 7 | Sub-millisecond caching, sorted sets for rate limiting |
| Kafka (confluent-kafka) | Industry-standard event streaming, C-based for performance |
| APScheduler 3.x | Simple, no additional broker required (vs. Celery) |
| httpx | Modern sync/async HTTP client with type hints |
| Pydantic v2 | Enforced validation at every schema boundary |
| Docker Compose | One-command reproducible environment (5 containers) |
| Mangum | Wraps FastAPI for AWS Lambda execution |
| AWS ECS Fargate | Serverless containers for production workloads |
| CloudFormation | Infrastructure as Code for repeatable deployments |

## Design Patterns Used

- **Upsert pattern** (`external_id` as idempotency key) — safe to re-run syncs
- **Dead-letter queue** (`failed_records`) — captures errors without data loss
- **Correlation ID** — unique UUID per sync job for distributed tracing
- **Transformation layer** — isolates external schema from internal model
- **Repository-style services** — all DB access through typed service functions
- **Dependency injection** — `db: Session = Depends(get_db)` for testability
- **Cache-aside pattern** — check Redis before DB, populate on miss
- **Sliding-window rate limiting** — Redis sorted sets for accurate per-IP limits
- **Event-driven architecture** — Kafka events for async system integration
- **Graceful degradation** — Redis/Kafka unavailability doesn't break core flows
- **Infrastructure as Code** — CloudFormation for reproducible AWS deployments
