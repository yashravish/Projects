# Architecture Overview

## System Architecture

```
                         ┌─────────────────────────────────────┐
                         │       Enterprise Integration        │
                         │            Gateway                  │
                         │                                     │
  ┌──────────────┐        │  ┌─────────────────────────────┐   │
  │  Mock CRM    │        │  │       FastAPI App            │   │
  │  (JSON API)  │◄───────┼──│  /api/v1/sync/crm           │   │
  │  port 8001   │        │  │  /api/v1/sync/vendor        │   │
  └──────────────┘        │  │  /api/v1/sync/all           │   │
                          │  │  /api/v1/customers          │   │
  ┌──────────────┐        │  │  /api/v1/orders             │   │
  │  Mock Vendor │        │  │  /api/v1/shipments          │   │
  │  (XML Feed)  │◄───────┼──│  /api/v1/integration-jobs  │   │
  │  port 8001   │        │  │  /api/v1/failed-records     │   │
  └──────────────┘        │  │  /api/v1/health             │   │
                          │  │  /api/v1/metrics            │   │
  ┌──────────────┐        │  │  /api/v1/admin/status       │   │
  │  Postman /   │        │  └────────────┬────────────────┘   │
  │  curl /      │───────►│               │                    │
  │  any client  │        │  ┌────────────▼────────────────┐   │
  └──────────────┘        │  │   APScheduler Background   │   │
                          │  │   Jobs (15-min intervals)  │   │
                          │  │   crm_sync_job             │   │
                          │  │   vendor_sync_job          │   │
                          │  └────────────┬────────────────┘   │
                          │               │                    │
                          │  ┌────────────▼────────────────┐   │
                          │  │    PostgreSQL Database      │   │
                          │  │    customers                │   │
                          │  │    orders                   │   │
                          │  │    shipments                │   │
                          │  │    sync_jobs                │   │
                          │  │    failed_records           │   │
                          │  └─────────────────────────────┘   │
                          └─────────────────────────────────────┘
```

## Component Responsibilities

### API Layer (`app/api/v1/`)
- Receives HTTP requests via FastAPI
- Validates path/query parameters via Pydantic
- Delegates to service layer, returns typed responses
- No business logic — thin controllers only

### Service Layer (`app/services/`)
- Owns business logic for sync orchestration
- Calls HTTP clients, transformers, and entity services
- Manages `SyncJob` lifecycle (pending → running → terminal)
- Handles per-record failures and writes to `failed_records`
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
| APScheduler 3.x | Simple, no additional broker required (vs. Celery) |
| httpx | Modern sync/async HTTP client with type hints |
| Pydantic v2 | Enforced validation at every schema boundary |
| Docker Compose | One-command reproducible environment |

## Design Patterns Used

- **Upsert pattern** (`external_id` as idempotency key) — safe to re-run syncs
- **Dead-letter queue** (`failed_records`) — captures errors without data loss
- **Correlation ID** — unique UUID per sync job for distributed tracing
- **Transformation layer** — isolates external schema from internal model
- **Repository-style services** — all DB access through typed service functions
- **Dependency injection** — `db: Session = Depends(get_db)` for testability
