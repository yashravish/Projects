# Enterprise Integration Gateway

A production-quality integration platform that syncs customer, order, and shipment data between a mock CRM (JSON), a mock Vendor EDI feed (XML), and a normalized PostgreSQL datastore — with full job tracking, dead-letter retry, admin monitoring, and a unified REST API.

**Built to demonstrate entry-level enterprise integration engineering skills**: REST API integration, JSON + XML data exchange, ETL pipelines, system-to-system automation, SQL design, monitoring, error handling, and testing.

---

## Quick Start

```bash
# Clone and start everything with Docker Compose
git clone <repo-url>
cd enterprise-integration-gateway
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---------|-----|
| Main API + Swagger UI | http://localhost:8000/docs |
| Health Check | http://localhost:8000/api/v1/health |
| Admin Status | http://localhost:8000/api/v1/admin/status |
| Mock Providers (CRM + Vendor) | http://localhost:8001/docs |

### Seed the database

```bash
# Trigger a full sync (CRM + Vendor)
curl -s -X POST http://localhost:8000/api/v1/sync/all | python -m json.tool
```

---

## Architecture

```
Mock CRM (JSON, port 8001) ──────────┐
                                     ▼
Mock Vendor (XML, port 8001) ──► Integration Gateway (port 8000) ──► PostgreSQL
                                     │
                              APScheduler (15-min auto-sync)
```

Three Docker containers:
1. **`db`** — PostgreSQL 16
2. **`mock_providers`** — Simulated CRM (JSON) + Vendor (XML) on port 8001
3. **`app`** — FastAPI integration gateway on port 8000

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and component breakdown.

---

## Core Features

### Integration Flows
| Flow | Trigger | Source Format |
|------|---------|--------------|
| CRM Sync | `POST /sync/crm` or scheduled | JSON (camelCase) |
| Vendor Sync | `POST /sync/vendor` or scheduled | XML (PascalCase) |
| Full Sync | `POST /sync/all` | Both |

### ETL Pipeline
1. Fetch raw data from external source (JSON or XML)
2. **Validate** and **transform** into the internal normalized schema
3. **Upsert** into PostgreSQL (idempotent — safe to re-run)
4. Log counts: inserted / updated / failed per record
5. Capture failed records into `failed_records` dead-letter table

### Error Handling
- Malformed XML records → captured in `failed_records`, sync continues
- HTTP failures → retried up to 3× with exponential backoff
- Manual retry: `POST /api/v1/failed-records/{id}/retry`
- Records abandoned after `MAX_RETRY_COUNT` failures

### Monitoring
- `GET /health` — service + database connectivity
- `GET /metrics` — record counts
- `GET /admin/status` — full dashboard (counts, recent jobs, scheduler state)
- `GET /integration-jobs` — full sync history with timestamps and counts
- Structured JSON logs with `correlation_id` per sync job

---

## API Reference

Full documentation: [`docs/api-reference.md`](docs/api-reference.md)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/metrics` | Record counts |
| GET | `/api/v1/admin/status` | Full operational dashboard |
| POST | `/api/v1/sync/crm` | Trigger CRM sync |
| POST | `/api/v1/sync/vendor` | Trigger Vendor sync |
| POST | `/api/v1/sync/all` | Trigger full sync |
| GET | `/api/v1/customers` | List customers |
| GET | `/api/v1/customers/{id}` | Get customer |
| GET | `/api/v1/orders` | List orders |
| GET | `/api/v1/orders/{id}` | Get order |
| GET | `/api/v1/shipments` | List shipments |
| GET | `/api/v1/shipments/{id}` | Get shipment |
| GET | `/api/v1/integration-jobs` | List sync jobs |
| GET | `/api/v1/integration-jobs/{id}` | Get sync job |
| GET | `/api/v1/failed-records` | List failed records |
| POST | `/api/v1/failed-records/{id}/retry` | Retry a failed record |

### Sample `curl` Commands

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Trigger CRM sync
curl -X POST http://localhost:8000/api/v1/sync/crm

# Trigger Vendor sync (will show 1 malformed record)
curl -X POST http://localhost:8000/api/v1/sync/vendor

# List customers (CRM source only)
curl "http://localhost:8000/api/v1/customers?source=crm"

# View last 5 sync jobs
curl "http://localhost:8000/api/v1/integration-jobs?limit=5"

# View failed records
curl http://localhost:8000/api/v1/failed-records

# Retry a failed record (replace 1 with actual ID)
curl -X POST http://localhost:8000/api/v1/failed-records/1/retry

# Admin status dashboard
curl http://localhost:8000/api/v1/admin/status
```

---

## Postman

### Import
1. Open Postman → **Import**
2. Import `postman/Enterprise_Integration_Gateway.postman_collection.json`
3. Import `postman/EIG_Local.postman_environment.json`
4. Select **EIG Local** as the active environment

### Happy Path Flow
1. `GET /health` — verify service is up
2. `POST /sync/crm` — seed CRM data (stores `last_job_id`)
3. `POST /sync/vendor` — seed Vendor data (expect 1 failed record)
4. `GET /customers` — view normalized customers
5. `GET /orders` — view orders from both sources
6. `GET /shipments` — view shipments from vendor
7. `GET /integration-jobs` — view job history
8. `GET /admin/status` — operational dashboard

### Failure + Retry Flow
1. `POST /sync/vendor` — creates a malformed failed record
2. `GET /failed-records` — stores `last_failed_record_id`
3. `POST /failed-records/{{last_failed_record_id}}/retry` — attempt retry
4. Note the `status` in the response (`resolved` or `pending_retry`)

---

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/ -v          # Pure transformation/parsing logic
pytest tests/api/ -v           # HTTP endpoint tests
pytest tests/integration/ -v   # Full sync flow tests

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing
```

Tests use SQLite in memory — no running PostgreSQL required.

---

## Local Development (Without Docker)

```bash
# 1. Start PostgreSQL
docker run -d \
  --name eig_db \
  -e POSTGRES_DB=eig_db \
  -e POSTGRES_USER=eig_user \
  -e POSTGRES_PASSWORD=eig_password \
  -p 5432:5432 \
  postgres:16-alpine

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Start mock providers (Terminal 1)
uvicorn mock_providers.main:app --host 0.0.0.0 --port 8001 --reload

# 5. Start main app (Terminal 2)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://eig_user:eig_password@localhost:5432/eig_db` | PostgreSQL connection string |
| `CRM_BASE_URL` | `http://localhost:8001` | Base URL for mock CRM API |
| `VENDOR_BASE_URL` | `http://localhost:8001` | Base URL for mock Vendor API |
| `SCHEDULER_ENABLED` | `true` | Enable/disable background sync |
| `CRM_SYNC_INTERVAL_MINUTES` | `15` | CRM sync frequency |
| `VENDOR_SYNC_INTERVAL_MINUTES` | `15` | Vendor sync frequency |
| `MAX_RETRY_COUNT` | `3` | Max failed-record retries before abandonment |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | `json` (structured) or `text` (human-readable) |
| `HTTP_TIMEOUT_SECONDS` | `30` | Timeout for outbound HTTP calls |
| `HTTP_MAX_RETRIES` | `3` | Max HTTP retry attempts |

See `.env.example` for the full list.

---

## Project Structure

```
enterprise-integration-gateway/
├── app/
│   ├── main.py                  # FastAPI app, middleware, lifespan
│   ├── core/                    # Config, logging, exceptions, dependencies
│   ├── db/                      # SQLAlchemy engine, session, init
│   ├── models/                  # SQLAlchemy ORM models
│   ├── schemas/                 # Pydantic request/response schemas
│   ├── api/v1/                  # FastAPI route handlers
│   ├── services/                # Business logic: sync, upsert, retry
│   ├── clients/                 # HTTP clients (CRM, Vendor) with retry
│   ├── utils/                   # Transformers, XML/JSON parsers, retry
│   └── jobs/                    # APScheduler jobs
├── mock_providers/
│   ├── main.py                  # Mock provider FastAPI app (port 8001)
│   ├── crm/                     # Mock CRM: JSON customers + orders
│   └── vendor/                  # Mock Vendor: XML orders + shipments
├── alembic/                     # Database migrations
├── tests/
│   ├── unit/                    # Transformer + parser unit tests
│   ├── api/                     # Endpoint tests (TestClient + SQLite)
│   └── integration/             # Full sync flow tests (mocked HTTP)
├── postman/                     # Collection + environment JSON
├── docs/                        # Architecture, ERD, runbook, security
├── scripts/                     # health_check.sh, seed_db.sh
├── .github/workflows/ci.yml     # GitHub Actions CI
├── Dockerfile                   # Main app container
├── Dockerfile.mock              # Mock providers container
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Database Schema

Five tables — see [`docs/erd.md`](docs/erd.md) for the full ERD.

| Table | Purpose |
|-------|---------|
| `customers` | Normalized customer records from CRM and Vendor |
| `orders` | Normalized orders from CRM (JSON) and Vendor (XML) |
| `shipments` | Shipment tracking data from Vendor XML feed |
| `sync_jobs` | Audit log of every sync run with counts and status |
| `failed_records` | Dead-letter queue for records that failed transformation |

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [`docs/architecture.md`](docs/architecture.md) | System diagram, component responsibilities |
| [`docs/data-flow.md`](docs/data-flow.md) | Sequence diagrams for all sync flows |
| [`docs/erd.md`](docs/erd.md) | Entity relationship diagram |
| [`docs/api-reference.md`](docs/api-reference.md) | Full endpoint documentation |
| [`docs/runbook.md`](docs/runbook.md) | Operations guide, troubleshooting |
| [`docs/security.md`](docs/security.md) | Security considerations and SDLC practices |
| [`docs/sprint-backlog.md`](docs/sprint-backlog.md) | Simulated agile sprint history |

---

## Assumptions & Trade-offs

- **Mock providers as internal service**: In production, these would be real external systems. Keeping them internal makes the project fully self-contained for local demo.
- **APScheduler over Celery**: Simpler setup without requiring Redis/RabbitMQ. Celery would be preferred for high-throughput production workloads.
- **Sync over push**: Polling-based sync rather than webhook/event-driven. Suitable for batch integration; real-time would use webhooks or message queues.
- **SQLite for tests**: Avoids test database infrastructure. Uses `create_all()` instead of Alembic in test context.
- **No authentication on sync endpoints**: Out of scope for v1. Production would require API keys or JWT.

---

## How This Project Matches Enterprise Integration Roles

| Job Requirement | What This Project Demonstrates |
|----------------|--------------------------------|
| REST API integration | CRM client, Vendor client, unified internal API |
| JSON data handling | CRM JSON parsing, transformation, validation |
| XML data handling | Vendor XML parsing, malformed record handling |
| ETL / data pipeline | Full extract → transform → load flow |
| System-to-system automation | Scheduled background sync jobs |
| SQL / database design | Normalized schema, FK relationships, indexes |
| Error handling / troubleshooting | Failed records, retry logic, structured logging |
| Monitoring | Health check, admin dashboard, sync job history |
| Testing | Unit, API, and integration tests |
| Docker / DevOps | Docker Compose, multi-container setup |
| CI/CD | GitHub Actions workflow |
| Documentation | Architecture, ERD, runbook, API reference |
| Agile practices | Sprint backlog, CHANGELOG, modular structure |

---

## Resume Bullet Points

```
• Built an Enterprise Integration Gateway in Python/FastAPI that synchronizes customer,
  order, and shipment data between a JSON CRM API and an XML vendor EDI feed into a
  normalized PostgreSQL database via scheduled ETL pipelines.

• Designed a 5-table PostgreSQL schema with upsert-based ETL supporting idempotent
  sync runs, a dead-letter queue for failed records with retry logic, and full sync
  job audit history.

• Implemented safe XML parsing with malformed-record capture and JSON transformation
  pipelines, converting camelCase CRM fields and PascalCase XML tags into a unified
  internal schema using Pydantic v2 validators.

• Built a REST API with 15+ endpoints exposing integrated data, sync triggers, job
  history, failed-record management, and operational health/metrics dashboards.

• Achieved 85%+ test coverage with pytest unit tests (transformers, XML/JSON parsers),
  API tests (FastAPI TestClient + SQLite), and integration tests (monkeypatched HTTP
  clients) — all running without external dependencies.

• Containerized the entire platform with Docker Compose (3 services) and configured
  a GitHub Actions CI pipeline running the full test suite on every commit.
```
