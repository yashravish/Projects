# Sprint Backlog — Simulated Agile History

This document shows how the project was structured as if built by an agile engineering team.

---

## Sprint 1 — Foundation (2 weeks)

**Goal**: Working skeleton with database, mock providers, and basic sync

| Story | Points | Status |
|-------|--------|--------|
| Set up project structure and Docker Compose | 3 | Done |
| Design and implement database schema (5 tables) | 5 | Done |
| Build mock CRM API (JSON customers + orders) | 3 | Done |
| Build mock Vendor API (XML orders + shipments) | 3 | Done |
| Implement CRM HTTP client with retry logic | 3 | Done |
| Implement Vendor HTTP client | 2 | Done |
| Basic CRM sync job (customers only) | 5 | Done |

**Sprint Review Notes**: DB schema finalized. Mock providers are callable and return realistic payloads. Basic CRM customer sync runs end-to-end.

---

## Sprint 2 — Core ETL Pipeline (2 weeks)

**Goal**: Full transformation layer, XML parsing, and vendor sync

| Story | Points | Status |
|-------|--------|--------|
| Implement XML parser for vendor order feed | 5 | Done |
| Implement XML parser for vendor shipment feed | 3 | Done |
| Build CRM order transformer and upsert | 5 | Done |
| Build vendor order transformer with FK resolution | 5 | Done |
| Build vendor shipment transformer | 3 | Done |
| Add intentional malformed record to vendor XML | 2 | Done |
| Implement `failed_records` dead-letter tracking | 5 | Done |
| SyncJob lifecycle (pending → running → terminal) | 3 | Done |

**Sprint Review Notes**: ETL pipeline fully operational. Malformed vendor record goes to `failed_records` as expected. Upsert is idempotent — re-running syncs does not create duplicates.

---

## Sprint 3 — Unified API + Monitoring (2 weeks)

**Goal**: Full REST API, health checks, admin endpoints

| Story | Points | Status |
|-------|--------|--------|
| Implement `/customers`, `/orders`, `/shipments` endpoints | 5 | Done |
| Implement `/integration-jobs` endpoints | 3 | Done |
| Implement `/failed-records` with retry endpoint | 5 | Done |
| Implement `/sync/crm`, `/sync/vendor`, `/sync/all` | 3 | Done |
| Health check endpoint with DB connectivity | 2 | Done |
| Admin status dashboard endpoint | 3 | Done |
| Metrics endpoint (record counts) | 2 | Done |
| APScheduler background sync (15-min interval) | 5 | Done |

---

## Sprint 4 — Quality & Documentation (1 week)

**Goal**: Tests, Postman collection, docs

| Story | Points | Status |
|-------|--------|--------|
| Unit tests for all transformer functions | 5 | Done |
| Unit tests for XML and JSON parsers | 5 | Done |
| API tests for all endpoints | 8 | Done |
| Integration tests for CRM sync (with mock) | 5 | Done |
| Integration tests for vendor sync (with mock) | 5 | Done |
| Postman collection + environment file | 3 | Done |
| Architecture and ERD documentation | 3 | Done |
| Runbook and troubleshooting guide | 3 | Done |
| GitHub Actions CI workflow | 2 | Done |
| README and CHANGELOG | 2 | Done |

---

## Backlog (Future Sprints)

| Story | Priority | Notes |
|-------|----------|-------|
| JWT authentication for sync endpoints | High | Prevent unauthorized triggers |
| Rate limiting on `/sync/*` | Medium | Prevent accidental flood of jobs |
| Webhook support (push-based sync trigger) | Medium | More real-time than polling |
| Delta sync (only changed records) | High | Reduce DB write load at scale |
| Alembic migration CI gate | Medium | Ensure migrations are always up to date |
| Prometheus + Grafana dashboards | Medium | Replace custom metrics endpoint |
| Multi-source routing (add Source C) | Low | Extensibility demonstration |
| PII field masking in logs | High | GDPR/compliance requirement |
