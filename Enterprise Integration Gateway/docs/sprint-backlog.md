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

## Sprint 5 — Redis, Kafka & AWS Deployment (2 weeks)

**Goal**: Production-grade infrastructure — caching, event streaming, cloud deployment

| Story | Points | Status |
|-------|--------|--------|
| Redis connection factory with graceful degradation | 3 | Done |
| `@cached` decorator for GET endpoint response caching | 5 | Done |
| Sliding-window rate limiter (Redis sorted sets) | 5 | Done |
| Apply caching to customers, orders, shipments, metrics | 3 | Done |
| Apply rate limiter to sync trigger endpoints | 2 | Done |
| Cache invalidation after sync writes | 2 | Done |
| Kafka producer/consumer wrappers (confluent-kafka) | 5 | Done |
| Event schemas (SyncStarted, SyncCompleted, RecordFailed) | 3 | Done |
| Integrate event publishing into sync_service | 3 | Done |
| Kafka inbound sync request consumer (daemon thread) | 5 | Done |
| Events API endpoints (publish + status) | 2 | Done |
| Enhanced health check (Redis + Kafka status) | 2 | Done |
| Docker Compose: add Redis + Kafka services | 3 | Done |
| ECS Fargate task/service definitions | 3 | Done |
| Lambda handler (Mangum) + SAM template | 5 | Done |
| CloudFormation infrastructure stack (VPC, RDS, Redis, MSK, ALB) | 8 | Done |
| CodeBuild buildspec for ECR push | 2 | Done |
| AWS deployment documentation | 3 | Done |
| Unit tests: cache, rate limiter, event publisher | 5 | Done |
| Update CI workflow for Redis/Kafka env isolation | 2 | Done |
| Update all project documentation | 3 | Done |

**Sprint Review Notes**: v2.0.0 release. Redis caching delivers sub-millisecond response times on cached endpoints. Rate limiter protects sync endpoints from accidental flooding. Kafka events provide full audit trail for sync lifecycle. AWS manifests are validated and ready for deployment. All features designed with graceful degradation — core sync operations work without Redis or Kafka.

---

## Backlog (Future Sprints)

| Story | Priority | Notes |
|-------|----------|-------|
| JWT authentication for sync endpoints | High | Prevent unauthorized triggers |
| Delta sync (only changed records) | High | Reduce DB write load at scale |
| Webhook support (push-based sync trigger) | Medium | More real-time than polling |
| Alembic migration CI gate | Medium | Ensure migrations are always up to date |
| Prometheus + Grafana dashboards | Medium | Replace custom metrics endpoint |
| MSK Lambda trigger integration | Medium | Replace polling Kafka consumer in Lambda |
| Multi-source routing (add Source C) | Low | Extensibility demonstration |
| PII field masking in logs | High | GDPR/compliance requirement |
| Redis cluster mode support | Low | High-availability caching |
| Kafka schema registry integration | Medium | Enforce event contract compatibility |
