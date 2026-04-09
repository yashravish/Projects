# Changelog

All notable changes to the Enterprise Integration Gateway are documented here.

---

## [1.0.0] — 2024-03-15

### Added
- Full CRM JSON sync: customers and orders from mock CRM API
- Full Vendor XML sync: orders and shipments from mock Vendor XML feed
- PostgreSQL schema: customers, orders, shipments, sync_jobs, failed_records
- Dead-letter queue: failed records tracked with retry support
- REST API: customers, orders, shipments, integration-jobs, failed-records, sync
- APScheduler background sync (15-minute interval for CRM and Vendor)
- Health check endpoint with database connectivity verification
- Admin status dashboard: record counts, recent jobs, failed record summary
- Metrics endpoint: lightweight record counts
- Structured JSON logging with correlation IDs
- Retry logic for HTTP calls (exponential backoff)
- Docker Compose for one-command local setup
- Alembic migration: initial schema
- GitHub Actions CI workflow
- Postman collection + environment file
- Architecture, ERD, data-flow, runbook, and security documentation
- Unit tests for transformers, XML parser, JSON parser
- API tests for all endpoints
- Integration tests for CRM and Vendor sync flows (with monkeypatched HTTP)

### Technical Decisions
- Used APScheduler over Celery to avoid broker dependency in local dev
- Used SQLite for test isolation; PostgreSQL for production
- Used stdlib `ElementTree` for XML parsing (no network entity fetches)
- Kept mock providers as a separate FastAPI service to simulate real enterprise separation

---

## Future

- [Planned] JWT auth on sync trigger endpoints
- [Planned] Delta sync (only changed records since last run)
- [Planned] Webhook-based push sync trigger
- [Planned] Prometheus metrics integration
