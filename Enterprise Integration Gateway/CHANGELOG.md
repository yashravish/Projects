# Changelog

All notable changes to the Enterprise Integration Gateway are documented here.

---

## [2.0.0] — 2024-04-16

### Added
- **Redis caching**: response caching on all GET endpoints (customers, orders, shipments, metrics) via `@cached` decorator with configurable TTL and automatic invalidation after sync operations
- **Redis rate limiting**: sliding-window rate limiter on `POST /sync/*` endpoints using Redis sorted sets with per-IP isolation and `429 Too Many Requests` + `Retry-After` header
- **Kafka event producer**: publishes `sync.started`, `sync.completed`, and `record.failed` events to `eig.integration.events` topic with correlation IDs for distributed tracing
- **Kafka event consumer**: background consumer on `eig.inbound.sync.requests` topic for async sync triggering from external systems
- **Events API**: `POST /events/publish` for manual event publishing and `GET /events/status` for Kafka connectivity status
- **AWS ECS deployment**: Fargate task/service definitions, CodeDeploy AppSpec for blue/green deployments
- **AWS Lambda deployment**: Mangum handler wrapping FastAPI for API Gateway + EventBridge scheduled sync rules via SAM template
- **AWS CloudFormation**: full infrastructure stack (VPC, subnets, NAT gateway, RDS PostgreSQL, ElastiCache Redis, MSK Kafka, ECS cluster, ALB, security groups, CloudWatch logs, SSM parameters)
- **AWS CodeBuild**: buildspec for Docker image build/push to ECR with test suite gate
- **AWS deployment guide**: comprehensive docs covering ECS and Lambda paths, auto-scaling, monitoring, and cost estimation
- **Enhanced health check**: now includes Redis and Kafka connectivity status alongside database
- **Enhanced admin dashboard**: shows Redis and Kafka status in operational snapshot
- **Docker Compose**: Redis 7 (alpine) and Kafka 7.6 (KRaft mode, no Zookeeper) services added
- **New tests**: unit tests for cache decorator, rate limiter, and event publisher using fakeredis
- **Graceful degradation**: Redis and Kafka are fully optional — app operates normally without them

### Changed
- Bumped app version from 1.0.0 to 2.0.0
- Docker Compose now runs 5 services (was 3): added Redis and Kafka
- Dockerfile installs `librdkafka-dev` for confluent-kafka
- CI workflow adds `REDIS_ENABLED=false` and `KAFKA_ENABLED=false` for test isolation
- Sync service now emits Kafka events at job start, job finish, and on record failures
- Sync trigger endpoints now include rate limiter dependency
- GET endpoints now include `@cached` decorator with `request: Request` parameter

### Technical Decisions
- Used Redis sorted sets for rate limiting (vs. token bucket) for accurate sliding-window behavior
- Used confluent-kafka (C-based librdkafka) over aiokafka for production reliability
- Used Kafka KRaft mode to eliminate Zookeeper dependency in Docker Compose
- Used Mangum for Lambda deployment to avoid rewriting the FastAPI app
- CloudFormation over CDK/Terraform to stay within AWS-native tooling

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
- [Planned] Prometheus + Grafana dashboards
- [Planned] MSK Lambda trigger (replace polling consumer)
