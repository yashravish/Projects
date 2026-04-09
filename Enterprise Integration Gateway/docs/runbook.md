# Runbook — Operations Guide

## Starting the Application

### Docker Compose (recommended)
```bash
git clone <repo-url>
cd enterprise-integration-gateway
cp .env.example .env

docker compose up --build
```

Services:
- Main app: http://localhost:8000
- Mock providers: http://localhost:8001
- PostgreSQL: localhost:5432

### Local Development (without Docker)
```bash
# 1. Start PostgreSQL (Docker recommended)
docker run -d \
  --name eig_db \
  -e POSTGRES_DB=eig_db \
  -e POSTGRES_USER=eig_user \
  -e POSTGRES_PASSWORD=eig_password \
  -p 5432:5432 \
  postgres:16-alpine

# 2. Create virtualenv and install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Copy env file
cp .env.example .env

# 4. Start mock providers (terminal 1)
uvicorn mock_providers.main:app --host 0.0.0.0 --port 8001 --reload

# 5. Start main app (terminal 2)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Running Alembic Migrations

```bash
# Run all pending migrations
alembic upgrade head

# Check current migration
alembic current

# Generate a new migration (after model changes)
alembic revision --autogenerate -m "add_new_column"

# Rollback one step
alembic downgrade -1
```

## Triggering Syncs Manually

```bash
# CRM sync
curl -s -X POST http://localhost:8000/api/v1/sync/crm | python -m json.tool

# Vendor sync
curl -s -X POST http://localhost:8000/api/v1/sync/vendor | python -m json.tool

# Full sync
curl -s -X POST http://localhost:8000/api/v1/sync/all | python -m json.tool
```

## Monitoring

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```
Expected: `{"status": "healthy", "checks": {"database": "ok"}, ...}`

### Metrics
```bash
curl http://localhost:8000/api/v1/metrics
```

### Admin Status (operational dashboard)
```bash
curl http://localhost:8000/api/v1/admin/status
```
Shows: record counts, recent sync jobs, failed record summary, scheduler state.

### View Recent Jobs
```bash
curl "http://localhost:8000/api/v1/integration-jobs?limit=10" | python -m json.tool
```

### View Failed Records
```bash
curl "http://localhost:8000/api/v1/failed-records?status=pending_retry" | python -m json.tool
```

## Troubleshooting

### Sync job shows `failed` status

1. Check the job's `error_message`:
   ```bash
   curl http://localhost:8000/api/v1/integration-jobs/{job_id}
   ```

2. Common causes:
   - **Connection refused to mock providers**: Ensure `mock_providers` container is healthy
   - **Database connection error**: Check `DATABASE_URL` in `.env`
   - **Transformation error in all records**: Check for schema changes in mock data

3. View application logs:
   ```bash
   docker compose logs app --tail=100
   ```

### Failed records accumulating

1. List all pending failed records:
   ```bash
   curl "http://localhost:8000/api/v1/failed-records?status=pending_retry"
   ```

2. Inspect a specific failed record's `raw_data` and `error_message`.

3. Attempt retry:
   ```bash
   curl -X POST http://localhost:8000/api/v1/failed-records/{id}/retry
   ```

4. If `status=abandoned`, the record has exceeded `MAX_RETRY_COUNT`. Review the error and manually intervene.

### Duplicate customers / orders appearing

The upsert logic uses `external_id` as the idempotency key. If duplicates appear:
- Check that `external_id` values are stable across syncs from the source system
- Verify the `external_id` mapping in `transformers.py`

### Scheduler not running

Check scheduler state:
```bash
curl http://localhost:8000/api/v1/admin/status | python -m json.tool
```
Look at `scheduler.running`. If `false`, check `SCHEDULER_ENABLED=true` in `.env`.

### Database schema issues

Reset the DB (development only — **destroys all data**):
```bash
docker compose down -v
docker compose up --build
```

## Log Reference

| Log Message | Meaning |
|-------------|---------|
| `sync_job_started` | A sync job has been created and is running |
| `sync_job_finished` | Sync completed; check `status` field |
| `crm_customer_failed` | A CRM customer record failed transformation |
| `vendor_order_malformed` | XML parser rejected a vendor order record |
| `failed_record_created` | A failed record was written to dead-letter queue |
| `failed_record_resolved` | A retry succeeded |
| `http_request / http_response` | Outbound HTTP to mock provider |
| `retry_backoff` | HTTP retry is sleeping before next attempt |
| `health_check_db_error` | DB unreachable during health check |
