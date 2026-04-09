# API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive docs (Swagger UI): `http://localhost:8000/docs`
ReDoc: `http://localhost:8000/redoc`

---

## Health & Monitoring

### `GET /health`
Application health check including DB connectivity.

**Response 200**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development",
  "checks": {
    "database": "ok"
  }
}
```

### `GET /metrics`
Lightweight record count metrics.

**Response 200**:
```json
{
  "eig_customers_total": 5,
  "eig_orders_total": 9,
  "eig_shipments_total": 4,
  "eig_failed_records_total": 1,
  "eig_sync_jobs_total": 3
}
```

### `GET /admin/status`
Full operational dashboard.

**Response 200**:
```json
{
  "record_counts": { "customers": 5, "orders": 9, "shipments": 4 },
  "failed_records": { "pending_retry": 1, "abandoned": 0, "resolved": 0 },
  "recent_sync_jobs": [ ... ],
  "scheduler": { "running": true, "jobs": [ ... ] }
}
```

---

## Sync Triggers

### `POST /sync/crm`
Trigger CRM JSON sync.

**Response 200 — SyncResult**:
```json
{
  "job_id": 1,
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_type": "crm_sync",
  "status": "success",
  "records_processed": 11,
  "records_inserted": 10,
  "records_updated": 0,
  "records_failed": 1,
  "message": "CRM sync completed. 10 inserted, 0 updated, 1 failed."
}
```

### `POST /sync/vendor`
Trigger Vendor XML sync. Expect `records_failed >= 1` due to intentional malformed record.

### `POST /sync/all`
Trigger CRM + Vendor syncs sequentially. Returns aggregated result with `job_type=full_sync`.

---

## Customers

### `GET /customers`
**Query params**:
- `source` — filter by `crm` or `vendor`
- `status` — filter by `active` or `inactive`
- `skip` — pagination offset (default 0)
- `limit` — page size (default 100, max 500)

**Response 200**: Array of `CustomerResponse`

```json
[
  {
    "id": 1,
    "external_id": "CRM-CUST-001",
    "source": "crm",
    "name": "Alice Johnson",
    "email": "alice.johnson@acmecorp.com",
    "company": "Acme Corporation",
    "address_line1": "123 Commerce Blvd",
    "city": "New York",
    "state": "NY",
    "country": "US",
    "postal_code": "10001",
    "status": "active",
    "created_at": "2024-01-10T10:00:00",
    "updated_at": "2024-01-10T10:00:00"
  }
]
```

### `GET /customers/{id}`
**Response 200**: Single `CustomerResponse`
**Response 404**: `{"detail": "Customer 99 not found"}`

---

## Orders

### `GET /orders`
**Query params**: `source`, `status`, `customer_id`, `skip`, `limit`

### `GET /orders/{id}`

---

## Shipments

### `GET /shipments`
**Query params**: `source`, `status`, `order_id`, `skip`, `limit`

### `GET /shipments/{id}`

---

## Integration Jobs

### `GET /integration-jobs`
**Query params**: `job_type` (crm_sync|vendor_sync|full_sync), `status`, `skip`, `limit`

Returns jobs newest-first.

**Response 200 — Array of SyncJobResponse**:
```json
[
  {
    "id": 3,
    "correlation_id": "abc123-...",
    "job_type": "crm_sync",
    "status": "success",
    "triggered_by": "api",
    "started_at": "2024-03-01T10:00:00",
    "completed_at": "2024-03-01T10:00:02",
    "records_processed": 11,
    "records_inserted": 5,
    "records_updated": 5,
    "records_failed": 1,
    "error_message": null,
    "created_at": "2024-03-01T10:00:00"
  }
]
```

### `GET /integration-jobs/{id}`

---

## Failed Records

### `GET /failed-records`
**Query params**: `source`, `status` (pending_retry|retrying|resolved|abandoned), `record_type`, `skip`, `limit`

**Response 200 — Array of FailedRecordResponse**:
```json
[
  {
    "id": 1,
    "sync_job_id": 2,
    "source": "vendor",
    "record_type": "order",
    "external_id": null,
    "raw_data": "<Order><OrderId></OrderId>...</Order>",
    "error_message": "Missing or empty <OrderId>",
    "retry_count": 0,
    "status": "pending_retry",
    "last_retried_at": null,
    "created_at": "2024-03-01T10:00:01",
    "updated_at": "2024-03-01T10:00:01"
  }
]
```

### `POST /failed-records/{id}/retry`
Re-process a failed record.

**Response 200 — RetryResponse**:
```json
{
  "record_id": 1,
  "status": "resolved",
  "retry_count": 1,
  "message": "Record successfully re-processed"
}
```

**Response 404**: Record not found
**Response 409**: Max retry count exceeded (record is abandoned)

---

## Error Responses

All errors follow:
```json
{
  "error": "error_type",
  "message": "Human-readable description",
  "request_id": "a1b2c3d4"
}
```

| HTTP Code | Error Type | Trigger |
|-----------|-----------|---------|
| 404 | Not found | Resource ID doesn't exist |
| 409 | Conflict | Max retries exceeded |
| 422 | Validation error | Invalid query parameter |
| 502 | integration_error | External provider unreachable |
| 500 | internal_server_error | Unexpected exception |

## Response Headers

| Header | Description |
|--------|-------------|
| `X-Request-ID` | 8-char unique ID for each request, useful for log correlation |
