# Data Flow — Sequence Diagrams

## 1. CRM Sync Flow

```
Client (API / Scheduler)
        │
        │  POST /api/v1/sync/crm  (or scheduled trigger)
        │                          ↓ rate limiter check (Redis)
        ▼
┌───────────────┐
│  sync_service │  1. Create SyncJob (status=running, correlation_id=UUID)
│               │  → Publish SyncStartedEvent to Kafka
│               │
│               │  2. GET /mock/crm/customers  ─────────►  ┌───────────────────┐
│               │  ◄── JSON [{customerId, fullName,...}] ──  │ Mock CRM Service  │
│               │                                            │ (port 8001)       │
│               │  3. GET /mock/crm/orders  ───────────────► │                   │
│               │  ◄── JSON [{orderId, customerId,...}] ───  └───────────────────┘
│               │
│               │  4. For each customer:
│               │     transform_crm_customer(raw) → CustomerCreate
│               │     upsert_customer(db, schema) → (Customer, created)
│               │     On error: create FailedRecord
│               │               → Publish RecordFailedEvent to Kafka
│               │
│               │  5. For each order:
│               │     resolve customer FK via external_id → customer.id
│               │     transform_crm_order(raw, customer_id) → OrderCreate
│               │     upsert_order(db, schema)
│               │     On error: create FailedRecord
│               │               → Publish RecordFailedEvent to Kafka
│               │
│               │  6. Update SyncJob (status, counts)
│               │  → Publish SyncCompletedEvent to Kafka
│               │  → Invalidate Redis cache (customers, orders, metrics)
└───────────────┘
        │
        │  SyncResult {job_id, status, inserted, updated, failed}
        ▼
     Client
```

## 2. Vendor XML Sync Flow

```
Client (API / Scheduler)
        │
        │  POST /api/v1/sync/vendor
        │                            ↓ rate limiter check (Redis)
        ▼
┌───────────────┐
│  sync_service │
│               │  1. Create SyncJob
│               │  → Publish SyncStartedEvent to Kafka
│               │
│               │  2. GET /mock/vendor/orders  ──────────►  ┌───────────────────┐
│               │  ◄── XML <OrderFeed>...</OrderFeed>  ───  │ Mock Vendor XML   │
│               │                                            │ Service (port 8001│
│               │  3. GET /mock/vendor/shipments  ─────────► │                   │
│               │  ◄── XML <ShipmentFeed>...</ShipmentFeed>  └───────────────────┘
│               │
│               │  4. parse_vendor_orders(xml_string)
│               │     → (valid_orders[], malformed[])
│               │
│               │  5. malformed → FailedRecord (source=vendor, type=order)
│               │                → Publish RecordFailedEvent to Kafka
│               │
│               │  6. For each valid order:
│               │     transform_vendor_order(parsed_dict) → OrderCreate
│               │     upsert_order(db, schema)
│               │     track order_id_map[vendor_id] = internal_id
│               │
│               │  7. parse_vendor_shipments(xml_string)
│               │     → (valid_shipments[], malformed[])
│               │
│               │  8. For each valid shipment:
│               │     resolve order FK via vendor_order_id
│               │     transform_vendor_shipment(parsed_dict, order_id)
│               │     upsert_shipment(db, schema)
│               │
│               │  9. Update SyncJob
│               │  → Publish SyncCompletedEvent to Kafka
│               │  → Invalidate Redis cache (orders, shipments, metrics)
└───────────────┘
        │
        │  SyncResult
        ▼
     Client
```

## 3. Failed Record Retry Flow

```
Admin / Operator
        │
        │  GET /api/v1/failed-records?status=pending_retry
        │  ◄── [{id: 12, source: "vendor", record_type: "order", raw_data: "..."}]
        │
        │  POST /api/v1/failed-records/12/retry
        ▼
┌─────────────────────────────┐
│  failed_record_service      │
│  retry_failed_record(db,12) │
│                             │
│  1. Load FailedRecord #12   │
│  2. Check retry_count < MAX │  → if exceeded: status=abandoned, raise RetryExhaustedError
│  3. Set status=retrying     │
│  4. Increment retry_count   │
│                             │
│  5. Re-run transformation:  │
│     source=vendor, type=order
│     → parse_vendor_orders(raw_data)
│     → transform_vendor_order(parsed)
│     → upsert_order(db, schema)
│                             │
│  6a. Success:               │
│      status = resolved      │
│      return RetryResponse   │
│                             │
│  6b. Failure:               │
│      status = pending_retry │
│      update error_message   │
│      return RetryResponse   │
└─────────────────────────────┘
        │
        │  RetryResponse {record_id, status, retry_count, message}
        ▼
  Admin / Operator
```

## 4. Full Sync Flow

```
POST /api/v1/sync/all
        │           ↓ rate limiter check (Redis)
        ▼
  execute_full_sync()
        │       → Publish SyncStartedEvent (full_sync)
        │
        ├─► execute_crm_sync()   ─► SyncJob (type=crm_sync) + Kafka events
        │
        ├─► execute_vendor_sync() ─► SyncJob (type=vendor_sync) + Kafka events
        │
        └─► Aggregate results
            SyncJob (type=full_sync, rolled-up counts)
            → Publish SyncCompletedEvent (full_sync)
            → Invalidate all Redis caches
            Return SyncResult
```

## 5. Redis Cache Flow (GET Endpoints)

```
Client
  │
  │  GET /api/v1/customers?source=crm&limit=50
  ▼
┌────────────────────────┐
│  @cached decorator     │
│                        │
│  1. Build cache key:   │
│     cache:customers:/api/v1/customers:{param_hash}
│                        │
│  2. Check Redis:       │
│     GET cache_key      │
│     ┌── HIT ──────────────► Return cached JSON
│     │                  │
│     └── MISS           │
│         │              │
│  3. Execute handler:   │
│     query DB → result  │
│                        │
│  4. Store in Redis:    │
│     SETEX key TTL json │
│                        │
│  5. Return result      │
└────────────────────────┘
```

## 6. Kafka Event Flow

```
                      ┌─────────────────────────────────────────────┐
                      │          Kafka Topics                       │
                      │                                             │
  sync_service ──────►│  eig.integration.events                     │
    (producer)        │    • sync.started  {job_id, type, trigger}  │
                      │    • sync.completed {job_id, status, counts}│
                      │    • record.failed {job_id, source, error}  │
                      │                                             │
                      │  eig.inbound.sync.requests                  │──► event_consumer
                      │    • sync.request {sync_type, requested_by} │     (background thread)
                      │                                             │     → triggers sync_service
                      └─────────────────────────────────────────────┘

  External System ─── POST /api/v1/events/publish ───► Kafka ───► Consumer ───► Sync
```

## 7. Rate Limiting Flow (POST /sync/*)

```
Client
  │
  │  POST /api/v1/sync/crm
  ▼
┌────────────────────────────┐
│  RateLimiter (dependency)  │
│                            │
│  1. Key: ratelimit:{path}:{client_ip}
│                            │
│  2. ZREMRANGEBYSCORE       │  ← remove entries outside window
│     key, 0, (now - 60s)   │
│                            │
│  3. ZCARD key              │  ← count requests in window
│                            │
│  4. If count >= RPM limit: │
│     → 429 Too Many Reqs   │
│     → Retry-After header   │
│                            │
│  5. ZADD key {now: now}    │  ← add current request
│     EXPIRE key 70s         │
│                            │
│  6. Allow request ─────────┼──► sync_service.execute_crm_sync()
└────────────────────────────┘
```

## Data Transformation Summary

| Source | Format | Field Style | Example Field | → Internal Field |
|--------|--------|-------------|---------------|-----------------|
| CRM | JSON | camelCase | `customerId` | `external_id` |
| CRM | JSON | camelCase | `fullName` | `name` |
| CRM | JSON | nested | `billingAddress.street` | `address_line1` |
| CRM | JSON | camelCase | `accountStatus` | `status` |
| Vendor | XML | PascalCase | `<OrderId>` | `external_id` (`VND-` prefix) |
| Vendor | XML | PascalCase | `<TotalAmount>` | `total_amount` (Decimal) |
| Vendor | XML | PascalCase | `<WeightKg>` | `weight_kg` (Decimal) |
| Vendor | XML | PascalCase | `<VendorOrderId>` | FK link to `orders.id` |
