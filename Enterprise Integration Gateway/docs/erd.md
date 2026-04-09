# Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE INTEGRATION GATEWAY                    │
│                         Database Schema (ERD)                        │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌────────────────────────────────────────┐
│    customers     │         │              orders                    │
├──────────────────┤         ├────────────────────────────────────────┤
│ PK id            │◄────────│ PK id                                  │
│    external_id   │  1:many │    external_id   (UNIQUE, idx)         │
│    source        │         │ FK customer_id   → customers.id        │
│    name          │         │    source                              │
│    email         │         │    order_number  (idx)                 │
│    phone         │         │    status        (idx composite)       │
│    company       │         │    total_amount                        │
│    address_line1 │         │    currency                            │
│    address_line2 │         │    order_date                          │
│    city          │         │    notes                               │
│    state         │         │    raw_data      (JSONB)               │
│    country       │         │    created_at                          │
│    postal_code   │         │    updated_at                          │
│    status        │         └────────────────────────────────────────┘
│    raw_data(JSONB│                         │
│    created_at    │                         │ 1:many
│    updated_at    │                         ▼
└──────────────────┘         ┌────────────────────────────────────────┐
                             │            shipments                   │
                             ├────────────────────────────────────────┤
                             │ PK id                                  │
                             │    external_id   (UNIQUE, idx)         │
                             │ FK order_id      → orders.id           │
                             │    source                              │
                             │    tracking_number (idx)               │
                             │    carrier                             │
                             │    status        (idx composite)       │
                             │    estimated_delivery                  │
                             │    actual_delivery                     │
                             │    weight_kg                           │
                             │    raw_data      (JSONB)               │
                             │    created_at                          │
                             │    updated_at                          │
                             └────────────────────────────────────────┘


┌──────────────────────────────────────────────┐
│                  sync_jobs                   │
├──────────────────────────────────────────────┤
│ PK id                                        │
│    correlation_id  (UNIQUE, idx)             │
│    job_type        crm_sync|vendor_sync|full │
│    status          pending|running|success   │
│                    partial_success|failed    │
│    triggered_by    api|scheduler|retry       │
│    started_at                                │
│    completed_at                              │
│    records_processed                         │
│    records_inserted                          │
│    records_updated                           │
│    records_failed                            │
│    error_message                             │
│    job_metadata    (JSONB)                   │
│    created_at      (idx)                     │
└──────────────────────────────────────────────┘
         │
         │ 1:many
         ▼
┌──────────────────────────────────────────────┐
│               failed_records                 │
├──────────────────────────────────────────────┤
│ PK id                                        │
│ FK sync_job_id → sync_jobs.id                │
│    source        crm|vendor                  │
│    record_type   customer|order|shipment     │
│    external_id   (idx)                       │
│    raw_data      (Text — original payload)   │
│    error_message                             │
│    retry_count                               │
│    status        pending_retry|retrying      │
│                  resolved|abandoned          │
│    last_retried_at                           │
│    created_at                                │
│    updated_at                                │
└──────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `external_id` UNIQUE per table | Enables idempotent upserts across repeated syncs |
| `raw_data JSONB` on entity tables | Preserves original payload for audit, debugging, and schema evolution |
| `raw_data TEXT` on `failed_records` | Stores XML strings and malformed payloads that are not valid JSON |
| `source` column on all entities | Allows querying by origin system without JOINs |
| Nullable `customer_id` on orders | Vendor orders may arrive before the customer record is synced |
| Nullable `order_id` on shipments | Same cross-source linking flexibility |
| `correlation_id` UUID on `sync_jobs` | Distributed tracing identifier, safe to log |
| Composite indexes on `(source, status)` | Supports filtered list queries efficiently |
