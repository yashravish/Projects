# ERP Integration Strategy

This application acts as a downstream consumer of the corporate ERP system. It maintains a local cache of Products and Orders to ensure high performance and offline availability for warehouse operations.

## Mock ERP API
For this project, a Mock ERP is hosted within the same API application under `/mock-erp`.
*   `GET /mock-erp/products`: Returns a fixed list of product definitions.
*   `GET /mock-erp/orders`: Returns a list of open customer orders.
*   `POST /mock-erp/orders/{id}/ship-confirm`: Endpoint to notify ERP of shipment.

## Sync Process
Managed by `ErpSyncJob` (BackgroundService) running every **60 minutes** (configurable).

1.  **Product Sync:**
    *   Fetches all products.
    *   Matches by SKU.
    *   Inserts new records or updates existing names/categories.

2.  **Order Sync:**
    *   Fetches open orders.
    *   Matches by `ExternalOrderNumber`.
    *   Inserts **new** orders only (idempotent).
    *   Validates Product SKUs; skips lines with unknown products (logs warning).

## Failure Handling
*   **Logging:** All activities are logged to `IntegrationLogs` with a `JobId`.
*   **Exceptions:** Caught at the job level; the job is marked as `Failed`.
*   **Retry/Replay:** Failed jobs can be manually re-triggered via the **Support UI** (`/Support/Jobs`). The Replay action creates a new job record with the suffix `_REPLAY` and re-executes the logic.

