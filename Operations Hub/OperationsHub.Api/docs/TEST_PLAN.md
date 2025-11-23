# Test Plan: 21st Century Operations Hub

This document outlines the testing strategy for the Operations Hub application. It covers the critical paths, business logic, and integration points that require verification.

## 1. Critical Business Logic (Unit Test Candidates)

These areas contain complex logic and should be covered by automated unit tests using mocks (e.g., xUnit + Moq).

### OrderService.AllocateOrderAsync
*   **Scenario:** Order quantity <= Inventory.
    *   *Expectation:* LineStatus becomes "Allocated", InventoryBalance decreases, InventoryMovement created ("Pick").
*   **Scenario:** Order quantity > Inventory.
    *   *Expectation:* LineStatus becomes "Backordered", Order Status becomes "Pending" or "Partially Allocated".
*   **Scenario:** Multiple warehouses.
    *   *Expectation:* Logic prefers warehouse in same Region first, then highest quantity.

### InventoryService.ApplyInventoryMovementAsync
*   **Scenario:** Receipt.
    *   *Expectation:* Balance increases, Movement recorded.
*   **Scenario:** Pick (Manual).
    *   *Expectation:* Balance decreases.
*   **Scenario:** Reorder Trigger.
    *   *Expectation:* If new quantity <= Threshold, a `ReplenishmentSuggestion` is created (if none Open exists).

### IntegrationService.SyncOrdersFromErpAsync
*   **Scenario:** New Order.
    *   *Expectation:* Order created in DB.
*   **Scenario:** Duplicate Order (same ExternalOrderNumber).
    *   *Expectation:* Skipped (idempotency).
*   **Scenario:** Unknown Product SKU.
    *   *Expectation:* Line skipped, Warning logged.

---

## 2. API Integration Scenarios (.http / Postman)

Use `OperationsHub.Api.http` to execute these requests against the running API.

### Products & Inventory
1.  **GET /api/products:** Verify seed data exists.
2.  **GET /api/warehouses/{id}/inventory:** Check initial balances.
3.  **POST /api/inventory/movements:**
    *   Add stock (Receipt).
    *   Verify balance update.

### Order Lifecycle
1.  **POST /api/orders:** Create a standard order.
2.  **POST /api/orders/{id}/allocate:** Trigger allocation.
    *   Verify status changes to "Allocated".
3.  **POST /api/orders/{id}/ship:** Trigger shipment.
    *   Verify status changes to "Shipped".

### ERP Integration
1.  **GET /mock-erp/products:** Ensure mock endpoint is up.
2.  **GET /mock-erp/orders:** Ensure mock endpoint is up.
3.  *(Background Job runs automatically, verify via Support UI)*.

---

## 3. User Acceptance Tests (Manual / UI)

### Scenario A: Operations Dashboard & Reporting
1.  **Sales Report:**
    *   Navigate to `/Reports/Sales`.
    *   Filter by date range.
    *   Verify the Pie Chart renders and Table shows data (requires Shipped orders).
2.  **Inventory Aging:**
    *   Navigate to `/Reports/InventoryAging`.
    *   Verify batches are listed.
    *   Check conditional formatting (Yellow/Red) for near-expiry items.

### Scenario B: Support & Troubleshooting
1.  **View Jobs:**
    *   Navigate to `/Support/Jobs`.
    *   Verify "ERP_SYNC" jobs are appearing (every hour, or manually triggered).
2.  **Job Details:**
    *   Click a Job ID.
    *   Verify "Info" logs regarding products/orders fetched.
3.  **Replay:**
    *   Find a job (or wait for a failure).
    *   Click "Replay".
    *   Verify a new job appears with type `..._REPLAY`.

### Scenario C: Configuration & Automation
1.  **Set Threshold:**
    *   Use API to set MinQuantity = 100 for a Product/Warehouse.
2.  **Trigger Low Stock:**
    *   Use API to "Pick" inventory down to 90.
3.  **Verify Suggestion:**
    *   Call `GET /api/replenishment-suggestions`.
    *   Verify a new suggestion exists for that product.
4.  **Manage Suggestion:**
    *   Call `POST .../status` to mark it "Reviewed".

---

## 4. Database Verification (SQL)

Run these queries to verify data integrity:

```sql
-- Check for orphaned inventory
SELECT * FROM InventoryBalances WHERE ProductId NOT IN (SELECT Id FROM Products);

-- Check Allocation Integrity
SELECT * FROM OrderLines WHERE LineStatus = 'Allocated' AND AllocatedWarehouseId IS NULL;
-- (Should return 0 rows)

-- Verify History
SELECT Count(*) FROM InventoryMovements;
```

