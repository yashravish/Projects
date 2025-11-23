# Database Schema

The database is normalized and implemented in SQL Server via EF Core Code-First migrations.

## Core Entities

### Products & Inventory
*   **Products:** Master product list. (Unique Index: `Sku`)
*   **Batches:** Manufacturing batches linked to Products. (Unique Index: `ProductId` + `BatchNumber`)
*   **Warehouses:** Physical storage facilities. (Unique Index: `Code`)
*   **InventoryLocations:** Bin/shelf locations within a warehouse. (Unique Index: `WarehouseId` + `Code`)
*   **InventoryBalances:** Current quantity snapshot per Product/Warehouse. (Unique Index: `WarehouseId` + `ProductId`)
*   **InventoryMovements:** Immutable ledger of all stock changes (Receipts, Picks, Adjustments).

### Orders
*   **CustomerOrders:** Header information.
*   **OrderLines:** Line items linked to Orders and Products.
    *   *AllocatedWarehouseId:* Tracks which warehouse is fulfilling this line.

### Integration & Support
*   **IntegrationJobs:** Tracks background sync sessions.
*   **IntegrationLogs:** Detailed execution logs linked to Jobs.
*   **ConfigSettings:** Key-Value store for general app settings.

### Automation
*   **ReorderThresholds:** Configuration for min/max stock levels per warehouse.
*   **ReplenishmentSuggestions:** Generated alerts when stock is low.

## Key Relationships
*   `One-to-Many`: Warehouse -> InventoryLocations
*   `One-to-Many`: Product -> Batches
*   `One-to-Many`: CustomerOrder -> OrderLines
*   `Many-to-One`: InventoryMovement -> Product, Warehouse, FromLocation, ToLocation

## SQL Objects (Reporting)
*   `v_SalesByProductAndRegion` (View): Aggregates sales data.
*   `v_InventoryAging` (View): Calculates batch expiry timelines.
*   `usp_CalculateFillRate` (Stored Procedure): Computes order fulfillment metrics.

