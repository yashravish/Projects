# Architecture Overview

21stCentury.OperationsHub is built using a **Layered Architecture** on the Microsoft .NET 8 stack. It is designed to support warehouse operations, order management, and ERP integration.

## Technology Stack
- **Framework:** ASP.NET Core 8 (Web API + MVC)
- **Database:** SQL Server (via Entity Framework Core)
- **UI:** Razor Views + Bootstrap + Chart.js
- **Integration:** HttpClient for ERP communication
- **Background Processing:** IHostedService (Worker Services)

## Logical Layers

1.  **Presentation Layer (OperationsHub.Api)**
    *   **Controllers (API):** Expose RESTful endpoints for frontend and external consumers.
    *   **Controllers (MVC):** Serve server-side rendered Razor views for reports and support pages.
    *   **Views:** HTML/Razor templates.

2.  **Application/Service Layer**
    *   Contains business logic and orchestration.
    *   `ProductService`: Validations, lifecycle management.
    *   `OrderService`: Allocation strategies, shipping logic.
    *   `InventoryService`: Inventory movements, replenishment rules.
    *   `IntegrationService`: Sync orchestration.

3.  **Domain Layer (Models/Entities)**
    *   Pure POCO classes representing the database schema and core business entities (e.g., `Product`, `Batch`, `CustomerOrder`).

4.  **Data Access Layer**
    *   `ApplicationDbContext`: EF Core context managing DB sessions.
    *   `IRepository<T>` / `EfRepository<T>`: Generic abstraction for data access to keep services decoupled from EF specifics.

5.  **Integration Layer**
    *   `ErpClient`: Typed HTTP client to communicate with the external ERP system.

## Key Modules

*   **Products & Batches:** Master data management.
*   **Warehousing:** Multi-warehouse inventory tracking with bin locations.
*   **Orders:** Customer order lifecycle (Pending -> Allocated -> Shipped).
*   **ERP Sync:** Background job that periodically pulls updates from the ERP.
*   **Reporting:** SQL-backed analytics for fill rates and sales.
*   **Configuration:** Dynamic reorder thresholds and automated replenishment suggestions.

