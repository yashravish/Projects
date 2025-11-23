# Operations Hub

## Overview

**Operations Hub** is a portfolio-grade web application inspired by the operations of a modern vitamin and dietary supplements manufacturer. It is designed to demonstrate enterprise application development skills, serving as a strategic companion to Enterprise Resource Planning (ERP) and Warehouse Management Systems (WMS).

This project simulates real-world workflows for product batch management, inventory control, customer order fulfillment, and operational analytics. It demonstrates a modern, layered architecture using **.NET 8**, **Entity Framework Core**, and **SQL Server**, emphasizing clean separation of concerns, robust error handling, and automated integration processes.

> **Disclaimer:** This is a demonstration project and is not affiliated with, endorsed by, or a proprietary product of any specific entity named "21st Century HealthCare". Any resemblance to actual proprietary systems is purely inspirational or coincidental.

## Key Features

### 1. Domain Management
*   **Product Master:** Centralized management of SKU, categories, and lifecycle status.
*   **Batch Tracking:** Comprehensive tracking of manufacturing batches, including expiration dates and initial/remaining quantities.
*   **Multi-Warehouse Inventory:** Support for multiple facilities (e.g., US, EU, APAC) with granular bin/shelf location tracking.

### 2. Order Fulfillment Engine
*   **Intelligent Allocation:** Automated logic to reserve stock for customer orders based on warehouse region and availability.
*   **Lifecycle Management:** Full state tracking from "Pending" to "Allocated" to "Shipped".
*   **Backorder Handling:** Automatic handling of insufficient stock scenarios.

### 3. ERP Integration
*   **Sync Agent:** A background worker (IHostedService) that runs periodically to synchronize Product and Order data from the upstream ERP system.
*   **Resilience:** Implements logging and error handling for external API calls.
*   **Mock ERP:** Includes a simulated ERP API endpoint within the solution for standalone demonstration and testing.

### 4. Operational Reporting
*   **Sales Analytics:** Visualization of sales performance by region and product category.
*   **Inventory Aging:** Analysis of batch expiry timelines to prevent spoilage.
*   **Fill Rate Metrics:** Performance tracking of order fulfillment efficiency.

### 5. Automation & Configuration
*   **Replenishment Triggers:** Configurable reorder thresholds per product and warehouse.
*   **Automated Suggestions:** System-generated replenishment tasks when stock falls below defined limits.

## Technical Architecture

The solution follows a strict **Layered Architecture** to ensure maintainability and testability.

1.  **Presentation Layer (OperationsHub.Api)**
    *   **API Controllers:** Expose RESTful endpoints for data access and operations.
    *   **MVC Views:** Server-side rendered Razor pages for internal dashboards and reports.
2.  **Service Layer:** Encapsulates business logic, validation, and workflow orchestration (e.g., `OrderService`, `InventoryService`).
3.  **Data Access Layer:** Implements the Repository Pattern over Entity Framework Core to abstract database interactions.
4.  **Integration Layer:** Specialized HTTP clients for external system communication.
5.  **Database:** SQL Server (Production) or SQLite (Development/Test) managing relational data with appropriate constraints and indexes.

## Getting Started

### Prerequisites
*   .NET 8.0 SDK
*   SQL Server (or use the configured SQLite fallback for development)
*   Visual Studio 2022 or Visual Studio Code

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/operations-hub.git
    cd operations-hub
    ```

2.  **Database Setup**
    The project uses Entity Framework Core Migrations.
    
    *Modify `appsettings.json` if you wish to switch between SQL Server and SQLite connection strings.*

    Run the following command to apply migrations and seed initial data:
    ```bash
    dotnet ef database update --project OperationsHub.Api
    ```

3.  **Running the Application**
    ```bash
    dotnet run --project OperationsHub.Api
    ```
    
    The application will launch on `https://localhost:5051` (or similar, check console output).

### Accessing the Interfaces

*   **Swagger API Documentation:** `/swagger`
*   **Support Dashboard:** `/Support/Jobs`
*   **Operational Reports:** 
    *   Sales: `/Reports/Sales`
    *   Inventory Aging: `/Reports/InventoryAging`
    *   Fill Rate: `/Reports/FillRate`

## Testing Strategies

The solution includes a comprehensive test suite located in `OperationsHub.Tests`.

### Unit Tests
Focus on critical business logic within the Service layer, utilizing **Moq** for dependency isolation and **FluentAssertions** for readability. Key areas covered:
*   Order allocation logic and warehouse selection.
*   Inventory movement calculations.
*   Replenishment trigger rules.

### Integration Tests
End-to-end tests using `WebApplicationFactory` and an **In-Memory Database** to verify API endpoints, middleware configuration, and data persistence flows.

**Running Tests:**
```bash
dotnet test
```

## Documentation

Detailed documentation is available in the `OperationsHub.Api/docs/` directory:
*   **Architecture.md:** Deep dive into the system design and decision log.
*   **DB-Schema.md:** Entity relationships, constraints, and SQL objects.
*   **ERP-Integration.md:** Protocol for data synchronization and failure recovery.
*   **Operations-Playbook.md:** Guide for support engineers troubleshooting the system.
*   **TEST_PLAN.md:** Comprehensive testing strategy and scenarios.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
