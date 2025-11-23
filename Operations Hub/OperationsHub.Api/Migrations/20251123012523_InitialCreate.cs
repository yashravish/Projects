using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace OperationsHub.Api.Migrations
{
    /// <inheritdoc />
    public partial class InitialCreate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "ConfigSettings",
                columns: table => new
                {
                    Key = table.Column<string>(type: "TEXT", maxLength: 100, nullable: false),
                    Value = table.Column<string>(type: "TEXT", maxLength: 500, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ConfigSettings", x => x.Key);
                });

            migrationBuilder.CreateTable(
                name: "CustomerOrders",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    ExternalOrderNumber = table.Column<string>(type: "TEXT", maxLength: 100, nullable: true),
                    CustomerName = table.Column<string>(type: "TEXT", maxLength: 200, nullable: false),
                    Region = table.Column<string>(type: "TEXT", maxLength: 50, nullable: true),
                    OrderDate = table.Column<DateTime>(type: "TEXT", nullable: false),
                    Status = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "TEXT", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_CustomerOrders", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "IntegrationJobs",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    JobType = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    StartedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    FinishedAt = table.Column<DateTime>(type: "TEXT", nullable: true),
                    Status = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    Details = table.Column<string>(type: "TEXT", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_IntegrationJobs", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "Products",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Sku = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    Name = table.Column<string>(type: "TEXT", maxLength: 200, nullable: false),
                    Category = table.Column<string>(type: "TEXT", maxLength: 100, nullable: true),
                    IsActive = table.Column<bool>(type: "INTEGER", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "TEXT", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Products", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "Warehouses",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Name = table.Column<string>(type: "TEXT", maxLength: 100, nullable: false),
                    Code = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    Region = table.Column<string>(type: "TEXT", maxLength: 50, nullable: true),
                    IsActive = table.Column<bool>(type: "INTEGER", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "TEXT", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Warehouses", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "IntegrationLogs",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    JobId = table.Column<int>(type: "INTEGER", nullable: true),
                    Level = table.Column<string>(type: "TEXT", maxLength: 20, nullable: false),
                    Message = table.Column<string>(type: "TEXT", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_IntegrationLogs", x => x.Id);
                    table.ForeignKey(
                        name: "FK_IntegrationLogs_IntegrationJobs_JobId",
                        column: x => x.JobId,
                        principalTable: "IntegrationJobs",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "Batches",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    ProductId = table.Column<int>(type: "INTEGER", nullable: false),
                    BatchNumber = table.Column<string>(type: "TEXT", maxLength: 100, nullable: false),
                    ManufactureDate = table.Column<DateTime>(type: "TEXT", nullable: false),
                    ExpiryDate = table.Column<DateTime>(type: "TEXT", nullable: false),
                    InitialQuantity = table.Column<int>(type: "INTEGER", nullable: false),
                    RemainingQuantity = table.Column<int>(type: "INTEGER", nullable: false),
                    Status = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "TEXT", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Batches", x => x.Id);
                    table.ForeignKey(
                        name: "FK_Batches_Products_ProductId",
                        column: x => x.ProductId,
                        principalTable: "Products",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "InventoryBalances",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    WarehouseId = table.Column<int>(type: "INTEGER", nullable: false),
                    ProductId = table.Column<int>(type: "INTEGER", nullable: false),
                    Quantity = table.Column<int>(type: "INTEGER", nullable: false),
                    LastUpdatedAt = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_InventoryBalances", x => x.Id);
                    table.ForeignKey(
                        name: "FK_InventoryBalances_Products_ProductId",
                        column: x => x.ProductId,
                        principalTable: "Products",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_InventoryBalances_Warehouses_WarehouseId",
                        column: x => x.WarehouseId,
                        principalTable: "Warehouses",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "InventoryLocations",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    WarehouseId = table.Column<int>(type: "INTEGER", nullable: false),
                    Code = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    Description = table.Column<string>(type: "TEXT", maxLength: 200, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_InventoryLocations", x => x.Id);
                    table.ForeignKey(
                        name: "FK_InventoryLocations_Warehouses_WarehouseId",
                        column: x => x.WarehouseId,
                        principalTable: "Warehouses",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "OrderLines",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    OrderId = table.Column<int>(type: "INTEGER", nullable: false),
                    ProductId = table.Column<int>(type: "INTEGER", nullable: false),
                    Quantity = table.Column<int>(type: "INTEGER", nullable: false),
                    AllocatedWarehouseId = table.Column<int>(type: "INTEGER", nullable: true),
                    ShippedQuantity = table.Column<int>(type: "INTEGER", nullable: false),
                    LineStatus = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_OrderLines", x => x.Id);
                    table.ForeignKey(
                        name: "FK_OrderLines_CustomerOrders_OrderId",
                        column: x => x.OrderId,
                        principalTable: "CustomerOrders",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_OrderLines_Products_ProductId",
                        column: x => x.ProductId,
                        principalTable: "Products",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_OrderLines_Warehouses_AllocatedWarehouseId",
                        column: x => x.AllocatedWarehouseId,
                        principalTable: "Warehouses",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "ReorderThresholds",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    ProductId = table.Column<int>(type: "INTEGER", nullable: false),
                    WarehouseId = table.Column<int>(type: "INTEGER", nullable: false),
                    MinQuantity = table.Column<int>(type: "INTEGER", nullable: false),
                    ReorderQuantity = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ReorderThresholds", x => x.Id);
                    table.ForeignKey(
                        name: "FK_ReorderThresholds_Products_ProductId",
                        column: x => x.ProductId,
                        principalTable: "Products",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_ReorderThresholds_Warehouses_WarehouseId",
                        column: x => x.WarehouseId,
                        principalTable: "Warehouses",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "ReplenishmentSuggestions",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    ProductId = table.Column<int>(type: "INTEGER", nullable: false),
                    WarehouseId = table.Column<int>(type: "INTEGER", nullable: false),
                    CurrentQuantity = table.Column<int>(type: "INTEGER", nullable: false),
                    SuggestedReorderQuantity = table.Column<int>(type: "INTEGER", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    Status = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ReplenishmentSuggestions", x => x.Id);
                    table.ForeignKey(
                        name: "FK_ReplenishmentSuggestions_Products_ProductId",
                        column: x => x.ProductId,
                        principalTable: "Products",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_ReplenishmentSuggestions_Warehouses_WarehouseId",
                        column: x => x.WarehouseId,
                        principalTable: "Warehouses",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "InventoryMovements",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    ProductId = table.Column<int>(type: "INTEGER", nullable: false),
                    FromLocationId = table.Column<int>(type: "INTEGER", nullable: true),
                    ToLocationId = table.Column<int>(type: "INTEGER", nullable: true),
                    WarehouseId = table.Column<int>(type: "INTEGER", nullable: false),
                    Quantity = table.Column<int>(type: "INTEGER", nullable: false),
                    MovementType = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    PerformedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    Reference = table.Column<string>(type: "TEXT", maxLength: 100, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_InventoryMovements", x => x.Id);
                    table.ForeignKey(
                        name: "FK_InventoryMovements_InventoryLocations_FromLocationId",
                        column: x => x.FromLocationId,
                        principalTable: "InventoryLocations",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_InventoryMovements_InventoryLocations_ToLocationId",
                        column: x => x.ToLocationId,
                        principalTable: "InventoryLocations",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_InventoryMovements_Products_ProductId",
                        column: x => x.ProductId,
                        principalTable: "Products",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_InventoryMovements_Warehouses_WarehouseId",
                        column: x => x.WarehouseId,
                        principalTable: "Warehouses",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.InsertData(
                table: "Products",
                columns: new[] { "Id", "Category", "CreatedAt", "IsActive", "Name", "Sku", "UpdatedAt" },
                values: new object[,]
                {
                    { 1, "Vitamin", new DateTime(2024, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc), true, "Daily Multi-Vitamin (Men)", "VIT-MULTI-001", null },
                    { 2, "Vitamin", new DateTime(2024, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc), true, "Vitamin C 500mg", "VIT-C-500", null },
                    { 3, "Herbal Extract", new DateTime(2024, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc), true, "Organic Echinacea Tea", "HERB-ECHIN", null },
                    { 4, "Pet Nutrition", new DateTime(2024, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc), true, "Canine Joint Support Chews", "PET-JOINT", null },
                    { 5, "Sports Nutrition", new DateTime(2024, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc), true, "Whey Protein Isolate - Chocolate", "PRO-WHEY-CHOC", null }
                });

            migrationBuilder.InsertData(
                table: "Warehouses",
                columns: new[] { "Id", "Code", "CreatedAt", "IsActive", "Name", "Region", "UpdatedAt" },
                values: new object[,]
                {
                    { 1, "TEMPE-HQ", new DateTime(2024, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc), true, "Tempe HQ", "US", null },
                    { 2, "EU-HUB", new DateTime(2024, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc), true, "EU Hub", "EU", null },
                    { 3, "APAC-HUB", new DateTime(2024, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc), true, "APAC Hub", "APAC", null }
                });

            migrationBuilder.InsertData(
                table: "Batches",
                columns: new[] { "Id", "BatchNumber", "CreatedAt", "ExpiryDate", "InitialQuantity", "ManufactureDate", "ProductId", "RemainingQuantity", "Status", "UpdatedAt" },
                values: new object[,]
                {
                    { 1, "B2401-001", new DateTime(2024, 1, 16, 0, 0, 0, 0, DateTimeKind.Utc), new DateTime(2026, 1, 15, 0, 0, 0, 0, DateTimeKind.Utc), 5000, new DateTime(2024, 1, 15, 0, 0, 0, 0, DateTimeKind.Utc), 1, 4500, "Open", null },
                    { 2, "B2402-002", new DateTime(2024, 2, 11, 0, 0, 0, 0, DateTimeKind.Utc), new DateTime(2026, 2, 10, 0, 0, 0, 0, DateTimeKind.Utc), 3000, new DateTime(2024, 2, 10, 0, 0, 0, 0, DateTimeKind.Utc), 1, 3000, "Open", null },
                    { 3, "C2312-099", new DateTime(2023, 12, 21, 0, 0, 0, 0, DateTimeKind.Utc), new DateTime(2025, 12, 20, 0, 0, 0, 0, DateTimeKind.Utc), 10000, new DateTime(2023, 12, 20, 0, 0, 0, 0, DateTimeKind.Utc), 2, 8200, "Open", null },
                    { 4, "P2403-FAIL", new DateTime(2024, 3, 2, 0, 0, 0, 0, DateTimeKind.Utc), new DateTime(2025, 3, 1, 0, 0, 0, 0, DateTimeKind.Utc), 500, new DateTime(2024, 3, 1, 0, 0, 0, 0, DateTimeKind.Utc), 4, 500, "Quarantined", null }
                });

            migrationBuilder.InsertData(
                table: "InventoryBalances",
                columns: new[] { "Id", "LastUpdatedAt", "ProductId", "Quantity", "WarehouseId" },
                values: new object[,]
                {
                    { 1, new DateTime(2024, 3, 10, 0, 0, 0, 0, DateTimeKind.Utc), 1, 4500, 1 },
                    { 2, new DateTime(2024, 3, 10, 0, 0, 0, 0, DateTimeKind.Utc), 2, 8000, 1 },
                    { 3, new DateTime(2024, 2, 15, 0, 0, 0, 0, DateTimeKind.Utc), 2, 200, 2 },
                    { 4, new DateTime(2024, 3, 5, 0, 0, 0, 0, DateTimeKind.Utc), 4, 500, 3 }
                });

            migrationBuilder.InsertData(
                table: "InventoryLocations",
                columns: new[] { "Id", "Code", "Description", "WarehouseId" },
                values: new object[,]
                {
                    { 1, "A1-SHELF1-BIN1", "Main Floor, Aisle 1", 1 },
                    { 2, "A1-SHELF1-BIN2", "Main Floor, Aisle 1", 1 },
                    { 3, "EU-RECEIVING", "Receiving Dock", 2 },
                    { 4, "EU-STORAGE-01", "Bulk Storage", 2 },
                    { 5, "APAC-GENERAL", "General Storage", 3 }
                });

            migrationBuilder.CreateIndex(
                name: "IX_Batches_ProductId_BatchNumber",
                table: "Batches",
                columns: new[] { "ProductId", "BatchNumber" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_IntegrationLogs_JobId",
                table: "IntegrationLogs",
                column: "JobId");

            migrationBuilder.CreateIndex(
                name: "IX_InventoryBalances_ProductId",
                table: "InventoryBalances",
                column: "ProductId");

            migrationBuilder.CreateIndex(
                name: "IX_InventoryBalances_WarehouseId_ProductId",
                table: "InventoryBalances",
                columns: new[] { "WarehouseId", "ProductId" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_InventoryLocations_WarehouseId_Code",
                table: "InventoryLocations",
                columns: new[] { "WarehouseId", "Code" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_InventoryMovements_FromLocationId",
                table: "InventoryMovements",
                column: "FromLocationId");

            migrationBuilder.CreateIndex(
                name: "IX_InventoryMovements_ProductId",
                table: "InventoryMovements",
                column: "ProductId");

            migrationBuilder.CreateIndex(
                name: "IX_InventoryMovements_ToLocationId",
                table: "InventoryMovements",
                column: "ToLocationId");

            migrationBuilder.CreateIndex(
                name: "IX_InventoryMovements_WarehouseId",
                table: "InventoryMovements",
                column: "WarehouseId");

            migrationBuilder.CreateIndex(
                name: "IX_OrderLines_AllocatedWarehouseId",
                table: "OrderLines",
                column: "AllocatedWarehouseId");

            migrationBuilder.CreateIndex(
                name: "IX_OrderLines_OrderId",
                table: "OrderLines",
                column: "OrderId");

            migrationBuilder.CreateIndex(
                name: "IX_OrderLines_ProductId",
                table: "OrderLines",
                column: "ProductId");

            migrationBuilder.CreateIndex(
                name: "IX_Products_Sku",
                table: "Products",
                column: "Sku",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_ReorderThresholds_ProductId",
                table: "ReorderThresholds",
                column: "ProductId");

            migrationBuilder.CreateIndex(
                name: "IX_ReorderThresholds_WarehouseId_ProductId",
                table: "ReorderThresholds",
                columns: new[] { "WarehouseId", "ProductId" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_ReplenishmentSuggestions_ProductId",
                table: "ReplenishmentSuggestions",
                column: "ProductId");

            migrationBuilder.CreateIndex(
                name: "IX_ReplenishmentSuggestions_WarehouseId",
                table: "ReplenishmentSuggestions",
                column: "WarehouseId");

            migrationBuilder.CreateIndex(
                name: "IX_Warehouses_Code",
                table: "Warehouses",
                column: "Code",
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Batches");

            migrationBuilder.DropTable(
                name: "ConfigSettings");

            migrationBuilder.DropTable(
                name: "IntegrationLogs");

            migrationBuilder.DropTable(
                name: "InventoryBalances");

            migrationBuilder.DropTable(
                name: "InventoryMovements");

            migrationBuilder.DropTable(
                name: "OrderLines");

            migrationBuilder.DropTable(
                name: "ReorderThresholds");

            migrationBuilder.DropTable(
                name: "ReplenishmentSuggestions");

            migrationBuilder.DropTable(
                name: "IntegrationJobs");

            migrationBuilder.DropTable(
                name: "InventoryLocations");

            migrationBuilder.DropTable(
                name: "CustomerOrders");

            migrationBuilder.DropTable(
                name: "Products");

            migrationBuilder.DropTable(
                name: "Warehouses");
        }
    }
}
