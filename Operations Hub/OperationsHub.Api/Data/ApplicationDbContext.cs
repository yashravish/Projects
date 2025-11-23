using Microsoft.EntityFrameworkCore;
using OperationsHub.Api.Models.Entities;

namespace OperationsHub.Api.Data
{
    public class ApplicationDbContext : DbContext
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options)
        {
        }

        public DbSet<Product> Products { get; set; }
        public DbSet<Batch> Batches { get; set; }
        public DbSet<Warehouse> Warehouses { get; set; }
        public DbSet<InventoryLocation> InventoryLocations { get; set; }
        public DbSet<InventoryBalance> InventoryBalances { get; set; }
        public DbSet<InventoryMovement> InventoryMovements { get; set; }
        public DbSet<CustomerOrder> CustomerOrders { get; set; }
        public DbSet<OrderLine> OrderLines { get; set; }
        public DbSet<ConfigSetting> ConfigSettings { get; set; }
        public DbSet<IntegrationJob> IntegrationJobs { get; set; }
        public DbSet<IntegrationLog> IntegrationLogs { get; set; }
        public DbSet<ReplenishmentSuggestion> ReplenishmentSuggestions { get; set; }
        public DbSet<ReorderThreshold> ReorderThresholds { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // Product
            modelBuilder.Entity<Product>()
                .HasIndex(p => p.Sku)
                .IsUnique();

            // Batch
            modelBuilder.Entity<Batch>()
                .HasIndex(b => new { b.ProductId, b.BatchNumber })
                .IsUnique();

            // Warehouse
            modelBuilder.Entity<Warehouse>()
                .HasIndex(w => w.Code)
                .IsUnique();

            // InventoryLocation
            modelBuilder.Entity<InventoryLocation>()
                .HasIndex(l => new { l.WarehouseId, l.Code })
                .IsUnique();

            // InventoryBalance
            modelBuilder.Entity<InventoryBalance>()
                .HasIndex(ib => new { ib.WarehouseId, ib.ProductId })
                .IsUnique();
            
            // InventoryMovement: Restrict delete behaviors to prevent history loss
            modelBuilder.Entity<InventoryMovement>()
                .HasOne(m => m.FromLocation)
                .WithMany(l => l.MovementsFrom)
                .HasForeignKey(m => m.FromLocationId)
                .OnDelete(DeleteBehavior.Restrict);

            modelBuilder.Entity<InventoryMovement>()
                .HasOne(m => m.ToLocation)
                .WithMany(l => l.MovementsTo)
                .HasForeignKey(m => m.ToLocationId)
                .OnDelete(DeleteBehavior.Restrict);

             modelBuilder.Entity<InventoryMovement>()
                .HasOne(m => m.Product)
                .WithMany(p => p.InventoryMovements)
                .HasForeignKey(m => m.ProductId)
                .OnDelete(DeleteBehavior.Restrict);
            
            // OrderLine relationships
            modelBuilder.Entity<OrderLine>()
                .HasOne(ol => ol.Order)
                .WithMany(o => o.OrderLines)
                .HasForeignKey(ol => ol.OrderId)
                .OnDelete(DeleteBehavior.Cascade); // Deleting order deletes lines

             modelBuilder.Entity<OrderLine>()
                .HasOne(ol => ol.AllocatedWarehouse)
                .WithMany(w => w.OrderLines)
                .HasForeignKey(ol => ol.AllocatedWarehouseId)
                .OnDelete(DeleteBehavior.Restrict);

            // IntegrationLog relationships
            modelBuilder.Entity<IntegrationLog>()
                .HasOne(l => l.Job)
                .WithMany(j => j.Logs)
                .HasForeignKey(l => l.JobId)
                .OnDelete(DeleteBehavior.Cascade); // Deleting job deletes logs

            // ReorderThreshold
            modelBuilder.Entity<ReorderThreshold>()
                .HasIndex(rt => new { rt.WarehouseId, rt.ProductId })
                .IsUnique();

            SeedData(modelBuilder);
        }

        private void SeedData(ModelBuilder modelBuilder)
        {
            // 1. Products
            modelBuilder.Entity<Product>().HasData(
                new Product { Id = 1, Sku = "VIT-MULTI-001", Name = "Daily Multi-Vitamin (Men)", Category = "Vitamin", IsActive = true, CreatedAt = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc) },
                new Product { Id = 2, Sku = "VIT-C-500", Name = "Vitamin C 500mg", Category = "Vitamin", IsActive = true, CreatedAt = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc) },
                new Product { Id = 3, Sku = "HERB-ECHIN", Name = "Organic Echinacea Tea", Category = "Herbal Extract", IsActive = true, CreatedAt = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc) },
                new Product { Id = 4, Sku = "PET-JOINT", Name = "Canine Joint Support Chews", Category = "Pet Nutrition", IsActive = true, CreatedAt = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc) },
                new Product { Id = 5, Sku = "PRO-WHEY-CHOC", Name = "Whey Protein Isolate - Chocolate", Category = "Sports Nutrition", IsActive = true, CreatedAt = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc) }
            );

            // 2. Warehouses
            modelBuilder.Entity<Warehouse>().HasData(
                new Warehouse { Id = 1, Name = "Tempe HQ", Code = "TEMPE-HQ", Region = "US", IsActive = true, CreatedAt = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc) },
                new Warehouse { Id = 2, Name = "EU Hub", Code = "EU-HUB", Region = "EU", IsActive = true, CreatedAt = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc) },
                new Warehouse { Id = 3, Name = "APAC Hub", Code = "APAC-HUB", Region = "APAC", IsActive = true, CreatedAt = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc) }
            );

            // 3. Inventory Locations
            modelBuilder.Entity<InventoryLocation>().HasData(
                new InventoryLocation { Id = 1, WarehouseId = 1, Code = "A1-SHELF1-BIN1", Description = "Main Floor, Aisle 1" },
                new InventoryLocation { Id = 2, WarehouseId = 1, Code = "A1-SHELF1-BIN2", Description = "Main Floor, Aisle 1" },
                new InventoryLocation { Id = 3, WarehouseId = 2, Code = "EU-RECEIVING", Description = "Receiving Dock" },
                new InventoryLocation { Id = 4, WarehouseId = 2, Code = "EU-STORAGE-01", Description = "Bulk Storage" },
                new InventoryLocation { Id = 5, WarehouseId = 3, Code = "APAC-GENERAL", Description = "General Storage" }
            );

            // 4. Batches
            modelBuilder.Entity<Batch>().HasData(
                // Vitamin Multi Batches
                new Batch { Id = 1, ProductId = 1, BatchNumber = "B2401-001", ManufactureDate = new DateTime(2024, 1, 15, 0, 0, 0, DateTimeKind.Utc), ExpiryDate = new DateTime(2026, 1, 15, 0, 0, 0, DateTimeKind.Utc), InitialQuantity = 5000, RemainingQuantity = 4500, Status = "Open", CreatedAt = new DateTime(2024, 1, 16, 0, 0, 0, DateTimeKind.Utc) },
                new Batch { Id = 2, ProductId = 1, BatchNumber = "B2402-002", ManufactureDate = new DateTime(2024, 2, 10, 0, 0, 0, DateTimeKind.Utc), ExpiryDate = new DateTime(2026, 2, 10, 0, 0, 0, DateTimeKind.Utc), InitialQuantity = 3000, RemainingQuantity = 3000, Status = "Open", CreatedAt = new DateTime(2024, 2, 11, 0, 0, 0, DateTimeKind.Utc) },
                
                // Vitamin C Batches
                new Batch { Id = 3, ProductId = 2, BatchNumber = "C2312-099", ManufactureDate = new DateTime(2023, 12, 20, 0, 0, 0, DateTimeKind.Utc), ExpiryDate = new DateTime(2025, 12, 20, 0, 0, 0, DateTimeKind.Utc), InitialQuantity = 10000, RemainingQuantity = 8200, Status = "Open", CreatedAt = new DateTime(2023, 12, 21, 0, 0, 0, DateTimeKind.Utc) },
                
                // Pet Joint Support (Quarantined example)
                new Batch { Id = 4, ProductId = 4, BatchNumber = "P2403-FAIL", ManufactureDate = new DateTime(2024, 3, 1, 0, 0, 0, DateTimeKind.Utc), ExpiryDate = new DateTime(2025, 3, 1, 0, 0, 0, DateTimeKind.Utc), InitialQuantity = 500, RemainingQuantity = 500, Status = "Quarantined", CreatedAt = new DateTime(2024, 3, 2, 0, 0, 0, DateTimeKind.Utc) }
            );

            // 5. Inventory Balances
            modelBuilder.Entity<InventoryBalance>().HasData(
                // Tempe HQ has Multi-Vitamins & Vitamin C
                new InventoryBalance { Id = 1, WarehouseId = 1, ProductId = 1, Quantity = 4500, LastUpdatedAt = new DateTime(2024, 3, 10, 0, 0, 0, DateTimeKind.Utc) },
                new InventoryBalance { Id = 2, WarehouseId = 1, ProductId = 2, Quantity = 8000, LastUpdatedAt = new DateTime(2024, 3, 10, 0, 0, 0, DateTimeKind.Utc) },
                
                // EU Hub has some Vitamin C
                new InventoryBalance { Id = 3, WarehouseId = 2, ProductId = 2, Quantity = 200, LastUpdatedAt = new DateTime(2024, 2, 15, 0, 0, 0, DateTimeKind.Utc) },
                
                // APAC Hub has Pet Nutrition
                new InventoryBalance { Id = 4, WarehouseId = 3, ProductId = 4, Quantity = 500, LastUpdatedAt = new DateTime(2024, 3, 5, 0, 0, 0, DateTimeKind.Utc) }
            );
        }
    }
}
