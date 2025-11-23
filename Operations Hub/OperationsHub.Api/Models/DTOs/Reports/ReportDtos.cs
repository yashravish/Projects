namespace OperationsHub.Api.Models.DTOs.Reports
{
    public class SalesByProductRegionDto
    {
        public string Month { get; set; } = string.Empty;
        public string ProductName { get; set; } = string.Empty;
        public string Category { get; set; } = string.Empty;
        public string Region { get; set; } = string.Empty;
        public int TotalQuantity { get; set; }
    }

    public class InventoryAgingDto
    {
        public string ProductName { get; set; } = string.Empty;
        public string BatchNumber { get; set; } = string.Empty;
        public string WarehouseName { get; set; } = string.Empty; // Derived from location or context if possible, but Batch is product-level. 
        // Wait, Batch entity has RemainingQuantity but not Location. InventoryBalance has Location/Warehouse but not Batch.
        // Requirement says: "Joins Batches with Products and Warehouses."
        // In our current model: Batch is global per product. InventoryBalance is per warehouse. They are not directly linked (no BatchId on InventoryBalance).
        // This is a common simplified ERP model limitation.
        // For this exercise, we will list Batches and assume they are in the primary warehouse or aggregate globally, OR
        // we can just show Batch Aging globally.
        // Let's stick to the view definition: "Outputs: ProductName, BatchNumber, WarehouseName, DaysToExpiry, RemainingQuantity."
        // Since we can't easily link Batch to Warehouse without a BatchBalance table, we will omit WarehouseName or use a placeholder/assumption.
        // Actually, let's assume the view does a cross join or we just show global batch aging.
        public int DaysToExpiry { get; set; }
        public int RemainingQuantity { get; set; }
        public string Status { get; set; } = string.Empty;
    }

    public class FillRateDto
    {
        public string ProductName { get; set; } = string.Empty;
        public int TotalOrdered { get; set; }
        public int TotalShipped { get; set; }
        public double FillRatePercent { get; set; }
    }
}

