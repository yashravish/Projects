using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace OperationsHub.Api.Models.Entities
{
    // Rationale for Option B (Dedicated Entity):
    // While ConfigSetting (Option A) is flexible, reorder thresholds are structured business data 
    // that relate specific relational entities (Product + Warehouse). 
    // Using a dedicated table allows for:
    // 1. Referential Integrity: We can enforce FKs to Product and Warehouse.
    // 2. Query Efficiency: Easier to join in SQL/LINQ when calculating replenishments.
    // 3. Type Safety: Integers for thresholds vs parsing strings from ConfigSetting.
    
    public class ReorderThreshold
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public int ProductId { get; set; }

        [Required]
        public int WarehouseId { get; set; }

        [Required]
        public int MinQuantity { get; set; } // Trigger point

        [Required]
        public int ReorderQuantity { get; set; } // Suggested amount to order

        // Navigation Properties
        [ForeignKey(nameof(ProductId))]
        public Product? Product { get; set; }

        [ForeignKey(nameof(WarehouseId))]
        public Warehouse? Warehouse { get; set; }
    }
}

