using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace OperationsHub.Api.Models.Entities
{
    public class InventoryMovement
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public int ProductId { get; set; }

        public int? FromLocationId { get; set; }

        public int? ToLocationId { get; set; }

        [Required]
        public int WarehouseId { get; set; }

        public int Quantity { get; set; }

        [Required]
        [MaxLength(50)]
        public string MovementType { get; set; } = string.Empty;

        public DateTime PerformedAt { get; set; } = DateTime.UtcNow;

        [MaxLength(100)]
        public string? Reference { get; set; }

        // Navigation Properties
        [ForeignKey(nameof(ProductId))]
        public Product? Product { get; set; }

        [ForeignKey(nameof(WarehouseId))]
        public Warehouse? Warehouse { get; set; }

        [ForeignKey(nameof(FromLocationId))]
        public InventoryLocation? FromLocation { get; set; }

        [ForeignKey(nameof(ToLocationId))]
        public InventoryLocation? ToLocation { get; set; }
    }
}

