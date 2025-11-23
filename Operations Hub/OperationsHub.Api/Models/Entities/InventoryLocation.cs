using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace OperationsHub.Api.Models.Entities
{
    public class InventoryLocation
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public int WarehouseId { get; set; }

        [Required]
        [MaxLength(50)]
        public string Code { get; set; } = string.Empty;

        [MaxLength(200)]
        public string? Description { get; set; }

        // Navigation Properties
        [ForeignKey(nameof(WarehouseId))]
        public Warehouse? Warehouse { get; set; }

        [InverseProperty(nameof(InventoryMovement.FromLocation))]
        public ICollection<InventoryMovement> MovementsFrom { get; set; } = new List<InventoryMovement>();

        [InverseProperty(nameof(InventoryMovement.ToLocation))]
        public ICollection<InventoryMovement> MovementsTo { get; set; } = new List<InventoryMovement>();
    }
}

