using System.ComponentModel.DataAnnotations;

namespace OperationsHub.Api.Models.DTOs
{
    public class CreateInventoryMovementDto
    {
        [Required]
        public int ProductId { get; set; }

        [Required]
        public int WarehouseId { get; set; }

        [Required]
        public int Quantity { get; set; }

        [Required]
        [MaxLength(50)]
        public string MovementType { get; set; } = string.Empty; // "Receipt", "Adjustment"

        [MaxLength(100)]
        public string? Reference { get; set; }
    }
}

