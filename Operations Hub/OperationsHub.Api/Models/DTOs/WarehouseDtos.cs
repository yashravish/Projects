using System.ComponentModel.DataAnnotations;

namespace OperationsHub.Api.Models.DTOs
{
    public class WarehouseDto
    {
        public int Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public string Code { get; set; } = string.Empty;
        public string? Region { get; set; }
        public bool IsActive { get; set; }
    }

    public class CreateWarehouseDto
    {
        [Required]
        [MaxLength(100)]
        public string Name { get; set; } = string.Empty;

        [Required]
        [MaxLength(50)]
        public string Code { get; set; } = string.Empty;

        [MaxLength(50)]
        public string? Region { get; set; }
    }

    public class InventoryBalanceDto
    {
        public int Id { get; set; }
        public int ProductId { get; set; }
        public string ProductName { get; set; } = string.Empty;
        public string ProductSku { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public DateTime LastUpdatedAt { get; set; }
    }
}

