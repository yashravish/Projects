using System.ComponentModel.DataAnnotations;

namespace OperationsHub.Api.Models.DTOs
{
    public class ReorderThresholdDto
    {
        public int Id { get; set; }
        public int ProductId { get; set; }
        public string ProductName { get; set; } = string.Empty;
        public int WarehouseId { get; set; }
        public string WarehouseName { get; set; } = string.Empty;
        public int MinQuantity { get; set; }
        public int ReorderQuantity { get; set; }
    }

    public class CreateReorderThresholdDto
    {
        [Required]
        public int ProductId { get; set; }

        [Required]
        public int WarehouseId { get; set; }

        [Required]
        [Range(1, int.MaxValue)]
        public int MinQuantity { get; set; }

        [Required]
        [Range(1, int.MaxValue)]
        public int ReorderQuantity { get; set; }
    }

    public class ReplenishmentSuggestionDto
    {
        public int Id { get; set; }
        public int ProductId { get; set; }
        public string ProductName { get; set; } = string.Empty;
        public int WarehouseId { get; set; }
        public string WarehouseName { get; set; } = string.Empty;
        public int CurrentQuantity { get; set; }
        public int SuggestedReorderQuantity { get; set; }
        public string Status { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; }
    }
}

