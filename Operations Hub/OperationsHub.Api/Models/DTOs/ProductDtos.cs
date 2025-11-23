using System.ComponentModel.DataAnnotations;

namespace OperationsHub.Api.Models.DTOs
{
    public class CreateProductDto
    {
        [Required]
        [MaxLength(50)]
        public string Sku { get; set; } = string.Empty;

        [Required]
        [MaxLength(200)]
        public string Name { get; set; } = string.Empty;

        [MaxLength(100)]
        public string? Category { get; set; }
    }

    public class UpdateProductDto
    {
        [Required]
        [MaxLength(200)]
        public string Name { get; set; } = string.Empty;

        [MaxLength(100)]
        public string? Category { get; set; }
    }

    public class ProductDetailDto
    {
        public int Id { get; set; }
        public string Sku { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string? Category { get; set; }
        public bool IsActive { get; set; }
        public int TotalStock { get; set; }
        public DateTime CreatedAt { get; set; }
    }
}

