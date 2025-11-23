using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace OperationsHub.Api.Models.Entities
{
    public class Batch
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public int ProductId { get; set; }

        [Required]
        [MaxLength(100)]
        public string BatchNumber { get; set; } = string.Empty;

        public DateTime ManufactureDate { get; set; }

        public DateTime ExpiryDate { get; set; }

        public int InitialQuantity { get; set; }

        public int RemainingQuantity { get; set; }

        [Required]
        [MaxLength(50)]
        public string Status { get; set; } = "Open";

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public DateTime? UpdatedAt { get; set; }

        // Navigation Properties
        [ForeignKey(nameof(ProductId))]
        public Product? Product { get; set; }
    }
}

