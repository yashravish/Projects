using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace OperationsHub.Api.Models.Entities
{
    public class CustomerOrder
    {
        [Key]
        public int Id { get; set; }

        [MaxLength(100)]
        public string? ExternalOrderNumber { get; set; }

        [Required]
        [MaxLength(200)]
        public string CustomerName { get; set; } = string.Empty;

        [MaxLength(50)]
        public string? Region { get; set; }

        public DateTime OrderDate { get; set; }

        [Required]
        [MaxLength(50)]
        public string Status { get; set; } = "Pending";

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public DateTime? UpdatedAt { get; set; }

        // Navigation Properties
        public ICollection<OrderLine> OrderLines { get; set; } = new List<OrderLine>();
    }
}

