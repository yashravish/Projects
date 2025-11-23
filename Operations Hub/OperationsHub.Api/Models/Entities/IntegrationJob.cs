using System.ComponentModel.DataAnnotations;

namespace OperationsHub.Api.Models.Entities
{
    public class IntegrationJob
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [MaxLength(50)]
        public string JobType { get; set; } = string.Empty;

        public DateTime StartedAt { get; set; } = DateTime.UtcNow;

        public DateTime? FinishedAt { get; set; }

        [Required]
        [MaxLength(50)]
        public string Status { get; set; } = "Running";

        public string? Details { get; set; }

        // Navigation Properties
        public ICollection<IntegrationLog> Logs { get; set; } = new List<IntegrationLog>();
    }
}

