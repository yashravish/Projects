using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace OperationsHub.Api.Models.Entities
{
    public class IntegrationLog
    {
        [Key]
        public int Id { get; set; }

        public int? JobId { get; set; }

        [Required]
        [MaxLength(20)]
        public string Level { get; set; } = "Info";

        [Required]
        public string Message { get; set; } = string.Empty;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        // Navigation Properties
        [ForeignKey(nameof(JobId))]
        public IntegrationJob? Job { get; set; }
    }
}

