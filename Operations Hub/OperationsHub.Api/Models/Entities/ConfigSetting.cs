using System.ComponentModel.DataAnnotations;

namespace OperationsHub.Api.Models.Entities
{
    public class ConfigSetting
    {
        [Key]
        [MaxLength(100)]
        public string Key { get; set; } = string.Empty;

        [Required]
        [MaxLength(500)]
        public string Value { get; set; } = string.Empty;
    }
}

