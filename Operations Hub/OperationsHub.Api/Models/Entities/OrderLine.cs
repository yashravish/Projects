using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace OperationsHub.Api.Models.Entities
{
    public class OrderLine
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public int OrderId { get; set; }

        [Required]
        public int ProductId { get; set; }

        public int Quantity { get; set; }

        public int? AllocatedWarehouseId { get; set; }

        public int ShippedQuantity { get; set; }

        [Required]
        [MaxLength(50)]
        public string LineStatus { get; set; } = "Open";

        // Navigation Properties
        [ForeignKey(nameof(OrderId))]
        public CustomerOrder? Order { get; set; }

        [ForeignKey(nameof(ProductId))]
        public Product? Product { get; set; }

        [ForeignKey(nameof(AllocatedWarehouseId))]
        public Warehouse? AllocatedWarehouse { get; set; }
    }
}

