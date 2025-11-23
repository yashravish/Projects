using System.ComponentModel.DataAnnotations;

namespace OperationsHub.Api.Models.DTOs
{
    public class CreateOrderDto
    {
        [Required]
        [MaxLength(200)]
        public string CustomerName { get; set; } = string.Empty;

        [MaxLength(50)]
        public string? Region { get; set; }

        public string? ExternalOrderNumber { get; set; }

        [Required]
        public List<CreateOrderLineDto> Lines { get; set; } = new();
    }

    public class CreateOrderLineDto
    {
        [Required]
        public int ProductId { get; set; }

        [Required]
        [Range(1, int.MaxValue)]
        public int Quantity { get; set; }
    }

    public class OrderDto
    {
        public int Id { get; set; }
        public string CustomerName { get; set; } = string.Empty;
        public string? Region { get; set; }
        public string Status { get; set; } = string.Empty;
        public string? ExternalOrderNumber { get; set; }
        public DateTime OrderDate { get; set; }
        public List<OrderLineDto> Lines { get; set; } = new();
    }

    public class OrderLineDto
    {
        public int Id { get; set; }
        public int ProductId { get; set; }
        public string ProductName { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public int? AllocatedWarehouseId { get; set; }
        public string? AllocatedWarehouseName { get; set; }
        public int ShippedQuantity { get; set; }
        public string LineStatus { get; set; } = string.Empty;
    }
}

