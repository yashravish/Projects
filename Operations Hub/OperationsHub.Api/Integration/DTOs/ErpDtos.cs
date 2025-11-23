namespace OperationsHub.Api.Integration.DTOs
{
    public class ErpProductDto
    {
        public string ErpProductId { get; set; } = string.Empty;
        public string Sku { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string? Category { get; set; }
        public bool IsActive { get; set; }
    }

    public class ErpOrderDto
    {
        public string ErpOrderId { get; set; } = string.Empty;
        public string? ExternalOrderNumber { get; set; }
        public string CustomerName { get; set; } = string.Empty;
        public string? Region { get; set; }
        public DateTime OrderDate { get; set; }
        public List<ErpOrderLineDto> Lines { get; set; } = new();
    }

    public class ErpOrderLineDto
    {
        public string ProductSku { get; set; } = string.Empty;
        public int Quantity { get; set; }
    }
}

