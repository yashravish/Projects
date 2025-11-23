using Microsoft.AspNetCore.Mvc;

namespace OperationsHub.Api.Controllers
{
    [Route("mock-erp")]
    [ApiController]
    public class MockErpController : ControllerBase
    {
        [HttpGet("products")]
        public IActionResult GetProducts()
        {
            var products = new[]
            {
                new { ErpProductId = "ERP-101", Sku = "VIT-MULTI-001", Name = "Daily Multi-Vitamin (Men)", Category = "Vitamin", IsActive = true },
                new { ErpProductId = "ERP-102", Sku = "VIT-C-500", Name = "Vitamin C 500mg", Category = "Vitamin", IsActive = true },
                new { ErpProductId = "ERP-103", Sku = "NEW-PROD-001", Name = "New Fish Oil 1000mg", Category = "Supplement", IsActive = true }, // New product to test sync
                new { ErpProductId = "ERP-104", Sku = "HERB-ECHIN", Name = "Organic Echinacea Tea", Category = "Herbal Extract", IsActive = true }
            };
            return Ok(products);
        }

        [HttpGet("orders")]
        public IActionResult GetOrders()
        {
            var orders = new[]
            {
                new 
                { 
                    ErpOrderId = "ORD-9991", 
                    ExternalOrderNumber = "PO-CLIENT-A-001", 
                    CustomerName = "Health Foods Inc.", 
                    Region = "US", 
                    OrderDate = DateTime.UtcNow.AddDays(-2),
                    Lines = new[] 
                    {
                        new { ProductSku = "VIT-MULTI-001", Quantity = 100 },
                        new { ProductSku = "VIT-C-500", Quantity = 50 }
                    }
                },
                new 
                { 
                    ErpOrderId = "ORD-9992", 
                    ExternalOrderNumber = "PO-CLIENT-B-005", 
                    CustomerName = "Global Supplements Ltd.", 
                    Region = "EU", 
                    OrderDate = DateTime.UtcNow.AddDays(-1),
                    Lines = new[] 
                    {
                        new { ProductSku = "VIT-C-500", Quantity = 200 }
                    }
                }
            };
            return Ok(orders);
        }

        [HttpPost("orders/{erpOrderId}/ship-confirm")]
        public IActionResult ConfirmShipment(string erpOrderId)
        {
            // Simulate processing
            Console.WriteLine($"[MOCK ERP] Shipment confirmed for Order {erpOrderId}");
            return Ok(new { Message = "Shipment confirmed received." });
        }
    }
}

