using Microsoft.EntityFrameworkCore;
using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Repositories;
using OperationsHub.Api.Services;

namespace OperationsHub.Api.Integration
{
    public class IntegrationService : IIntegrationService
    {
        private readonly IErpClient _erpClient;
        private readonly IRepository<Product> _productRepository;
        private readonly IRepository<CustomerOrder> _orderRepository;
        private readonly IIntegrationLogService _logService;
        private readonly ILogger<IntegrationService> _logger;

        public IntegrationService(
            IErpClient erpClient,
            IRepository<Product> productRepository,
            IRepository<CustomerOrder> orderRepository,
            IIntegrationLogService logService,
            ILogger<IntegrationService> logger)
        {
            _erpClient = erpClient;
            _productRepository = productRepository;
            _orderRepository = orderRepository;
            _logService = logService;
            _logger = logger;
        }

        public async Task SyncProductsFromErpAsync(int jobId)
        {
            await _logService.LogInfoAsync(jobId, "Starting Product Sync...");
            
            try 
            {
                // 1. Fetch from ERP
                var erpProducts = await _erpClient.GetProductsAsync();
                await _logService.LogInfoAsync(jobId, $"Fetched {erpProducts.Count} products from ERP.");

                int added = 0;
                int updated = 0;

                foreach (var erpProd in erpProducts)
                {
                    // 2. Find local product
                    var localProduct = await _productRepository.Query()
                        .FirstOrDefaultAsync(p => p.Sku == erpProd.Sku);

                    if (localProduct == null)
                    {
                        // Create
                        localProduct = new Product
                        {
                            Sku = erpProd.Sku,
                            Name = erpProd.Name,
                            Category = erpProd.Category,
                            IsActive = erpProd.IsActive,
                            CreatedAt = DateTime.UtcNow
                        };
                        await _productRepository.AddAsync(localProduct);
                        added++;
                    }
                    else
                    {
                        // Update
                        localProduct.Name = erpProd.Name;
                        localProduct.Category = erpProd.Category;
                        localProduct.IsActive = erpProd.IsActive;
                        localProduct.UpdatedAt = DateTime.UtcNow;
                        await _productRepository.UpdateAsync(localProduct);
                        updated++;
                    }
                }
                await _productRepository.SaveChangesAsync();
                await _logService.LogInfoAsync(jobId, $"Product Sync Completed. Added: {added}, Updated: {updated}");
            }
            catch (Exception ex)
            {
                await _logService.LogErrorAsync(jobId, "Product Sync Failed", ex);
                throw; // Re-throw to let job handler know it failed
            }
        }

        public async Task SyncOrdersFromErpAsync(int jobId)
        {
             await _logService.LogInfoAsync(jobId, "Starting Order Sync...");

             try
             {
                 var erpOrders = await _erpClient.GetOrdersAsync();
                 await _logService.LogInfoAsync(jobId, $"Fetched {erpOrders.Count} orders from ERP.");

                 int newOrders = 0;

                 foreach (var erpOrder in erpOrders)
                 {
                     // Check if order already exists (by External ID)
                     var exists = await _orderRepository.Query()
                         .AnyAsync(o => o.ExternalOrderNumber == erpOrder.ExternalOrderNumber);

                     if (exists) 
                     {
                         continue; 
                     }

                     // Create Order
                     var newOrder = new CustomerOrder
                     {
                         ExternalOrderNumber = erpOrder.ExternalOrderNumber,
                         CustomerName = erpOrder.CustomerName,
                         Region = erpOrder.Region,
                         OrderDate = erpOrder.OrderDate,
                         Status = "Pending",
                         CreatedAt = DateTime.UtcNow,
                         OrderLines = new List<OrderLine>()
                     };

                     // Map Lines
                     foreach (var line in erpOrder.Lines)
                     {
                         // Find product ID by SKU
                         var product = await _productRepository.Query()
                             .FirstOrDefaultAsync(p => p.Sku == line.ProductSku);

                         if (product != null)
                         {
                             newOrder.OrderLines.Add(new OrderLine
                             {
                                 ProductId = product.Id,
                                 Quantity = line.Quantity,
                                 LineStatus = "Open"
                             });
                         }
                         else
                         {
                             await _logService.LogWarningAsync(jobId, $"Skipping line for unknown SKU {line.ProductSku} in Order {erpOrder.ExternalOrderNumber}");
                         }
                     }

                     if (newOrder.OrderLines.Any())
                     {
                         await _orderRepository.AddAsync(newOrder);
                         newOrders++;
                     }
                     else
                     {
                         await _logService.LogWarningAsync(jobId, $"Order {erpOrder.ExternalOrderNumber} has no valid lines, skipping.");
                     }
                 }

                 await _orderRepository.SaveChangesAsync();
                 await _logService.LogInfoAsync(jobId, $"Order Sync Completed. New Orders Imported: {newOrders}");
             }
             catch (Exception ex)
             {
                 await _logService.LogErrorAsync(jobId, "Order Sync Failed", ex);
                 throw;
             }
        }
    }
}
