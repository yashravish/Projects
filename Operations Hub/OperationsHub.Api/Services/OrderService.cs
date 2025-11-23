using Microsoft.EntityFrameworkCore;
using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Repositories;

namespace OperationsHub.Api.Services
{
    public class OrderService : IOrderService
    {
        private readonly IRepository<CustomerOrder> _orderRepository;
        private readonly IRepository<Warehouse> _warehouseRepository;
        private readonly IRepository<InventoryBalance> _balanceRepository;
        private readonly IInventoryService _inventoryService;

        public OrderService(
            IRepository<CustomerOrder> orderRepository,
            IRepository<Warehouse> warehouseRepository,
            IRepository<InventoryBalance> balanceRepository,
            IInventoryService inventoryService)
        {
            _orderRepository = orderRepository;
            _warehouseRepository = warehouseRepository;
            _balanceRepository = balanceRepository;
            _inventoryService = inventoryService;
        }

        public async Task<CustomerOrder> CreateOrderAsync(CustomerOrder order)
        {
            order.Status = "Pending";
            order.CreatedAt = DateTime.UtcNow;
            foreach (var line in order.OrderLines)
            {
                line.LineStatus = "Open";
            }

            await _orderRepository.AddAsync(order);
            await _orderRepository.SaveChangesAsync();
            return order;
        }

        public async Task AllocateOrderAsync(int orderId)
        {
            var order = await _orderRepository.Query()
                .Include(o => o.OrderLines)
                .FirstOrDefaultAsync(o => o.Id == orderId);

            if (order == null) throw new KeyNotFoundException($"Order {orderId} not found.");
            if (order.Status != "Pending") throw new InvalidOperationException("Only pending orders can be allocated.");

            var warehouses = await _warehouseRepository.GetAllAsync();
            
            bool allLinesAllocated = true;

            foreach (var line in order.OrderLines)
            {
                if (line.LineStatus != "Open") continue;

                // Strategy:
                // 1. Filter warehouses by Region match (if possible)
                // 2. Sort by highest available quantity
                
                var balances = await _balanceRepository.Query()
                    .Include(b => b.Warehouse)
                    .Where(b => b.ProductId == line.ProductId && b.Quantity >= line.Quantity)
                    .ToListAsync();

                var bestMatch = balances
                    .OrderByDescending(b => b.Warehouse!.Region == order.Region) // true (1) comes after false (0)? No, bool sort is False then True.
                    // Let's be explicit:
                    .OrderByDescending(b => b.Warehouse!.Region == order.Region ? 1 : 0)
                    .ThenByDescending(b => b.Quantity)
                    .FirstOrDefault();

                if (bestMatch != null)
                {
                    // Allocate
                    line.AllocatedWarehouseId = bestMatch.WarehouseId;
                    line.LineStatus = "Allocated";

                    // Deduct inventory via movement (reservation logic could be more complex, but sticking to requirements)
                    // Requirement: "Adjust InventoryBalance and create InventoryMovement records."
                    // We treat allocation as a "hold" or "soft pick". 
                    // Actually, "Pick" usually happens at shipping. 
                    // But if we want to adjust balance now to prevent double allocation:
                    
                    await _inventoryService.ApplyInventoryMovementAsync(new InventoryMovement
                    {
                        ProductId = line.ProductId,
                        WarehouseId = bestMatch.WarehouseId,
                        Quantity = line.Quantity,
                        MovementType = "Pick", // Using Pick to decrement immediately for this exercise
                        Reference = $"Order-{order.ExternalOrderNumber ?? order.Id.ToString()}",
                        PerformedAt = DateTime.UtcNow
                    });
                }
                else
                {
                    line.LineStatus = "Backordered";
                    allLinesAllocated = false;
                }
            }

            order.Status = allLinesAllocated ? "Allocated" : "Partially Allocated"; // Or stick to requirements status list
            if (!allLinesAllocated && order.OrderLines.All(l => l.LineStatus == "Backordered"))
            {
                 // If everything failed
                 order.Status = "Pending"; // Or "Backordered" if we had that status
            }
            else if (allLinesAllocated)
            {
                 order.Status = "Allocated";
            }
            
            order.UpdatedAt = DateTime.UtcNow;
            await _orderRepository.UpdateAsync(order);
            await _orderRepository.SaveChangesAsync();
        }

        public async Task ShipOrderAsync(int orderId)
        {
            var order = await _orderRepository.Query()
                .Include(o => o.OrderLines)
                .FirstOrDefaultAsync(o => o.Id == orderId);

            if (order == null) throw new KeyNotFoundException($"Order {orderId} not found.");
            if (order.Status != "Allocated") throw new InvalidOperationException("Order must be allocated before shipping.");

            foreach (var line in order.OrderLines)
            {
                if (line.LineStatus == "Allocated")
                {
                    line.LineStatus = "Shipped";
                    line.ShippedQuantity = line.Quantity;
                }
            }

            order.Status = "Shipped";
            order.UpdatedAt = DateTime.UtcNow;
            await _orderRepository.UpdateAsync(order);
            await _orderRepository.SaveChangesAsync();
        }

        public async Task<CustomerOrder?> GetOrderDetailsAsync(int orderId)
        {
             return await _orderRepository.Query()
                .Include(o => o.OrderLines)
                .ThenInclude(ol => ol.Product)
                .Include(o => o.OrderLines)
                .ThenInclude(ol => ol.AllocatedWarehouse)
                .FirstOrDefaultAsync(o => o.Id == orderId);
        }
    }
}

