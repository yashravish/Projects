using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using FluentAssertions;
using Moq;
using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Repositories;
using OperationsHub.Api.Services;
using Xunit;

namespace OperationsHub.Tests
{
    public class OrderServiceTests
    {
        private readonly Mock<IRepository<CustomerOrder>> _mockOrderRepo;
        private readonly Mock<IRepository<Warehouse>> _mockWarehouseRepo;
        private readonly Mock<IRepository<InventoryBalance>> _mockBalanceRepo;
        private readonly Mock<IInventoryService> _mockInventoryService;
        private readonly OrderService _service;

        public OrderServiceTests()
        {
            _mockOrderRepo = new Mock<IRepository<CustomerOrder>>();
            _mockWarehouseRepo = new Mock<IRepository<Warehouse>>();
            _mockBalanceRepo = new Mock<IRepository<InventoryBalance>>();
            _mockInventoryService = new Mock<IInventoryService>();

            _service = new OrderService(
                _mockOrderRepo.Object,
                _mockWarehouseRepo.Object,
                _mockBalanceRepo.Object,
                _mockInventoryService.Object);
        }

        [Fact]
        public async Task AllocateOrderAsync_ShouldAllocate_WhenInventoryExists()
        {
            // Arrange
            var orderId = 1;
            var productId = 100;
            var warehouseId = 10;
            var quantity = 5;

            var order = new CustomerOrder
            {
                Id = orderId,
                Status = "Pending",
                Region = "US",
                OrderLines = new List<OrderLine>
                {
                    new OrderLine { ProductId = productId, Quantity = quantity, LineStatus = "Open" }
                }
            };

            var warehouses = new List<Warehouse>
            {
                new Warehouse { Id = warehouseId, Region = "US", Name = "US Warehouse" }
            };

            var balances = new List<InventoryBalance>
            {
                new InventoryBalance { 
                    WarehouseId = warehouseId, 
                    ProductId = productId, 
                    Quantity = 10,
                    Warehouse = warehouses[0]
                }
            };

            // Setup Mocks using manual helper
            _mockOrderRepo.Setup(r => r.Query()).Returns(new List<CustomerOrder> { order }.BuildMock().Object);
            _mockWarehouseRepo.Setup(r => r.GetAllAsync()).ReturnsAsync(warehouses);
            _mockBalanceRepo.Setup(r => r.Query()).Returns(balances.BuildMock().Object);

            // Act
            await _service.AllocateOrderAsync(orderId);

            // Assert
            order.Status.Should().Be("Allocated");
            order.OrderLines.First().LineStatus.Should().Be("Allocated");
            order.OrderLines.First().AllocatedWarehouseId.Should().Be(warehouseId);

            // Verify Inventory Movement
            _mockInventoryService.Verify(s => s.ApplyInventoryMovementAsync(It.Is<InventoryMovement>(
                m => m.ProductId == productId && 
                     m.WarehouseId == warehouseId && 
                     m.Quantity == quantity && 
                     m.MovementType == "Pick")), Times.Once);
            
            // Verify Order Update
            _mockOrderRepo.Verify(r => r.UpdateAsync(It.IsAny<CustomerOrder>()), Times.Once);
            _mockOrderRepo.Verify(r => r.SaveChangesAsync(), Times.Once);
        }

        [Fact]
        public async Task AllocateOrderAsync_ShouldBackorder_WhenInventoryInsufficient()
        {
            // Arrange
            var orderId = 1;
            var productId = 100;
            var quantity = 50; 

            var order = new CustomerOrder
            {
                Id = orderId,
                Status = "Pending",
                Region = "US",
                OrderLines = new List<OrderLine>
                {
                    new OrderLine { ProductId = productId, Quantity = quantity, LineStatus = "Open" }
                }
            };

            var warehouses = new List<Warehouse>
            {
                new Warehouse { Id = 10, Region = "US", Name = "US Warehouse" }
            };

            var balances = new List<InventoryBalance>
            {
                new InventoryBalance { 
                    WarehouseId = 10, 
                    ProductId = productId, 
                    Quantity = 5,
                    Warehouse = warehouses[0]
                }
            };

            // Setup Mocks
            _mockOrderRepo.Setup(r => r.Query()).Returns(new List<CustomerOrder> { order }.BuildMock().Object);
            _mockWarehouseRepo.Setup(r => r.GetAllAsync()).ReturnsAsync(warehouses);
            _mockBalanceRepo.Setup(r => r.Query()).Returns(balances.BuildMock().Object);

            // Act
            await _service.AllocateOrderAsync(orderId);

            // Assert
            order.Status.Should().Be("Pending"); 
            order.OrderLines.First().LineStatus.Should().Be("Backordered");
            order.OrderLines.First().AllocatedWarehouseId.Should().BeNull();

            // Verify NO Inventory Movement
            _mockInventoryService.Verify(s => s.ApplyInventoryMovementAsync(It.IsAny<InventoryMovement>()), Times.Never);
        }
    }
}
