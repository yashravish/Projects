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
    public class InventoryServiceTests
    {
        private readonly Mock<IRepository<InventoryBalance>> _mockBalanceRepo;
        private readonly Mock<IRepository<InventoryMovement>> _mockMovementRepo;
        private readonly Mock<IRepository<ReorderThreshold>> _mockThresholdRepo;
        private readonly Mock<IRepository<ReplenishmentSuggestion>> _mockSuggestionRepo;
        private readonly InventoryService _service;

        public InventoryServiceTests()
        {
            _mockBalanceRepo = new Mock<IRepository<InventoryBalance>>();
            _mockMovementRepo = new Mock<IRepository<InventoryMovement>>();
            _mockThresholdRepo = new Mock<IRepository<ReorderThreshold>>();
            _mockSuggestionRepo = new Mock<IRepository<ReplenishmentSuggestion>>();

            _service = new InventoryService(
                _mockBalanceRepo.Object,
                _mockMovementRepo.Object,
                _mockThresholdRepo.Object,
                _mockSuggestionRepo.Object);
        }

        [Fact]
        public async Task ApplyInventoryMovementAsync_ShouldIncreaseBalance_OnReceipt()
        {
            // Arrange
            var productId = 1;
            var warehouseId = 1;
            var initialQty = 10;
            var receiptQty = 5;

            var movement = new InventoryMovement
            {
                ProductId = productId,
                WarehouseId = warehouseId,
                Quantity = receiptQty,
                MovementType = "Receipt"
            };

            var balance = new InventoryBalance { ProductId = productId, WarehouseId = warehouseId, Quantity = initialQty };
            var balances = new List<InventoryBalance> { balance };

            _mockBalanceRepo.Setup(r => r.Query()).Returns(balances.BuildMock().Object);
            _mockThresholdRepo.Setup(r => r.Query()).Returns(new List<ReorderThreshold>().BuildMock().Object);

            // Act
            await _service.ApplyInventoryMovementAsync(movement);

            // Assert
            balance.Quantity.Should().Be(initialQty + receiptQty);
            _mockBalanceRepo.Verify(r => r.UpdateAsync(balance), Times.Once);
            _mockMovementRepo.Verify(r => r.AddAsync(movement), Times.Once);
        }

        [Fact]
        public async Task ApplyInventoryMovementAsync_ShouldDecreaseBalance_OnPick()
        {
            // Arrange
            var productId = 1;
            var warehouseId = 1;
            var initialQty = 10;
            var pickQty = 3;

            var movement = new InventoryMovement
            {
                ProductId = productId,
                WarehouseId = warehouseId,
                Quantity = pickQty,
                MovementType = "Pick"
            };

            var balance = new InventoryBalance { ProductId = productId, WarehouseId = warehouseId, Quantity = initialQty };
            var balances = new List<InventoryBalance> { balance };

            _mockBalanceRepo.Setup(r => r.Query()).Returns(balances.BuildMock().Object);
            _mockThresholdRepo.Setup(r => r.Query()).Returns(new List<ReorderThreshold>().BuildMock().Object);

            // Act
            await _service.ApplyInventoryMovementAsync(movement);

            // Assert
            balance.Quantity.Should().Be(initialQty - pickQty);
        }

        [Fact]
        public async Task ApplyInventoryMovementAsync_ShouldCreateSuggestion_WhenBelowThreshold()
        {
            // Arrange
            var productId = 1;
            var warehouseId = 1;
            var initialQty = 10;
            var pickQty = 6; // Result = 4
            var thresholdMin = 5;

            var movement = new InventoryMovement
            {
                ProductId = productId,
                WarehouseId = warehouseId,
                Quantity = pickQty,
                MovementType = "Pick"
            };

            var balance = new InventoryBalance { ProductId = productId, WarehouseId = warehouseId, Quantity = initialQty };
            var balances = new List<InventoryBalance> { balance };

            var threshold = new ReorderThreshold { ProductId = productId, WarehouseId = warehouseId, MinQuantity = thresholdMin, ReorderQuantity = 20 };
            var thresholds = new List<ReorderThreshold> { threshold };

            _mockBalanceRepo.Setup(r => r.Query()).Returns(balances.BuildMock().Object);
            _mockThresholdRepo.Setup(r => r.Query()).Returns(thresholds.BuildMock().Object);
            _mockSuggestionRepo.Setup(r => r.Query()).Returns(new List<ReplenishmentSuggestion>().BuildMock().Object); 

            // Act
            await _service.ApplyInventoryMovementAsync(movement);

            // Assert
            balance.Quantity.Should().Be(4);
            _mockSuggestionRepo.Verify(r => r.AddAsync(It.Is<ReplenishmentSuggestion>(
                s => s.ProductId == productId && 
                     s.WarehouseId == warehouseId && 
                     s.CurrentQuantity == 4 && 
                     s.Status == "Open")), Times.Once);
        }
    }
}
