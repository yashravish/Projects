using Microsoft.AspNetCore.Mvc;
using OperationsHub.Api.Models.DTOs;
using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Services;

namespace OperationsHub.Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class InventoryController : ControllerBase
    {
        private readonly IInventoryService _inventoryService;

        public InventoryController(IInventoryService inventoryService)
        {
            _inventoryService = inventoryService;
        }

        /// <summary>
        /// Applies a manual inventory movement (Receipt, Adjustment, etc.).
        /// </summary>
        [HttpPost("movements")]
        public async Task<IActionResult> CreateMovement(CreateInventoryMovementDto dto)
        {
            var movement = new InventoryMovement
            {
                ProductId = dto.ProductId,
                WarehouseId = dto.WarehouseId,
                Quantity = dto.Quantity,
                MovementType = dto.MovementType,
                Reference = dto.Reference
            };

            await _inventoryService.ApplyInventoryMovementAsync(movement);
            return Ok(new { Message = "Movement applied successfully." });
        }

        /// <summary>
        /// Retrieves items with stock below the specified threshold.
        /// </summary>
        [HttpGet("low-stock")]
        public async Task<ActionResult<IEnumerable<InventoryBalanceDto>>> GetLowStock([FromQuery] int threshold = 100)
        {
            // Ideally fetch threshold from config if not provided, keeping it simple here
            var balances = await _inventoryService.GetLowStockAsync(threshold);

            var dtos = balances.Select(b => new InventoryBalanceDto
            {
                Id = b.Id,
                ProductId = b.ProductId,
                ProductName = b.Product?.Name ?? "Unknown",
                ProductSku = b.Product?.Sku ?? "Unknown",
                Quantity = b.Quantity,
                LastUpdatedAt = b.LastUpdatedAt
            });

            return Ok(dtos);
        }

        /// <summary>
        /// Triggers bulk generation of replenishment suggestions based on configured thresholds.
        /// </summary>
        [HttpPost("generate-replenishment")]
        public async Task<IActionResult> GenerateReplenishmentSuggestions()
        {
            var suggestions = await _inventoryService.GenerateReplenishmentSuggestionsAsync();
            return Ok(new { Message = $"Generated {suggestions.Count} replenishment suggestions.", Suggestions = suggestions });
        }
    }
}

