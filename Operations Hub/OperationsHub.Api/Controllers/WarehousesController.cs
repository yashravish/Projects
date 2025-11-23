using Microsoft.AspNetCore.Mvc;
using OperationsHub.Api.Models.DTOs;
using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Repositories;
using OperationsHub.Api.Services;

namespace OperationsHub.Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class WarehousesController : ControllerBase
    {
        private readonly IRepository<Warehouse> _warehouseRepository;
        private readonly IInventoryService _inventoryService;

        public WarehousesController(
            IRepository<Warehouse> warehouseRepository,
            IInventoryService inventoryService)
        {
            _warehouseRepository = warehouseRepository;
            _inventoryService = inventoryService;
        }

        /// <summary>
        /// Retrieves all warehouses.
        /// </summary>
        [HttpGet]
        public async Task<ActionResult<IEnumerable<WarehouseDto>>> GetWarehouses()
        {
            var warehouses = await _warehouseRepository.GetAllAsync();
            var dtos = warehouses.Select(w => new WarehouseDto
            {
                Id = w.Id,
                Name = w.Name,
                Code = w.Code,
                Region = w.Region,
                IsActive = w.IsActive
            });
            return Ok(dtos);
        }

        /// <summary>
        /// Retrieves a specific warehouse by ID.
        /// </summary>
        [HttpGet("{id}")]
        public async Task<ActionResult<WarehouseDto>> GetWarehouse(int id)
        {
            var w = await _warehouseRepository.GetByIdAsync(id);
            if (w == null) return NotFound();

            return Ok(new WarehouseDto
            {
                Id = w.Id,
                Name = w.Name,
                Code = w.Code,
                Region = w.Region,
                IsActive = w.IsActive
            });
        }

        /// <summary>
        /// Creates a new warehouse.
        /// </summary>
        [HttpPost]
        public async Task<ActionResult<WarehouseDto>> CreateWarehouse(CreateWarehouseDto dto)
        {
            var warehouse = new Warehouse
            {
                Name = dto.Name,
                Code = dto.Code,
                Region = dto.Region,
                IsActive = true,
                CreatedAt = DateTime.UtcNow
            };

            await _warehouseRepository.AddAsync(warehouse);
            await _warehouseRepository.SaveChangesAsync();

            var responseDto = new WarehouseDto
            {
                Id = warehouse.Id,
                Name = warehouse.Name,
                Code = warehouse.Code,
                Region = warehouse.Region,
                IsActive = warehouse.IsActive
            };

            return CreatedAtAction(nameof(GetWarehouse), new { id = warehouse.Id }, responseDto);
        }

        /// <summary>
        /// Retrieves inventory balances for a specific warehouse.
        /// </summary>
        [HttpGet("{id}/inventory")]
        public async Task<ActionResult<IEnumerable<InventoryBalanceDto>>> GetWarehouseInventory(int id)
        {
            // Verify warehouse exists
            var w = await _warehouseRepository.GetByIdAsync(id);
            if (w == null) return NotFound("Warehouse not found.");

            var balances = await _inventoryService.GetInventoryByWarehouseAsync(id);
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
    }
}

