using OperationsHub.Api.Models.Entities;

namespace OperationsHub.Api.Services
{
    public interface IInventoryService
    {
        Task<IEnumerable<InventoryBalance>> GetInventoryByWarehouseAsync(int warehouseId);
        Task<IEnumerable<InventoryBalance>> GetLowStockAsync(int threshold);
        Task ApplyInventoryMovementAsync(InventoryMovement movement);
        Task<List<ReplenishmentSuggestion>> GenerateReplenishmentSuggestionsAsync();
    }
}

