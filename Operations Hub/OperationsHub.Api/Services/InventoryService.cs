using Microsoft.EntityFrameworkCore;
using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Repositories;

namespace OperationsHub.Api.Services
{
    public partial class InventoryService : IInventoryService
    {
        private readonly IRepository<InventoryBalance> _balanceRepository;
        private readonly IRepository<InventoryMovement> _movementRepository;
        private readonly IRepository<ReorderThreshold> _thresholdRepository;
        private readonly IRepository<ReplenishmentSuggestion> _suggestionRepository;

        public InventoryService(
            IRepository<InventoryBalance> balanceRepository,
            IRepository<InventoryMovement> movementRepository,
            IRepository<ReorderThreshold> thresholdRepository,
            IRepository<ReplenishmentSuggestion> suggestionRepository)
        {
            _balanceRepository = balanceRepository;
            _movementRepository = movementRepository;
            _thresholdRepository = thresholdRepository;
            _suggestionRepository = suggestionRepository;
        }

        public async Task<IEnumerable<InventoryBalance>> GetInventoryByWarehouseAsync(int warehouseId)
        {
            return await _balanceRepository.Query()
                .Include(b => b.Product)
                .Where(b => b.WarehouseId == warehouseId)
                .ToListAsync();
        }

        public async Task<IEnumerable<InventoryBalance>> GetLowStockAsync(int threshold)
        {
            return await _balanceRepository.Query()
                .Include(b => b.Product)
                .Include(b => b.Warehouse)
                .Where(b => b.Quantity <= threshold)
                .ToListAsync();
        }

        public async Task ApplyInventoryMovementAsync(InventoryMovement movement)
        {
            // 1. Record the movement
            movement.PerformedAt = DateTime.UtcNow;
            await _movementRepository.AddAsync(movement);

            // 2. Update the balance
            var balance = await _balanceRepository.Query()
                .FirstOrDefaultAsync(b => b.WarehouseId == movement.WarehouseId && b.ProductId == movement.ProductId);

            if (balance == null)
            {
                // Create new balance entry if it doesn't exist
                balance = new InventoryBalance
                {
                    WarehouseId = movement.WarehouseId,
                    ProductId = movement.ProductId,
                    Quantity = 0,
                    LastUpdatedAt = DateTime.UtcNow
                };
                await _balanceRepository.AddAsync(balance);
            }

            // Adjust quantity based on movement type
            switch (movement.MovementType)
            {
                case "Receipt":
                case "Adjustment":
                case "Transfer In":
                    balance.Quantity += movement.Quantity;
                    break;
                case "Pick":
                case "Transfer Out":
                    balance.Quantity -= movement.Quantity;
                    break;
                default:
                    if (movement.MovementType == "Adjustment" && movement.Quantity < 0)
                     {
                         balance.Quantity += movement.Quantity; 
                     }
                     break;
            }
            
            balance.LastUpdatedAt = DateTime.UtcNow;
            await _balanceRepository.UpdateAsync(balance);

            await _movementRepository.SaveChangesAsync();
            await _balanceRepository.SaveChangesAsync();
            
            // Trigger Replenishment Logic (Fire and Forget or Await)
            await CheckReplenishmentAsync(movement.WarehouseId, movement.ProductId, balance.Quantity);
        }

        public Task<List<ReplenishmentSuggestion>> GenerateReplenishmentSuggestionsAsync()
        {
             // Bulk generation logic could go here if we want to scan ALL inventory
             // For now, implemented per-movement check below.
             return Task.FromResult(new List<ReplenishmentSuggestion>());
        }

        private async Task CheckReplenishmentAsync(int warehouseId, int productId, int currentQty)
        {
            // 1. Get Threshold
            var threshold = await _thresholdRepository.Query()
                .FirstOrDefaultAsync(t => t.WarehouseId == warehouseId && t.ProductId == productId);

            if (threshold == null) return; // No policy defined

            // 2. Check Condition
            if (currentQty <= threshold.MinQuantity)
            {
                // 3. Check for existing Open suggestion to avoid spam
                var existing = await _suggestionRepository.Query()
                    .AnyAsync(s => s.WarehouseId == warehouseId 
                                   && s.ProductId == productId 
                                   && s.Status == "Open");
                
                if (!existing)
                {
                    // 4. Create Suggestion
                    var suggestion = new ReplenishmentSuggestion
                    {
                        WarehouseId = warehouseId,
                        ProductId = productId,
                        CurrentQuantity = currentQty,
                        SuggestedReorderQuantity = threshold.ReorderQuantity,
                        Status = "Open",
                        CreatedAt = DateTime.UtcNow
                    };
                    await _suggestionRepository.AddAsync(suggestion);
                    await _suggestionRepository.SaveChangesAsync();
                }
            }
        }
    }
}
