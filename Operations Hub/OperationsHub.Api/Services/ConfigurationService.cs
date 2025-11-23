using Microsoft.EntityFrameworkCore;
using OperationsHub.Api.Models.DTOs;
using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Repositories;

namespace OperationsHub.Api.Services
{
    public interface IConfigurationService
    {
        Task<IEnumerable<ReorderThresholdDto>> GetReorderThresholdsAsync();
        Task<ReorderThresholdDto> SetReorderThresholdAsync(CreateReorderThresholdDto dto);
        Task<IEnumerable<ReplenishmentSuggestionDto>> GetReplenishmentSuggestionsAsync();
        Task UpdateSuggestionStatusAsync(int id, string status);
    }

    public class ConfigurationService : IConfigurationService
    {
        private readonly IRepository<ReorderThreshold> _thresholdRepository;
        private readonly IRepository<ReplenishmentSuggestion> _suggestionRepository;
        private readonly IRepository<Product> _productRepository;
        private readonly IRepository<Warehouse> _warehouseRepository;

        public ConfigurationService(
            IRepository<ReorderThreshold> thresholdRepository,
            IRepository<ReplenishmentSuggestion> suggestionRepository,
            IRepository<Product> productRepository,
            IRepository<Warehouse> warehouseRepository)
        {
            _thresholdRepository = thresholdRepository;
            _suggestionRepository = suggestionRepository;
            _productRepository = productRepository;
            _warehouseRepository = warehouseRepository;
        }

        public async Task<IEnumerable<ReorderThresholdDto>> GetReorderThresholdsAsync()
        {
            return await _thresholdRepository.Query()
                .Include(t => t.Product)
                .Include(t => t.Warehouse)
                .Select(t => new ReorderThresholdDto
                {
                    Id = t.Id,
                    ProductId = t.ProductId,
                    ProductName = t.Product!.Name,
                    WarehouseId = t.WarehouseId,
                    WarehouseName = t.Warehouse!.Name,
                    MinQuantity = t.MinQuantity,
                    ReorderQuantity = t.ReorderQuantity
                })
                .ToListAsync();
        }

        public async Task<ReorderThresholdDto> SetReorderThresholdAsync(CreateReorderThresholdDto dto)
        {
            // Check if exists
            var existing = await _thresholdRepository.Query()
                .FirstOrDefaultAsync(t => t.ProductId == dto.ProductId && t.WarehouseId == dto.WarehouseId);

            if (existing != null)
            {
                existing.MinQuantity = dto.MinQuantity;
                existing.ReorderQuantity = dto.ReorderQuantity;
                await _thresholdRepository.UpdateAsync(existing);
                await _thresholdRepository.SaveChangesAsync();

                // Load nav properties for return
                // Ideally we'd reload or project, simplified here by assuming IDs valid
                return new ReorderThresholdDto
                {
                    Id = existing.Id,
                    ProductId = existing.ProductId,
                    WarehouseId = existing.WarehouseId,
                    MinQuantity = existing.MinQuantity,
                    ReorderQuantity = existing.ReorderQuantity
                    // Names might be missing in this simplified return if not re-fetched, handled by client or separate fetch
                };
            }
            else
            {
                var newThreshold = new ReorderThreshold
                {
                    ProductId = dto.ProductId,
                    WarehouseId = dto.WarehouseId,
                    MinQuantity = dto.MinQuantity,
                    ReorderQuantity = dto.ReorderQuantity
                };
                await _thresholdRepository.AddAsync(newThreshold);
                await _thresholdRepository.SaveChangesAsync();

                return new ReorderThresholdDto
                {
                    Id = newThreshold.Id,
                    ProductId = newThreshold.ProductId,
                    WarehouseId = newThreshold.WarehouseId,
                    MinQuantity = newThreshold.MinQuantity,
                    ReorderQuantity = newThreshold.ReorderQuantity
                };
            }
        }

        public async Task<IEnumerable<ReplenishmentSuggestionDto>> GetReplenishmentSuggestionsAsync()
        {
            return await _suggestionRepository.Query()
                .Include(s => s.Product)
                .Include(s => s.Warehouse)
                .Select(s => new ReplenishmentSuggestionDto
                {
                    Id = s.Id,
                    ProductId = s.ProductId,
                    ProductName = s.Product!.Name,
                    WarehouseId = s.WarehouseId,
                    WarehouseName = s.Warehouse!.Name,
                    CurrentQuantity = s.CurrentQuantity,
                    SuggestedReorderQuantity = s.SuggestedReorderQuantity,
                    Status = s.Status,
                    CreatedAt = s.CreatedAt
                })
                .OrderByDescending(s => s.CreatedAt)
                .ToListAsync();
        }

        public async Task UpdateSuggestionStatusAsync(int id, string status)
        {
            var suggestion = await _suggestionRepository.GetByIdAsync(id);
            if (suggestion != null)
            {
                suggestion.Status = status;
                await _suggestionRepository.UpdateAsync(suggestion);
                await _suggestionRepository.SaveChangesAsync();
            }
        }
    }
}

