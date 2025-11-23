using Microsoft.EntityFrameworkCore;
using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Repositories;

namespace OperationsHub.Api.Services
{
    public class ProductService : IProductService
    {
        private readonly IRepository<Product> _productRepository;

        public ProductService(IRepository<Product> productRepository)
        {
            _productRepository = productRepository;
        }

        public async Task<IEnumerable<Product>> GetProductsAsync()
        {
            return await _productRepository.Query()
                .Include(p => p.Batches)
                .Include(p => p.InventoryBalances)
                .ToListAsync();
        }

        public async Task<Product?> GetProductByIdAsync(int id)
        {
            return await _productRepository.Query()
                .Include(p => p.Batches)
                .Include(p => p.InventoryBalances)
                .FirstOrDefaultAsync(p => p.Id == id);
        }

        public async Task<Product> CreateProductAsync(Product product)
        {
            // Basic validation could go here
            if (await _productRepository.Query().AnyAsync(p => p.Sku == product.Sku))
            {
                throw new InvalidOperationException($"Product with SKU '{product.Sku}' already exists.");
            }

            product.CreatedAt = DateTime.UtcNow;
            product.IsActive = true;
            
            await _productRepository.AddAsync(product);
            await _productRepository.SaveChangesAsync();
            return product;
        }

        public async Task UpdateProductAsync(Product product)
        {
            var existingProduct = await _productRepository.GetByIdAsync(product.Id);
            if (existingProduct == null)
            {
                throw new KeyNotFoundException($"Product with ID {product.Id} not found.");
            }

            existingProduct.Name = product.Name;
            existingProduct.Category = product.Category;
            existingProduct.UpdatedAt = DateTime.UtcNow;

            await _productRepository.UpdateAsync(existingProduct);
            await _productRepository.SaveChangesAsync();
        }

        public async Task DeactivateProductAsync(int id)
        {
            var product = await _productRepository.GetByIdAsync(id);
            if (product == null)
            {
                throw new KeyNotFoundException($"Product with ID {id} not found.");
            }

            product.IsActive = false;
            product.UpdatedAt = DateTime.UtcNow;
            
            await _productRepository.UpdateAsync(product);
            await _productRepository.SaveChangesAsync();
        }
    }
}

