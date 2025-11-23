using OperationsHub.Api.Models.Entities;

namespace OperationsHub.Api.Services
{
    public interface IProductService
    {
        Task<IEnumerable<Product>> GetProductsAsync();
        Task<Product?> GetProductByIdAsync(int id);
        Task<Product> CreateProductAsync(Product product);
        Task UpdateProductAsync(Product product);
        Task DeactivateProductAsync(int id);
    }
}

