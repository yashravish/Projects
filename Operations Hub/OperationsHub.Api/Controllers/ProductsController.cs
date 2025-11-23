using Microsoft.AspNetCore.Mvc;
using OperationsHub.Api.Models.DTOs;
using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Services;

namespace OperationsHub.Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ProductsController : ControllerBase
    {
        private readonly IProductService _productService;

        public ProductsController(IProductService productService)
        {
            _productService = productService;
        }

        /// <summary>
        /// Retrieves all products with their basic details and total stock.
        /// </summary>
        [HttpGet]
        public async Task<ActionResult<IEnumerable<ProductDetailDto>>> GetProducts()
        {
            var products = await _productService.GetProductsAsync();
            var dtos = products.Select(p => new ProductDetailDto
            {
                Id = p.Id,
                Sku = p.Sku,
                Name = p.Name,
                Category = p.Category,
                IsActive = p.IsActive,
                TotalStock = p.InventoryBalances.Sum(b => b.Quantity),
                CreatedAt = p.CreatedAt
            });
            return Ok(dtos);
        }

        /// <summary>
        /// Retrieves a specific product by ID.
        /// </summary>
        [HttpGet("{id}")]
        public async Task<ActionResult<ProductDetailDto>> GetProduct(int id)
        {
            var product = await _productService.GetProductByIdAsync(id);
            if (product == null) return NotFound();

            var dto = new ProductDetailDto
            {
                Id = product.Id,
                Sku = product.Sku,
                Name = product.Name,
                Category = product.Category,
                IsActive = product.IsActive,
                TotalStock = product.InventoryBalances.Sum(b => b.Quantity),
                CreatedAt = product.CreatedAt
            };
            return Ok(dto);
        }

        /// <summary>
        /// Creates a new product.
        /// </summary>
        [HttpPost]
        public async Task<ActionResult<ProductDetailDto>> CreateProduct(CreateProductDto dto)
        {
            try
            {
                var product = new Product
                {
                    Sku = dto.Sku,
                    Name = dto.Name,
                    Category = dto.Category
                };

                var created = await _productService.CreateProductAsync(product);

                var responseDto = new ProductDetailDto
                {
                    Id = created.Id,
                    Sku = created.Sku,
                    Name = created.Name,
                    Category = created.Category,
                    IsActive = created.IsActive,
                    TotalStock = 0,
                    CreatedAt = created.CreatedAt
                };

                return CreatedAtAction(nameof(GetProduct), new { id = created.Id }, responseDto);
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(ex.Message);
            }
        }

        /// <summary>
        /// Updates an existing product.
        /// </summary>
        [HttpPut("{id}")]
        public async Task<IActionResult> UpdateProduct(int id, UpdateProductDto dto)
        {
            try
            {
                var product = new Product
                {
                    Id = id,
                    Name = dto.Name,
                    Category = dto.Category
                };

                await _productService.UpdateProductAsync(product);
                return NoContent();
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
        }

        /// <summary>
        /// Soft deletes a product by setting IsActive to false.
        /// </summary>
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteProduct(int id)
        {
            try
            {
                await _productService.DeactivateProductAsync(id);
                return NoContent();
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
        }
    }
}

