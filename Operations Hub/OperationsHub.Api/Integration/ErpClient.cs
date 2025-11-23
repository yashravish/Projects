using System.Text.Json;
using OperationsHub.Api.Integration.DTOs;

namespace OperationsHub.Api.Integration
{
    public interface IErpClient
    {
        Task<List<ErpProductDto>> GetProductsAsync();
        Task<List<ErpOrderDto>> GetOrdersAsync();
        Task ConfirmShipmentAsync(string erpOrderId);
    }

    public class ErpClient : IErpClient
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<ErpClient> _logger;

        public ErpClient(HttpClient httpClient, IConfiguration configuration, ILogger<ErpClient> logger)
        {
            _httpClient = httpClient;
            _logger = logger;
            
            var baseUrl = configuration["Integration:ErpBaseUrl"];
            if (!string.IsNullOrEmpty(baseUrl))
            {
                _httpClient.BaseAddress = new Uri(baseUrl);
            }
        }

        public async Task<List<ErpProductDto>> GetProductsAsync()
        {
            try 
            {
                // In a real app, we'd call the external API
                // For this mock setup hosted in the SAME app, we might need to be careful about self-calls during startup/dev
                // However, requirements say "Integration layer that talks to a mock ERP API (which we also host inside this app)"
                
                // NOTE: If running in same process, standard HttpClient call to localhost might fail if port isn't known or certificate issues.
                // For robustness in this specific demo environment, we might want to hardcode the relative path if BaseAddress is set correctly.
                
                var response = await _httpClient.GetAsync("/mock-erp/products");
                response.EnsureSuccessStatusCode();
                
                var content = await response.Content.ReadAsStringAsync();
                var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                return JsonSerializer.Deserialize<List<ErpProductDto>>(content, options) ?? new List<ErpProductDto>();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching products from ERP");
                throw;
            }
        }

        public async Task<List<ErpOrderDto>> GetOrdersAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync("/mock-erp/orders");
                response.EnsureSuccessStatusCode();

                var content = await response.Content.ReadAsStringAsync();
                var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                return JsonSerializer.Deserialize<List<ErpOrderDto>>(content, options) ?? new List<ErpOrderDto>();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching orders from ERP");
                throw;
            }
        }

        public async Task ConfirmShipmentAsync(string erpOrderId)
        {
             try
            {
                var response = await _httpClient.PostAsync($"/mock-erp/orders/{erpOrderId}/ship-confirm", null);
                response.EnsureSuccessStatusCode();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error confirming shipment to ERP for Order {OrderId}", erpOrderId);
                throw;
            }
        }
    }
}
