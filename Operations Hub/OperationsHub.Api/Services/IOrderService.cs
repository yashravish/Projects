using OperationsHub.Api.Models.Entities;

namespace OperationsHub.Api.Services
{
    public interface IOrderService
    {
        Task<CustomerOrder> CreateOrderAsync(CustomerOrder order);
        Task AllocateOrderAsync(int orderId);
        Task ShipOrderAsync(int orderId);
        Task<CustomerOrder?> GetOrderDetailsAsync(int orderId);
    }
}

