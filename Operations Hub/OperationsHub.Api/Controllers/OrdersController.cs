using Microsoft.AspNetCore.Mvc;
using OperationsHub.Api.Models.DTOs;
using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Repositories;
using OperationsHub.Api.Services;

namespace OperationsHub.Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class OrdersController : ControllerBase
    {
        private readonly IOrderService _orderService;
        private readonly IRepository<CustomerOrder> _orderRepository;

        public OrdersController(IOrderService orderService, IRepository<CustomerOrder> orderRepository)
        {
            _orderService = orderService;
            _orderRepository = orderRepository;
        }

        /// <summary>
        /// Retrieves all orders (basic summary).
        /// </summary>
        [HttpGet]
        public async Task<ActionResult<IEnumerable<OrderDto>>> GetOrders()
        {
            var orders = await _orderRepository.GetAllAsync();
            var dtos = orders.Select(o => new OrderDto
            {
                Id = o.Id,
                CustomerName = o.CustomerName,
                Region = o.Region,
                Status = o.Status,
                ExternalOrderNumber = o.ExternalOrderNumber,
                OrderDate = o.OrderDate,
                // Lines not loaded for list view performance usually, but here keeping it simple
            });
            return Ok(dtos);
        }

        /// <summary>
        /// Retrieves full details for a specific order.
        /// </summary>
        [HttpGet("{id}")]
        public async Task<ActionResult<OrderDto>> GetOrder(int id)
        {
            var order = await _orderService.GetOrderDetailsAsync(id);
            if (order == null) return NotFound();

            var dto = new OrderDto
            {
                Id = order.Id,
                CustomerName = order.CustomerName,
                Region = order.Region,
                Status = order.Status,
                ExternalOrderNumber = order.ExternalOrderNumber,
                OrderDate = order.OrderDate,
                Lines = order.OrderLines.Select(l => new OrderLineDto
                {
                    Id = l.Id,
                    ProductId = l.ProductId,
                    ProductName = l.Product?.Name ?? "Unknown",
                    Quantity = l.Quantity,
                    AllocatedWarehouseId = l.AllocatedWarehouseId,
                    AllocatedWarehouseName = l.AllocatedWarehouse?.Name,
                    ShippedQuantity = l.ShippedQuantity,
                    LineStatus = l.LineStatus
                }).ToList()
            };
            return Ok(dto);
        }

        /// <summary>
        /// Creates a new customer order.
        /// </summary>
        [HttpPost]
        public async Task<ActionResult<OrderDto>> CreateOrder(CreateOrderDto dto)
        {
            var order = new CustomerOrder
            {
                CustomerName = dto.CustomerName,
                Region = dto.Region,
                ExternalOrderNumber = dto.ExternalOrderNumber,
                OrderDate = DateTime.UtcNow,
                OrderLines = dto.Lines.Select(l => new OrderLine
                {
                    ProductId = l.ProductId,
                    Quantity = l.Quantity
                }).ToList()
            };

            var created = await _orderService.CreateOrderAsync(order);
            
            // Refetch to get details or construct simplistic response
            // Constructing simple response to avoid round trip if navigation props not loaded
             var responseDto = new OrderDto
            {
                Id = created.Id,
                CustomerName = created.CustomerName,
                Region = created.Region,
                Status = created.Status,
                ExternalOrderNumber = created.ExternalOrderNumber,
                OrderDate = created.OrderDate,
                 Lines = created.OrderLines.Select(l => new OrderLineDto
                {
                    Id = l.Id,
                    ProductId = l.ProductId,
                    Quantity = l.Quantity,
                    LineStatus = l.LineStatus
                }).ToList()
            };

            return CreatedAtAction(nameof(GetOrder), new { id = created.Id }, responseDto);
        }

        /// <summary>
        /// Allocates inventory for an order.
        /// </summary>
        [HttpPost("{id}/allocate")]
        public async Task<IActionResult> AllocateOrder(int id)
        {
            try
            {
                await _orderService.AllocateOrderAsync(id);
                return Ok(new { Message = "Order allocation process completed." });
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(ex.Message);
            }
        }

        /// <summary>
        /// Ships an allocated order.
        /// </summary>
        [HttpPost("{id}/ship")]
        public async Task<IActionResult> ShipOrder(int id)
        {
            try
            {
                await _orderService.ShipOrderAsync(id);
                return Ok(new { Message = "Order shipped successfully." });
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(ex.Message);
            }
        }
    }
}

