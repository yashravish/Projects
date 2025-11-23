using Microsoft.AspNetCore.Mvc;
using OperationsHub.Api.Services;

namespace OperationsHub.Api.Controllers
{
    [Route("api/reports")]
    [ApiController]
    public class ReportsController : ControllerBase
    {
        private readonly IReportingService _reportingService;

        public ReportsController(IReportingService reportingService)
        {
            _reportingService = reportingService;
        }

        [HttpGet("sales")]
        public async Task<IActionResult> GetSales([FromQuery] DateTime? start, [FromQuery] DateTime? end)
        {
            var data = await _reportingService.GetSalesByProductAndRegionAsync(start, end);
            return Ok(data);
        }

        [HttpGet("inventory-aging")]
        public async Task<IActionResult> GetInventoryAging()
        {
            var data = await _reportingService.GetInventoryAgingAsync();
            return Ok(data);
        }

        [HttpGet("fill-rate")]
        public async Task<IActionResult> GetFillRate([FromQuery] DateTime? start, [FromQuery] DateTime? end)
        {
            // Default to last 30 days if not specified
            var s = start ?? DateTime.UtcNow.AddDays(-30);
            var e = end ?? DateTime.UtcNow;
            
            var data = await _reportingService.GetFillRateAsync(s, e);
            return Ok(data);
        }
    }
}

