// dashboard/Controllers/DashboardController.cs
using Microsoft.AspNetCore.Mvc;
using FraudDashboard.Models;

namespace FraudDashboard.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class DashboardController : ControllerBase
    {
        [HttpPost("update")]
        public IActionResult UpdateDashboard([FromBody] TransactionData data)
        {
            // In a real application, this data would update the UI or trigger notifications.
            System.Console.WriteLine($"Dashboard updated with Transaction ID: {data.Id}, Amount: {data.Amount}, Risk Score: {data.RiskScore}");
            return Ok(new { message = "Dashboard updated successfully" });
        }
    }
}
