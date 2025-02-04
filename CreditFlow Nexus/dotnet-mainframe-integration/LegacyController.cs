using Microsoft.AspNetCore.Mvc;
using System;

namespace MainframeIntegration.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class LegacyController : ControllerBase
    {
        // GET api/legacy/status
        [HttpGet("status")]
        public IActionResult GetStatus()
        {
            // Simulate a legacy mainframe status check.
            return Ok(new { status = "Legacy system operational", timestamp = DateTime.UtcNow });
        }
 
        // POST api/legacy/integrate
        [HttpPost("integrate")]
        public IActionResult IntegrateData([FromBody] object payload)
        {
            // Simulate data integration with a legacy mainframe.
            // In production, convert the incoming JSON or Protobuf data as required.
            return Ok(new { message = "Data integrated with legacy system", payload });
        }
    }
}
