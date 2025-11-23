using Microsoft.AspNetCore.Mvc;
using OperationsHub.Api.Models.DTOs;
using OperationsHub.Api.Services;

namespace OperationsHub.Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ConfigurationController : ControllerBase
    {
        private readonly IConfigurationService _configService;

        public ConfigurationController(IConfigurationService configService)
        {
            _configService = configService;
        }

        [HttpGet("reorder-thresholds")]
        public async Task<ActionResult<IEnumerable<ReorderThresholdDto>>> GetThresholds()
        {
            var result = await _configService.GetReorderThresholdsAsync();
            return Ok(result);
        }

        [HttpPost("reorder-thresholds")]
        public async Task<ActionResult<ReorderThresholdDto>> SetThreshold(CreateReorderThresholdDto dto)
        {
            var result = await _configService.SetReorderThresholdAsync(dto);
            return Ok(result);
        }

        [HttpGet("replenishment-suggestions")]
        public async Task<ActionResult<IEnumerable<ReplenishmentSuggestionDto>>> GetSuggestions()
        {
            var result = await _configService.GetReplenishmentSuggestionsAsync();
            return Ok(result);
        }

        [HttpPost("replenishment-suggestions/{id}/status")]
        public async Task<IActionResult> UpdateSuggestionStatus(int id, [FromBody] string status)
        {
            // Basic validation
            if (status != "Open" && status != "Reviewed" && status != "Dismissed")
                return BadRequest("Invalid status. Allowed: Open, Reviewed, Dismissed");

            await _configService.UpdateSuggestionStatusAsync(id, status);
            return NoContent();
        }
    }
}

