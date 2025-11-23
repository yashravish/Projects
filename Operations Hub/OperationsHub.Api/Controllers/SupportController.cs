using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using OperationsHub.Api.Integration;
using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Repositories;
using OperationsHub.Api.Services;

namespace OperationsHub.Api.Controllers
{
    public class SupportController : Controller
    {
        private readonly IRepository<IntegrationJob> _jobRepository;
        private readonly IIntegrationLogService _logService;
        private readonly IIntegrationService _integrationService;

        public SupportController(
            IRepository<IntegrationJob> jobRepository,
            IIntegrationLogService logService,
            IIntegrationService integrationService)
        {
            _jobRepository = jobRepository;
            _logService = logService;
            _integrationService = integrationService;
        }

        [HttpGet("Support/Jobs")]
        public async Task<IActionResult> Jobs([FromQuery] string? jobType, [FromQuery] string? status, [FromQuery] int page = 1)
        {
            // Simple paging
            int pageSize = 20;
            
            var query = _jobRepository.Query();
            
            if (!string.IsNullOrEmpty(jobType)) query = query.Where(j => j.JobType == jobType);
            if (!string.IsNullOrEmpty(status)) query = query.Where(j => j.Status == status);

            var total = await query.CountAsync();
            var jobs = await query
                .OrderByDescending(j => j.StartedAt)
                .Skip((page - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();

            ViewData["Total"] = total;
            ViewData["Page"] = page;
            ViewData["CurrentStatus"] = status;
            ViewData["CurrentType"] = jobType;

            return View(jobs);
        }

        [HttpGet("Support/Jobs/{id}")]
        public async Task<IActionResult> JobDetails(int id)
        {
            var job = await _jobRepository.Query()
                .Include(j => j.Logs.OrderBy(l => l.CreatedAt))
                .FirstOrDefaultAsync(j => j.Id == id);

            if (job == null) return NotFound();

            return View(job);
        }

        [HttpPost("api/support/integration-jobs/{id}/replay")]
        public async Task<IActionResult> ReplayJob(int id)
        {
            var originalJob = await _jobRepository.GetByIdAsync(id);
            if (originalJob == null) return NotFound();

            // Create new job for replay
            var replayType = originalJob.JobType; // Could append REPLAY suffix, but usually we want to retry same logic
            int newJobId = await _logService.StartJobAsync(replayType + "_REPLAY");

            try
            {
                // Dispatch based on type (Hardcoded logic for this demo, could be strategy pattern)
                if (originalJob.JobType.Contains("ERP_SYNC") || originalJob.JobType == "ERP_SYNC")
                {
                     // Assuming generic sync job runs both products and orders
                     // If distinct jobs exist, split here.
                     await _integrationService.SyncProductsFromErpAsync(newJobId);
                     await _integrationService.SyncOrdersFromErpAsync(newJobId);
                }
                else if (originalJob.JobType == "ERP_PRODUCTS_SYNC")
                {
                    await _integrationService.SyncProductsFromErpAsync(newJobId);
                }
                else if (originalJob.JobType == "ERP_ORDERS_SYNC")
                {
                    await _integrationService.SyncOrdersFromErpAsync(newJobId);
                }
                else
                {
                    await _logService.LogWarningAsync(newJobId, "Unknown job type, no action taken.");
                }

                await _logService.CompleteJobAsync(newJobId, "Succeeded", $"Replay of Job #{id}");
                return Ok(new { Message = "Replay started successfully", NewJobId = newJobId });
            }
            catch (Exception ex)
            {
                await _logService.CompleteJobAsync(newJobId, "Failed", ex.Message);
                return StatusCode(500, new { Message = "Replay failed", Error = ex.Message });
            }
        }
    }
}

