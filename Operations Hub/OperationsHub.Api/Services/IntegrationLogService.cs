using OperationsHub.Api.Models.Entities;
using OperationsHub.Api.Repositories;

namespace OperationsHub.Api.Services
{
    public class IntegrationLogService : IIntegrationLogService
    {
        // Note: Since this service is often called from a Singleton HostedService scope,
        // we must be careful. Usually, we pass a transient/scoped repository or factory.
        // However, standard pattern is to let the caller manage the scope and pass the repository or 
        // instantiate this service within a scope.
        // Here, we'll design it to be Scoped, and the BackgroundJob will create a scope to resolve it.

        private readonly IRepository<IntegrationJob> _jobRepository;
        private readonly IRepository<IntegrationLog> _logRepository;

        public IntegrationLogService(
            IRepository<IntegrationJob> jobRepository,
            IRepository<IntegrationLog> logRepository)
        {
            _jobRepository = jobRepository;
            _logRepository = logRepository;
        }

        public async Task<int> StartJobAsync(string jobType)
        {
            var job = new IntegrationJob
            {
                JobType = jobType,
                Status = "Running",
                StartedAt = DateTime.UtcNow
            };

            await _jobRepository.AddAsync(job);
            await _jobRepository.SaveChangesAsync();
            return job.Id;
        }

        public async Task CompleteJobAsync(int jobId, string status, string? details = null)
        {
            var job = await _jobRepository.GetByIdAsync(jobId);
            if (job != null)
            {
                job.Status = status;
                job.FinishedAt = DateTime.UtcNow;
                if (details != null) job.Details = details;
                
                await _jobRepository.UpdateAsync(job);
                await _jobRepository.SaveChangesAsync();
            }
        }

        public async Task LogInfoAsync(int jobId, string message)
        {
            await AddLogAsync(jobId, "Info", message);
        }

        public async Task LogWarningAsync(int jobId, string message)
        {
            await AddLogAsync(jobId, "Warning", message);
        }

        public async Task LogErrorAsync(int jobId, string message, Exception? ex = null)
        {
            var fullMessage = message;
            if (ex != null)
            {
                fullMessage += $" | Exception: {ex.Message}";
            }
            await AddLogAsync(jobId, "Error", fullMessage);
        }

        private async Task AddLogAsync(int jobId, string level, string message)
        {
            var log = new IntegrationLog
            {
                JobId = jobId,
                Level = level,
                Message = message,
                CreatedAt = DateTime.UtcNow
            };

            await _logRepository.AddAsync(log);
            await _logRepository.SaveChangesAsync();
        }
    }
}

