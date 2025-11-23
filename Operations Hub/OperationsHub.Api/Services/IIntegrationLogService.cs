using OperationsHub.Api.Models.Entities;

namespace OperationsHub.Api.Services
{
    public interface IIntegrationLogService
    {
        Task<int> StartJobAsync(string jobType);
        Task CompleteJobAsync(int jobId, string status, string? details = null);
        Task LogInfoAsync(int jobId, string message);
        Task LogWarningAsync(int jobId, string message);
        Task LogErrorAsync(int jobId, string message, Exception? ex = null);
    }
}

