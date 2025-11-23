using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using System;
using System.Threading;
using System.Threading.Tasks;
using OperationsHub.Api.Integration;
using OperationsHub.Api.Services;

namespace OperationsHub.Api.BackgroundJobs
{
    public class ErpSyncJob : BackgroundService
    {
        private readonly IServiceProvider _serviceProvider;
        private readonly ILogger<ErpSyncJob> _logger;
        private readonly int _syncIntervalMinutes;

        public ErpSyncJob(
            IServiceProvider serviceProvider,
            IConfiguration configuration,
            ILogger<ErpSyncJob> logger)
        {
            _serviceProvider = serviceProvider;
            _logger = logger;
            _syncIntervalMinutes = configuration.GetValue<int>("Integration:SyncIntervalMinutes", 60);
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation($"ERP Sync Job started. Interval: {_syncIntervalMinutes} minutes.");

            while (!stoppingToken.IsCancellationRequested)
            {
                _logger.LogInformation("ERP Sync Job: Wake up.");

                try
                {
                    using (var scope = _serviceProvider.CreateScope())
                    {
                        var logService = scope.ServiceProvider.GetRequiredService<IIntegrationLogService>();
                        var integrationService = scope.ServiceProvider.GetRequiredService<IIntegrationService>();

                        // 1. Start Job Record
                        int jobId = await logService.StartJobAsync("ERP_SYNC");

                        try
                        {
                            // 2. Run Syncs
                            await integrationService.SyncProductsFromErpAsync(jobId);
                            await integrationService.SyncOrdersFromErpAsync(jobId);

                            // 3. Mark Success
                            await logService.CompleteJobAsync(jobId, "Succeeded");
                        }
                        catch (Exception ex)
                        {
                            _logger.LogError(ex, "ERP Sync Job execution failed internally.");
                            // 4. Mark Failure
                            await logService.CompleteJobAsync(jobId, "Failed", ex.Message);
                        }
                    }
                }
                catch (Exception ex)
                {
                    // Catch failures in creating scope or resolving services
                    _logger.LogError(ex, "Critical error in ERP Sync Job loop.");
                }

                // Wait for next interval
                await Task.Delay(TimeSpan.FromMinutes(_syncIntervalMinutes), stoppingToken);
            }
        }
    }
}
