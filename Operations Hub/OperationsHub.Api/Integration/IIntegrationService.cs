namespace OperationsHub.Api.Integration
{
    public interface IIntegrationService
    {
        Task SyncProductsFromErpAsync(int jobId);
        Task SyncOrdersFromErpAsync(int jobId);
    }
}
