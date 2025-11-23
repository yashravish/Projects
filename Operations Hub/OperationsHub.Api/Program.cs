using Microsoft.EntityFrameworkCore;
using OperationsHub.Api.Data;
using OperationsHub.Api.Services;
using OperationsHub.Api.Integration;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddHttpClient<IErpClient, ErpClient>(); // Typed Client Registration

builder.Services.AddControllersWithViews();

// EF Core Configuration
builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

// DI Registrations - Repositories
builder.Services.AddScoped(typeof(OperationsHub.Api.Repositories.IRepository<>), typeof(OperationsHub.Api.Repositories.EfRepository<>));

// DI Registrations - Services
builder.Services.AddScoped<IProductService, ProductService>();
builder.Services.AddScoped<IInventoryService, InventoryService>();
builder.Services.AddScoped<IOrderService, OrderService>();
builder.Services.AddScoped<IIntegrationLogService, IntegrationLogService>();
builder.Services.AddScoped<IIntegrationService, IntegrationService>();
builder.Services.AddScoped<IReportingService, ReportingService>();
builder.Services.AddScoped<IConfigurationService, ConfigurationService>();

// Background Jobs
builder.Services.AddHostedService<OperationsHub.Api.BackgroundJobs.ErpSyncJob>();

// Swagger/OpenAPI configuration
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(options =>
{
    // Only document API controllers (exclude MVC view controllers)
    options.DocInclusionPredicate((docName, apiDesc) =>
    {
        var controllerName = apiDesc.ActionDescriptor.RouteValues["controller"];
        // Only include controllers in the /api route
        return apiDesc.RelativePath?.StartsWith("api/") ?? false;
    });
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();

public partial class Program { }
