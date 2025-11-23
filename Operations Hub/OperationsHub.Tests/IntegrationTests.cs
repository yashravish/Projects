using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http.Json;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using OperationsHub.Api.Data;
using OperationsHub.Api.Models.DTOs;
using OperationsHub.Api.Models.Entities;
using Xunit;
using Xunit.Abstractions;

namespace OperationsHub.Tests
{
    public class IntegrationTests : IClassFixture<WebApplicationFactory<Program>>
    {
        private readonly WebApplicationFactory<Program> _factory;
        private readonly HttpClient _client;
        private readonly ITestOutputHelper _output;

        public IntegrationTests(WebApplicationFactory<Program> factory, ITestOutputHelper output)
        {
            _output = output;
            _factory = factory.WithWebHostBuilder(builder =>
            {
                builder.ConfigureTestServices(services =>
                {
                    var descriptor = services.SingleOrDefault(
                        d => d.ServiceType == typeof(DbContextOptions<ApplicationDbContext>));

                    if (descriptor != null)
                    {
                        services.Remove(descriptor);
                    }

                    services.AddDbContext<ApplicationDbContext>(options =>
                    {
                        options.UseInMemoryDatabase("OperationsHubTestDb");
                    });
                });
            });

            _client = _factory.CreateClient();
        }

        [Fact]
        public async Task GetProducts_ReturnsSuccess()
        {
            var response = await _client.GetAsync("/api/products");
            if (response.StatusCode != HttpStatusCode.OK)
            {
                var content = await response.Content.ReadAsStringAsync();
                _output.WriteLine($"Error: {content}");
            }
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task GetWarehouses_ReturnsSuccess()
        {
            var response = await _client.GetAsync("/api/warehouses");
             if (response.StatusCode != HttpStatusCode.OK)
            {
                var content = await response.Content.ReadAsStringAsync();
                _output.WriteLine($"Error: {content}");
            }
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task MockErp_GetProducts_ReturnsData()
        {
            var response = await _client.GetAsync("/mock-erp/products");
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task Reports_Sales_ReturnsSuccess()
        {
            var response = await _client.GetAsync("/api/reports/sales?start=2024-01-01");
             if (response.StatusCode != HttpStatusCode.OK)
            {
                var content = await response.Content.ReadAsStringAsync();
                _output.WriteLine($"Error: {content}");
            }
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }
    }
}
