using Microsoft.EntityFrameworkCore;
using OperationsHub.Api.Data;
using OperationsHub.Api.Models.DTOs.Reports;

namespace OperationsHub.Api.Services
{
    public interface IReportingService
    {
        Task<IEnumerable<SalesByProductRegionDto>> GetSalesByProductAndRegionAsync(DateTime? start, DateTime? end);
        Task<IEnumerable<InventoryAgingDto>> GetInventoryAgingAsync();
        Task<IEnumerable<FillRateDto>> GetFillRateAsync(DateTime start, DateTime end);
    }

    public class ReportingService : IReportingService
    {
        private readonly ApplicationDbContext _context;

        public ReportingService(ApplicationDbContext context)
        {
            _context = context;
        }

        public async Task<IEnumerable<SalesByProductRegionDto>> GetSalesByProductAndRegionAsync(DateTime? start, DateTime? end)
        {
            // Implementation using LINQ to mimic the SQL View
            // In production, this would query the actual SQL View: _context.Database.SqlQuery<SalesByProductRegionDto>("SELECT * FROM v_SalesByProductAndRegion")
            
            var query = _context.OrderLines
                .Include(ol => ol.Order)
                .Include(ol => ol.Product)
                .Where(ol => ol.LineStatus == "Shipped");

            if (start.HasValue) query = query.Where(ol => ol.Order!.OrderDate >= start.Value);
            if (end.HasValue) query = query.Where(ol => ol.Order!.OrderDate <= end.Value);

            var grouped = await query
                .GroupBy(ol => new
                {
                    Month = ol.Order!.OrderDate.Year + "-" + ol.Order.OrderDate.Month,
                    ProductName = ol.Product!.Name,
                    Category = ol.Product.Category,
                    Region = ol.Order.Region
                })
                .Select(g => new SalesByProductRegionDto
                {
                    Month = g.Key.Month,
                    ProductName = g.Key.ProductName,
                    Category = g.Key.Category ?? "Uncategorized",
                    Region = g.Key.Region ?? "Unknown",
                    TotalQuantity = g.Sum(x => x.Quantity)
                })
                .ToListAsync();

            return grouped;
        }

        public async Task<IEnumerable<InventoryAgingDto>> GetInventoryAgingAsync()
        {
            // Mimicking v_InventoryAging
            // Note: As mentioned in DTO, we lack Batch-Warehouse link in this simplified schema.
            // We will report Global Batch Aging.
            
            var today = DateTime.UtcNow;

            var batches = await _context.Batches
                .Include(b => b.Product)
                .Where(b => b.RemainingQuantity > 0)
                .Select(b => new InventoryAgingDto
                {
                    ProductName = b.Product!.Name,
                    BatchNumber = b.BatchNumber,
                    WarehouseName = "Global/Mixed", // Placeholder as per schema limitation
                    DaysToExpiry = (int)(b.ExpiryDate - today).TotalDays,
                    RemainingQuantity = b.RemainingQuantity,
                    Status = b.Status
                })
                .ToListAsync();

            return batches.OrderBy(b => b.DaysToExpiry);
        }

        public async Task<IEnumerable<FillRateDto>> GetFillRateAsync(DateTime start, DateTime end)
        {
            // Mimicking usp_CalculateFillRate
            
            var data = await _context.OrderLines
                .Include(ol => ol.Product)
                .Include(ol => ol.Order)
                .Where(ol => ol.Order!.OrderDate >= start && ol.Order.OrderDate <= end)
                .GroupBy(ol => ol.Product!.Name)
                .Select(g => new
                {
                    ProductName = g.Key,
                    TotalOrdered = g.Sum(x => x.Quantity),
                    TotalShipped = g.Sum(x => x.ShippedQuantity)
                })
                .ToListAsync();

            return data.Select(d => new FillRateDto
            {
                ProductName = d.ProductName,
                TotalOrdered = d.TotalOrdered,
                TotalShipped = d.TotalShipped,
                FillRatePercent = d.TotalOrdered == 0 ? 0 : Math.Round((double)d.TotalShipped / d.TotalOrdered * 100, 2)
            }).OrderBy(x => x.FillRatePercent);
        }
    }
}

