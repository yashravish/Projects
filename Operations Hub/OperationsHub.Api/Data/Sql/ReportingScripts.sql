-- View: v_SalesByProductAndRegion
-- Deployment: Run manually or via Migration .Sql() command

CREATE OR ALTER VIEW v_SalesByProductAndRegion AS
SELECT 
    FORMAT(o.OrderDate, 'yyyy-MM') AS [Month],
    p.Name AS ProductName,
    p.Category,
    o.Region,
    SUM(ol.Quantity) AS TotalQuantity
FROM OrderLines ol
JOIN CustomerOrders o ON ol.OrderId = o.Id
JOIN Products p ON ol.ProductId = p.Id
WHERE ol.LineStatus = 'Shipped'
GROUP BY FORMAT(o.OrderDate, 'yyyy-MM'), p.Name, p.Category, o.Region;
GO

-- View: v_InventoryAging
-- Deployment: Run manually or via Migration .Sql() command

CREATE OR ALTER VIEW v_InventoryAging AS
SELECT 
    p.Name AS ProductName,
    b.BatchNumber,
    'Global' AS WarehouseName, -- Limitation of current schema
    DATEDIFF(day, GETDATE(), b.ExpiryDate) AS DaysToExpiry,
    b.RemainingQuantity
FROM Batches b
JOIN Products p ON b.ProductId = p.Id
WHERE b.RemainingQuantity > 0;
GO

-- Stored Procedure: usp_CalculateFillRate
-- Deployment: Run manually or via Migration .Sql() command

CREATE OR ALTER PROCEDURE usp_CalculateFillRate
    @StartDate DATETIME2,
    @EndDate DATETIME2
AS
BEGIN
    SET NOCOUNT ON;

    SELECT 
        p.Name AS ProductName,
        SUM(ol.Quantity) AS TotalOrdered,
        SUM(ol.ShippedQuantity) AS TotalShipped,
        CASE 
            WHEN SUM(ol.Quantity) = 0 THEN 0 
            ELSE CAST(SUM(ol.ShippedQuantity) AS FLOAT) / SUM(ol.Quantity) * 100 
        END AS FillRatePercent
    FROM OrderLines ol
    JOIN CustomerOrders o ON ol.OrderId = o.Id
    JOIN Products p ON ol.ProductId = p.Id
    WHERE o.OrderDate >= @StartDate AND o.OrderDate <= @EndDate
    GROUP BY p.Name
    ORDER BY FillRatePercent ASC;
END
GO

