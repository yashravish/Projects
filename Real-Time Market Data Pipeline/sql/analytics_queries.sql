-- 1. Daily volume by symbol
SELECT 
    d.full_date,
    s.symbol,
    SUM(f.volume) as total_daily_volume
FROM fact_market_events f
JOIN dim_symbol s ON f.symbol_id = s.symbol_id
JOIN dim_date d ON f.date_id = d.date_id
GROUP BY 1, 2
ORDER BY 1 DESC, 3 DESC;

-- 2. Average price by symbol
SELECT
    s.symbol,
    AVG(f.price) as avg_price,
    MIN(f.price) as min_price,
    MAX(f.price) as max_price
FROM fact_market_events f
JOIN dim_symbol s ON f.symbol_id = s.symbol_id
GROUP BY 1
ORDER BY 1;

-- 3. Highest notional value trades
SELECT 
    f.event_id,
    s.symbol,
    e.exchange_name,
    f.timestamp,
    f.price,
    f.volume,
    f.notional_value
FROM fact_market_events f
JOIN dim_symbol s ON f.symbol_id = s.symbol_id
JOIN dim_exchange e ON f.exchange_id = e.exchange_id
ORDER BY f.notional_value DESC
LIMIT 100;

-- 4. Hourly market activity (transaction count)
SELECT
    DATE_TRUNC('hour', f.timestamp) as hour_bucket,
    COUNT(*) as total_trades,
    SUM(f.volume) as total_volume,
    SUM(f.notional_value) as total_notional
FROM fact_market_events f
GROUP BY 1
ORDER BY 1 DESC;

-- Query performance optimization example
-- EXPLAIN ANALYZE 
-- SELECT * FROM fact_market_events WHERE symbol_id = 1 AND timestamp >= '2023-01-01';
