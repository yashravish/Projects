CREATE INDEX IF NOT EXISTS idx_fact_timestamp ON fact_market_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_fact_symbol_id ON fact_market_events(symbol_id);
CREATE INDEX IF NOT EXISTS idx_fact_exchange_id ON fact_market_events(exchange_id);
CREATE INDEX IF NOT EXISTS idx_fact_date_id ON fact_market_events(date_id);
CREATE INDEX IF NOT EXISTS idx_fact_symbol_timestamp ON fact_market_events(symbol_id, timestamp);
