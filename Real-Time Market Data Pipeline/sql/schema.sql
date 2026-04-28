CREATE TABLE IF NOT EXISTS dim_symbol (
    symbol_id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    company_name VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS dim_exchange (
    exchange_id SERIAL PRIMARY KEY,
    exchange_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_id INT PRIMARY KEY,
    full_date DATE NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    day INT NOT NULL,
    day_of_week INT NOT NULL,
    is_weekend BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_market_events (
    event_id VARCHAR(50) PRIMARY KEY,
    symbol_id INT REFERENCES dim_symbol(symbol_id),
    exchange_id INT REFERENCES dim_exchange(exchange_id),
    date_id INT REFERENCES dim_date(date_id),
    timestamp TIMESTAMP NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    volume DECIMAL(18, 8) NOT NULL,
    notional_value DECIMAL(18, 8) NOT NULL
);
