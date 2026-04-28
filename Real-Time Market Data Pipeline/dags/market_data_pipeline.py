from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'batch_ingest_market_data',
    default_args=default_args,
    description='End-to-end Market Data Pipeline',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags=['market_data', 'etl'],
) as dag:

    # 1. Create schemas
    create_schemas = PostgresOperator(
        task_id='create_tables',
        postgres_conn_id='data_warehouse',
        sql='''
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
        '''
    )

    # 2. Ingest to MinIO (Simulate batch read from Kafka for the DAG)
    ingest_to_minio = BashOperator(
        task_id='ingest_to_minio',
        bash_command='python /opt/airflow/src/consumers/minio_writer.py --max-records 1000'
    )

    # 3. PySpark Transformation
    spark_transform = BashOperator(
        task_id='spark_transform',
        # Install java to be able to run pyspark locally in the airflow container, or we submit to remote spark
        # For this setup, we submit to the standalone spark cluster we created in docker-compose
        bash_command='''
        pip install pyspark==3.5.0 && \
        spark-submit \
        --master spark://spark-master:7077 \
        --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.postgresql:postgresql:42.6.0 \
        /opt/airflow/spark_jobs/transform_market_data.py
        '''
    )

    # 4. Load from staging to data warehouse
    load_warehouse = PostgresOperator(
        task_id='load_warehouse',
        postgres_conn_id='data_warehouse',
        sql='''
        -- Upsert dim_date
        INSERT INTO dim_date (date_id, full_date, year, month, day, day_of_week, is_weekend)
        SELECT date_id, full_date, year, month, day, day_of_week, is_weekend FROM stg_dim_date
        ON CONFLICT (date_id) DO NOTHING;

        -- Upsert dim_symbol
        INSERT INTO dim_symbol (symbol, company_name)
        SELECT symbol, company_name FROM stg_dim_symbol
        ON CONFLICT (symbol) DO NOTHING;

        -- Upsert dim_exchange
        INSERT INTO dim_exchange (exchange_name)
        SELECT exchange_name FROM stg_dim_exchange
        ON CONFLICT (exchange_name) DO NOTHING;

        -- Insert into fact_market_events
        INSERT INTO fact_market_events (event_id, symbol_id, exchange_id, date_id, timestamp, price, volume, notional_value)
        SELECT 
            f.event_id, 
            s.symbol_id, 
            e.exchange_id, 
            f.date_id, 
            f.timestamp, 
            f.price, 
            f.volume, 
            f.notional_value
        FROM stg_fact_market_events f
        JOIN dim_symbol s ON f.symbol = s.symbol
        JOIN dim_exchange e ON f.exchange = e.exchange_name
        ON CONFLICT (event_id) DO NOTHING;
        '''
    )

    # 5. Data Quality Checks
    data_quality_checks = PostgresOperator(
        task_id='data_quality_checks',
        postgres_conn_id='data_warehouse',
        sql='''
        DO $$
        DECLARE
            null_count INT;
            neg_price INT;
            dup_count INT;
        BEGIN
            -- Check for null event_ids
            SELECT COUNT(*) INTO null_count FROM fact_market_events WHERE event_id IS NULL;
            IF null_count > 0 THEN
                RAISE EXCEPTION 'Data Quality Check Failed: Found % null event_ids', null_count;
            END IF;

            -- Check for negative prices
            SELECT COUNT(*) INTO neg_price FROM fact_market_events WHERE price < 0;
            IF neg_price > 0 THEN
                RAISE EXCEPTION 'Data Quality Check Failed: Found % negative prices', neg_price;
            END IF;
            
            -- Ensure no duplicate event_ids
            SELECT COUNT(*) INTO dup_count FROM (
                SELECT event_id FROM fact_market_events GROUP BY event_id HAVING COUNT(*) > 1
            ) sub;
            IF dup_count > 0 THEN
                RAISE EXCEPTION 'Data Quality Check Failed: Found % duplicate event_ids', dup_count;
            END IF;
        END $$;
        '''
    )

    # 6. Apply Indexes 
    apply_indexes = PostgresOperator(
        task_id='apply_indexes',
        postgres_conn_id='data_warehouse',
        sql='''
        CREATE INDEX IF NOT EXISTS idx_fact_timestamp ON fact_market_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_fact_symbol_id ON fact_market_events(symbol_id);
        CREATE INDEX IF NOT EXISTS idx_fact_exchange_id ON fact_market_events(exchange_id);
        CREATE INDEX IF NOT EXISTS idx_fact_date_id ON fact_market_events(date_id);
        CREATE INDEX IF NOT EXISTS idx_fact_symbol_timestamp ON fact_market_events(symbol_id, timestamp);
        '''
    )

    create_schemas >> ingest_to_minio >> spark_transform >> load_warehouse >> data_quality_checks >> apply_indexes
