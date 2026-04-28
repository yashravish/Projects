import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_date, year, month, dayofmonth, dayofweek, date_format, lit
from pyspark.sql.types import TimestampType

def main():
    # Minio settings
    os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

    spark = SparkSession.builder \
        .appName("TransformMarketData") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.postgresql:postgresql:42.6.0") \
        .getOrCreate()

    # Read from MinIO
    df = spark.read.json("s3a://raw-market-data/year=*/month=*/day=*/hour=*/*.json")
    
    # 1. Clean and validate records
    df = df.dropna(subset=['event_id', 'symbol', 'price', 'volume', 'timestamp'])
    
    # Filter negative price/volume
    df = df.filter((col("price") >= 0) & (col("volume") >= 0))
    
    # Remove duplicates by event_id
    df = df.dropDuplicates(["event_id"])
    
    # 2. Derive fields
    df = df.withColumn("notional_value", col("price") * col("volume"))
    df = df.withColumn("timestamp", col("timestamp").cast(TimestampType()))
    
    df = df.withColumn("date_only", to_date(col("timestamp")))
    df = df.withColumn("date_id", date_format(col("date_only"), "yyyyMMdd").cast("integer"))
    
    # Create Dimensions
    dim_symbol = df.select("symbol").distinct() \
        .withColumn("company_name", lit(None).cast("string"))
        
    dim_exchange = df.select("exchange").withColumnRenamed("exchange", "exchange_name").distinct()
    
    dim_date = df.select("date_only", "date_id").distinct() \
        .withColumnRenamed("date_only", "full_date") \
        .withColumn("year", year(col("full_date"))) \
        .withColumn("month", month(col("full_date"))) \
        .withColumn("day", dayofmonth(col("full_date"))) \
        .withColumn("day_of_week", dayofweek(col("full_date"))) \
        .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), True).otherwise(False))
    
    # Connect to Postgres
    jdbc_url = "jdbc:postgresql://postgres:5432/data_warehouse"
    db_properties = {
        "user": "postgres",
        "password": "postgres",
        "driver": "org.postgresql.Driver"
    }

    # Write to staging tables
    dim_date.write.jdbc(url=jdbc_url, table="stg_dim_date", mode="overwrite", properties=db_properties)
    dim_symbol.write.jdbc(url=jdbc_url, table="stg_dim_symbol", mode="overwrite", properties=db_properties)
    dim_exchange.write.jdbc(url=jdbc_url, table="stg_dim_exchange", mode="overwrite", properties=db_properties)
    
    stg_fact = df.select(
        "event_id", "symbol", "exchange", "date_id", "timestamp", "price", "volume", "notional_value"
    )
    stg_fact.write.jdbc(url=jdbc_url, table="stg_fact_market_events", mode="overwrite", properties=db_properties)
    
    print("Successfully wrote staging tables to PostgreSQL.")
    spark.stop()

if __name__ == "__main__":
    main()
