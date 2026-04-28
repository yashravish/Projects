import json
import uuid
import sys
import argparse
from datetime import datetime
from confluent_kafka import Consumer, KafkaError
import boto3

def get_minio_client():
    return boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
        region_name='us-east-1'
    )

def main(max_records=None):
    conf = {
        'bootstrap.servers': 'kafka:29092',
        'group.id': 'minio_writer_group',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(conf)
    topic = 'market_events'
    consumer.subscribe([topic])

    s3_client = get_minio_client()
    bucket_name = 'raw-market-data'

    print(f"Starting consumer, reading from '{topic}' and writing to MinIO '{bucket_name}'...")

    batch = []
    batch_size = 5000
    total_processed = 0

    try:
        while True:
            msg = consumer.poll(timeout=2.0)
            if msg is None:
                # If we've processed everything and there are no new messages
                if max_records and total_processed >= max_records:
                    break
                print("No new messages, waiting...")
                # In batch mode, we can break if no messages for a while to let Airflow proceed
                # For this demo, let's break after 10s of no messages
                break 
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(msg.error())
                    break
            
            data = json.loads(msg.value().decode('utf-8'))
            batch.append(data)
            total_processed += 1

            if len(batch) >= batch_size:
                # Write batch to minio
                now = datetime.utcnow()
                prefix = f"year={now.year}/month={now.month:02d}/day={now.day:02d}/hour={now.hour:02d}"
                file_name = f"{prefix}/events_{uuid.uuid4()}.json"
                
                body = "\n".join([json.dumps(r) for r in batch])
                s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=body)
                print(f"Wrote {len(batch)} records to s3://{bucket_name}/{file_name}")
                batch.clear()

            if max_records and total_processed >= max_records:
                break

    except KeyboardInterrupt:
        pass
    finally:
        # write remaining
        if len(batch) > 0:
            now = datetime.utcnow()
            prefix = f"year={now.year}/month={now.month:02d}/day={now.day:02d}/hour={now.hour:02d}"
            file_name = f"{prefix}/events_{uuid.uuid4()}.json"
            body = "\n".join([json.dumps(r) for r in batch])
            s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=body)
            print(f"Wrote remaining {len(batch)} records to s3://{bucket_name}/{file_name}")

        consumer.close()
        print(f"Total processed: {total_processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=None, help="Maximum records to consume")
    args = parser.parse_args()
    main(args.max_records)
