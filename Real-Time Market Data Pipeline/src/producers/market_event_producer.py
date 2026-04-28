import json
import uuid
import random
import time
import argparse
from datetime import datetime
from confluent_kafka import Producer

SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BTC', 'ETH']
EXCHANGES = ['NASDAQ', 'NYSE', 'BINANCE', 'COINBASE']

def get_delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        pass # Successful delivery

def generate_event():
    symbol = random.choice(SYMBOLS)
    exchange = random.choice(EXCHANGES)
    price = round(random.uniform(10.0, 1000.0), 2)
    volume = round(random.uniform(1.0, 100.0), 4)
    timestamp = datetime.utcnow().isoformat()
    event_id = str(uuid.uuid4())
    
    # Introduce occasional data quality issues for testing
    if random.random() < 0.001:
        price = -1.0 # Negative price
    if random.random() < 0.001:
        event_id = None # Null event ID

    return {
        "event_id": event_id,
        "symbol": symbol,
        "exchange": exchange,
        "price": price,
        "volume": volume,
        "timestamp": timestamp
    }

def main(num_records, delay):
    conf = {'bootstrap.servers': 'kafka:29092'} # Internal docker network
    producer = Producer(conf)
    topic = 'market_events'

    print(f"Generating {num_records} records to topic '{topic}'...")

    for i in range(num_records):
        event = generate_event()
        producer.produce(
            topic, 
            key=event.get('symbol', 'UNKNOWN'), 
            value=json.dumps(event),
            callback=get_delivery_report
        )
        
        if (i+1) % 10000 == 0:
            print(f"Produced {i+1} messages...")
            producer.poll(0)

        if delay > 0:
            time.sleep(delay)

    producer.flush()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic market data")
    parser.add_argument("--records", type=int, default=1000, help="Number of records to generate")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between records in seconds")
    args = parser.parse_args()
    
    main(args.records, args.delay)
