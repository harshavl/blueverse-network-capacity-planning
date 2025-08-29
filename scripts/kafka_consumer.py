from kafka import KafkaConsumer
import pandas as pd
import json
import os
import time

def consume_network_data(output_path="data/raw/network_data.csv", kafka_topic="network_data", kafka_bootstrap_servers="localhost:9092"):
    consumer = KafkaConsumer(
        kafka_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='earliest',
        consumer_timeout_ms=10000
    )
    print(f"Consuming from Kafka topic: {kafka_topic}")
    data = []
    start_time = time.time()
    try:
        for message in consumer:
            record = message.value
            data.append(record)
            if len(data) >= 500:  # Adjusted for smaller dataset
                df = pd.DataFrame(data)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))
                print(f"Appended {len(data)} records to {output_path}")
                data = []
        if data:
            df = pd.DataFrame(data)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))
            print(f"Appended {len(data)} records to {output_path}")
    except Exception as e:
        print(f"Consumer error: {e}")
    finally:
        consumer.close()
        print(f"Consumer closed. Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    consume_network_data()