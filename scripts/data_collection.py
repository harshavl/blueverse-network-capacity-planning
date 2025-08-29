import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from kafka import KafkaProducer
import json

def collect_network_data(output_path="data/raw/network_data.csv", kafka_topic="network_data", kafka_bootstrap_servers="localhost:9092"):
    n_rows = 500  # Reduced dataset size
    dates = [datetime.now() - timedelta(days=x) for x in range(n_rows)]
    time = np.arange(n_rows)
    
    base_bandwidth = np.random.normal(230, 15, n_rows)
    trend = 0.4 * time
    spikes = 110 * np.sin(2 * np.pi * time / 4)
    bandwidth = np.clip(base_bandwidth + trend + spikes, 0, 1000)
    
    data = {
        "timestamp": dates,
        "router_id": ["router_1"] * n_rows,
        "bandwidth_mbps": bandwidth,
        "location": ["datacenter_1"] * n_rows,
        "application_type": ["web"] * n_rows
    }
    df = pd.DataFrame(data)
    
    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        for _, row in df.iterrows():
            producer.send(kafka_topic, row.to_dict())
        producer.flush()
        print(f"Data published to Kafka topic: {kafka_topic}")
    except Exception as e:
        print(f"Kafka error: {e}. Falling back to CSV.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print("\nrouter_1 (first 5 rows):")
    print(df[["timestamp", "bandwidth_mbps"]].head())

if __name__ == "__main__":
    collect_network_data()