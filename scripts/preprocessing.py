import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_path="data/raw/network_data.csv", output_path="data/processed/processed_network_data.csv"):
    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["router_id", "timestamp"])
    df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.days
    df["bandwidth_mbps"] = df["bandwidth_mbps"].astype(float)
    
    scaler = StandardScaler()
    df["bandwidth_mbps_scaled"] = scaler.fit_transform(df[["bandwidth_mbps"]])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    return df, scaler

if __name__ == "__main__":
    preprocess_data()
