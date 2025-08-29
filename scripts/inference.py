import torch
import pandas as pd
import numpy as np
from datetime import timedelta
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from preprocessing import preprocess_data
from model import create_dataset
import gc

def load_model(model_path="models/saved_models/tft_model.pth"):
    df = pd.read_csv("data/processed/processed_network_data.csv")
    df = df[['timestamp', 'router_id', 'bandwidth_mbps', 'location', 'application_type', 'time_idx', 'bandwidth_mbps_scaled']]
    training = create_dataset(df)
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.005,
        hidden_size=2,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=1,
        output_size=1,
        loss=RMSE(),
    )
    tft.load_state_dict(torch.load(model_path))
    return tft

def forecast(tft, df, scaler, router_ids=["router_1"], forecast_horizon=7):
    thresholds = {"router_1": 300}
    forecast_dfs = []
    recommendations = {}
    for router_id in router_ids:
        df_router = df[df["router_id"] == router_id].copy()
        df_router = df_router.reset_index(drop=True)
        dataset = create_dataset(df_router)
        dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0, pin_memory=False)
        predictions = tft.predict(dataloader, return_x=True)[0].numpy()
        predictions = scaler.inverse_transform(predictions).flatten()
        last_date = pd.to_datetime(df_router["timestamp"].max())
        forecast_dates = [last_date + timedelta(days=x) for x in range(1, forecast_horizon + 1)]
        forecast_df = pd.DataFrame({
            "timestamp": forecast_dates,
            "router_id": [router_id] * forecast_horizon,
            "bandwidth_mbps_forecast": predictions[:forecast_horizon]
        })
        forecast_dfs.append(forecast_df)
        threshold = thresholds[router_id]
        breaches = forecast_df[forecast_df["bandwidth_mbps_forecast"] > threshold]
        recommendation = ""
        if not breaches.empty:
            breach_date = breaches["timestamp"].iloc[0].strftime("%Y-%m-%d")
            recommendation = f"{router_id} expected to exceed {threshold} Mbps on {breach_date}. Recommend upgrading to {threshold + 200} Mbps link."
        recommendations[router_id] = recommendation
    forecast_df = pd.concat(forecast_dfs, ignore_index=True)
    return forecast_df, recommendations

if __name__ == "__main__":
    df, scaler = preprocess_data()
    tft = load_model()
    forecast_df, recommendations = forecast(tft, df, scaler)
    for router_id, recommendation in recommendations.items():
        print(f"Recommendation for {router_id}: {recommendation or 'No capacity issues detected.'}")
    print(forecast_df)
    del df, tft, forecast_df, recommendations
    gc.collect()