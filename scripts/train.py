import torch
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import RMSE
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from preprocessing import preprocess_data
from model import create_dataset
import os
import gc

def train_model():
    try:
        # Load data with minimal memory usage
        df, _ = preprocess_data()
        df = df[['timestamp', 'router_id', 'bandwidth_mbps', 'location', 'application_type', 'time_idx', 'bandwidth_mbps_scaled']]
        training = create_dataset(df)
        dataloader = training.to_dataloader(train=True, batch_size=8, num_workers=0, pin_memory=False)
        
        # Smaller model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.005,  # Lower learning rate
            hidden_size=2,       # Minimal model size
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=1,
            output_size=1,
            loss=RMSE(),
            log_interval=10,
            reduce_on_plateau_patience=2,
        )
        tft.save_hyperparameters(ignore=['loss', 'logging_metrics'])
        
        # Minimal trainer configuration
        trainer = Trainer(
            max_epochs=3,  # Minimal epochs
            accelerator="cpu",
            enable_checkpointing=True,
            callbacks=[EarlyStopping(monitor="train_loss", patience=1, mode="min")],
            logger=False,
            gradient_clip_val=0.1  # Prevent memory spikes
        )
        
        trainer.fit(tft, train_dataloaders=dataloader)
        
        os.makedirs("models/saved_models", exist_ok=True)
        torch.save(tft.state_dict(), "models/saved_models/tft_model.pth")
        print("Model saved to models/saved_models/tft_model.pth")
        
        # Clean up memory
        del df, training, dataloader, tft, trainer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"Training error: {e}")
        raise

if __name__ == "__main__":
    train_model()