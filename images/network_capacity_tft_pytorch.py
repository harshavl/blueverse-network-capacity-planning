import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Encode categorical variables
    le_router = LabelEncoder()
    le_location = LabelEncoder()
    le_app = LabelEncoder()
    
    df['router_id'] = le_router.fit_transform(df['router_id'])
    df['location'] = le_location.fit_transform(df['location'])
    df['application_type'] = le_app.fit_transform(df['application_type'])
    
    return df

# Create sequences for TFT
def create_sequences(data, seq_length=3):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][['router_id', 'location', 'application_type', 'bandwidth_mbps']].values)
        y.append(data.iloc[i+seq_length]['bandwidth_mbps'])
    return np.array(X), np.array(y)

# TFT-inspired model in PyTorch
class TFTModel(nn.Module):
    def __init__(self, seq_length, feature_dim, hidden_dim=64, num_heads=4):
        super(TFTModel, self).__init__()
        
        # Variable Selection Networks (simplified)
        self.dense1 = nn.Linear(feature_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(0.1)
        
        # Position-wise feed-forward network
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_length, feature_dim)
        x = torch.relu(self.dense1(x))
        x = self.layer_norm1(x)
        
        # Transpose for attention: (seq_length, batch, hidden_dim)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)  # Residual connection
        x = x.permute(1, 0, 2)  # Back to (batch, seq_length, hidden_dim)
        
        x = torch.relu(self.dense2(x))
        x = self.layer_norm2(x)
        
        # Take the last time step for prediction
        x = self.output(x[:, -1, :])
        return x

# Training function
def train_model(model, X, y, epochs=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

# Main function
def main():
    # Sample input data (replace with actual file path)
    input_data = """timestamp,router_id,bandwidth_mbps,location,application_type
2025-08-29 12:00:00,router_1,230.123456,datacenter_1,web
2025-08-28 12:00:00,router_1,361.097065,datacenter_1,web
2025-08-27 12:00:00,router_1,236.388226,datacenter_1,web
2025-08-26 12:00:00,router_1,146.348580,datacenter_1,web"""
    
    # Save sample data to temporary CSV
    with open('temp_network_data.csv', 'w') as f:
        f.write(input_data)
    
    # Load and preprocess data
    df = load_data('temp_network_data.csv')
    
    # Create sequences
    seq_length = 3
    X, y = create_sequences(df, seq_length)
    
    # Build and train model
    model = TFTModel(seq_length=seq_length, feature_dim=X.shape[2])
    train_model(model, X, y)
    
    # Predict future bandwidth
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    last_sequence = torch.FloatTensor(X[-1:]).to(device)
    with torch.no_grad():
        predicted_bandwidth = model(last_sequence).item()
    
    # Generate future timestamp
    last_timestamp = df['timestamp'].iloc[-1]
    next_timestamp = last_timestamp + timedelta(days=1)
    
    # Print prediction
    print(f"Predicted bandwidth for {next_timestamp}: {predicted_bandwidth:.2f} Mbps")
    
    # Basic capacity planning recommendation
    capacity_threshold = 400  # Example threshold in Mbps
    if predicted_bandwidth > capacity_threshold:
        print(f"Warning: Predicted bandwidth ({predicted_bandwidth:.2f} Mbps) exceeds capacity threshold ({capacity_threshold} Mbps). Consider upgrading infrastructure.")

if __name__ == "__main__":
    main()