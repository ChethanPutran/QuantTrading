import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Simulate some data for time series (you can replace this with your actual dataset)
data = np.sin(np.linspace(0, 100, 1000))  # Example: sine wave
data = data.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Function to create sequences
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Create sequences for training
seq_length = 50
X, y = create_sequences(data_scaled, seq_length)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM input

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split into training and test sets
train_size = int(len(X_tensor) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, lstm_output):
        # Attention mechanism
        attn_weights = self.attn(lstm_output)
        attn_weights = self.softmax(attn_weights)
        weighted_sum = torch.sum(attn_weights * lstm_output, dim=1)
        return weighted_sum

# LSTM with Attention Model
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        attention_out = self.attention(lstm_out)
        out = self.fc(attention_out)
        return out

# Model initialization
model = LSTMWithAttention(input_size=1, hidden_size=64, output_size=1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(X_train)
    
    # Compute loss
    loss = criterion(y_pred, y_train)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Predict on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# Inverse transform to get the original scale
y_pred = y_pred.numpy()
y_pred_actual = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test.numpy())

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='Actual')
plt.plot(y_pred_actual, label='Predicted')
plt.legend()
plt.title('Time Series Forecasting with LSTM + Attention (PyTorch)')
plt.show()
