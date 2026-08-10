from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for LSTMModel")


# Define the Attention mechanism
class Attention(nn.Module):
    def __init__(self, input_dim, output_dim=None):
        _require_torch()
        super(Attention, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, 1))  # Learnable weights

    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, hidden_dim]
        scores = torch.matmul(lstm_out, self.W)  # [batch_size, seq_len, 1]
        scores = torch.tanh(scores)
        attention_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Weighted sum of lstm_out based on attention weights
        weighted_sum = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, hidden_dim]
        return weighted_sum, attention_weights
    
### Create LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_len, n_features, output_len,use_attention=True,model_path="lstm_stock_model.pth"):
        _require_torch()
        super().__init__()
        self.input_len = input_len
        self.n_features = n_features
        self.output_len = output_len
        self.use_attention = use_attention
        self.model_path=model_path

        # Model layers
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)
        # self.dropout = nn.Dropout(0.2)
        if use_attention:
            self.attention = Attention(64)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, output_len)

    def forward(self, x):
        out, _ = self.lstm1(x)
        if self.use_attention:
            context, _ = self.attention(out)
            out = context.unsqueeze(1).repeat(1, x.shape[1], 1)
        # out = self.dropout(out)
        out, _ = self.lstm2(out)
        # print(out.shape)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
        
    def save_model(self):
        torch.save(self.state_dict(), self.model_path)
        print(f"Model saved at {self.model_path}")

    def load_model(self):
        self.load_state_dict(torch.load(self.model_path))
        self.eval()
        print(f"Model loaded from {self.model_path}")


@dataclass
class LSTMPredictorConfig:
    sequence_length: int = 30
    future_steps: int = 10
    epochs: int = 20
    batch_size: int = 32
    lstm_units: tuple[int, int] = (64, 32)
    dropout: float = 0.2
