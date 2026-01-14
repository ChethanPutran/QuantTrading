import numpy as np
import torch
from torch.functional import F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib  # for saving scaler
import os
import pandas as pd
import pattern_detector as pad
from data_fetcher import load_stock_data,get_last_n_min_data,get_ticker

def generate_data(data,n_future_steps=5):
    tool = pad.Tool()
    technical_indicator = tool.capture_technical_indicators(data).bfill().ffill()
    data_with_ti = pd.concat([data,technical_indicator],axis=1)
    data_with_ti = tool.generate_trading_signals(data_with_ti)
    # Calculate the technical indicators
    # data['Trend'] = tool.capture_trend(data,single=True)
    # data['Momentum'] = tool.capture_momentum(data,single=True)
    # data['Price_action'] = tool.capture_price_action(data,single=True)
    # data['Volatility'] = tool.capture_volatility_params(data,single=True)
    # data['VolumeIndicator'] = tool.capture_volume_params(data,single=True)
    
    
    # Add candlestick patterns
    candlestick_patterns = tool.get_signal_from_candlestick_pattern(data)
    data['Bullish'] = candlestick_patterns['Bullish']
    data['Bearish'] = candlestick_patterns['Bearish']
    data['Signal'] = data_with_ti['Signal']

    data = pd.concat([data,technical_indicator],axis=1)
    ### Generate Output
    output = calculate_future_returns(data,n_future_steps)
    output.head()
    return data.shape, data.values,output.values

    
def calculate_future_returns(data,n_future_steps=5):
    arr = []
    for i in range(1, n_future_steps + 1):
        sr = (data['Close'].shift(-i) - data['Close']) / data['Close'] * 100
        sr.name = f"Return_{i}s"
        arr.append(sr)
    return pd.concat(arr,axis=1)

def test_model(model, test_dataset,batch_size=32, device='cpu'):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_inputs)
            loss = F.mse_loss(outputs, batch_targets)  # Mean Squared Error

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Test MSE: {avg_loss:.4f}")
    return avg_loss

def train_model(model,dataset, epochs=20, batch_size=32, validation_split=0.1, learning_rate=0.001, save_model=True):
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"📈 Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
    print("✅ Training Complete")

    if save_model:
        model.save_model()

def preprocess_data(data,input_len,output_len,output_data=None,output_col="Output", load_scaler = False,fit_scaler=True,scaler_path="scaler.pkl"):    
    if load_scaler and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded.")
    else:
        scaler = MinMaxScaler()
        fit_scaler = True

    if fit_scaler:
        data_scaled = scaler.fit_transform(data)
        joblib.dump(scaler, scaler_path)  # Save after fitting
        print("✅ Scaler fitted and saved.")
    else:
        data_scaled = scaler.transform(data)

    # Generate time series seq
    if output_data is not None:
        n_examples,n_features = data.shape
        n_samples = n_examples - input_len - output_len
        X = np.zeros((n_samples,input_len,n_features))
        y = np.zeros((n_samples,output_len))
        
        for i in range(n_samples):
            X[i] = data[i:i+input_len, :]
            y[i] = output_data[i+input_len]
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return TensorDataset(X_tensor, y_tensor)
        
    # Get the last input_len items
    X = data[-input_len:]
    return torch.tensor(X, dtype=torch.float32)

# Define the Attention mechanism
class Attention(nn.Module):
    def __init__(self, input_dim,output_dim):
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
        super().__init__()
        self.input_len = input_len
        self.n_features = n_features
        self.output_len = output_len
        self.use_attention = use_attention
        self.model_path=model_path

        # Model layers
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)
        # self.dropout = nn.Dropout(0.2)
        # if use_attention:
        #     self.attention = Attention(64)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, output_len)

    def forward(self, x):
        out, _ = self.lstm1(x)
        if self.use_attention:
            out, attention_weights = self.attention(out)  
        # out = self.dropout(out)
        out, _ = self.lstm2(out)
        # print(out.shape)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
        
    def save_model(self):
        torch.save(self.state_dict(), self.model_path)
        print(f"✅ Model saved at {self.model_path}")

    def load_model(self):
        self.load_state_dict(torch.load(self.model_path))
        self.eval()
        print(f"✅ Model loaded from {self.model_path}")

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_len, n_features, output_len, d_model=64, num_heads=8, num_layers=6, use_attention=True,model_path="transformer_stock_model.pth"):
        super(TransformerModel, self).__init__()
        
        self.input_len = input_len
        self.n_features = n_features
        self.output_len = output_len
        self.use_attention = use_attention
        self.model_path=model_path
        
        # Model layers
        self.embedding = nn.Linear(n_features, d_model)  # Embedding layer to convert input into d_model dimension
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_len, d_model))  # Positional Encoding
        
        # Transformer Encoder and Decoder layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=num_layers
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=num_layers
        )
        
        # # Attention layer if specified
        # if use_attention:
        #     self.attention = Attention(d_model)
        
        # Output fully connected layer
        self.fc = nn.Linear(d_model, output_len)

    def forward(self, x):
        # x: [batch_size, seq_len, n_features]
        x = self.embedding(x) + self.positional_encoding  # Add positional encoding
        
        # Transformer expects inputs in shape [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        
        # Transformer Encoder
        encoder_output = self.encoder(x)
        
        # If attention is used, apply it after the encoder
        # if self.use_attention:
        #     encoder_output, attention_weights = self.attention(encoder_output)
        
        # For the decoder, we use the encoder output and target sequence (e.g., for seq2seq tasks)
        # Here we assume the target sequence is the same as the input sequence (for simplicity)
        decoder_output = self.decoder(encoder_output, encoder_output)
        
        # Get the output from the last time step
        # decoder_output = decoder_output[-1, :, :]
        
        # Output layer
        out = self.fc(decoder_output)
        return out
        
    def save_model(self):
        torch.save(self.state_dict(), self.model_path)
        print(f"✅ Model saved at {self.model_path}")

    def load_model(self):
        self.load_state_dict(torch.load(self.model_path))
        self.eval()
        print(f"✅ Model loaded from {self.model_path}")


if __name__ == "__main__":
    ## Get the stock data
    status,train_data = load_stock_data("WAAREEENER.NS",interval='5m',start="2025-03-01",end="2025-04-23",refresh=True)
    ## Get the stock data
    status,test_data = load_stock_data("WAAREEENER.NS",interval='5m',start="2025-04-24",end="2025-04-25",refresh=True)
    train_size,trainX,trainY = generate_data(train_data)
    test_size,testX,testY = generate_data(test_data)

    n_future_steps = 5
    input_len = 10
    n_features = train_size[1] 
    n_examples = train_size[0] 
    print(f"No. training examples :{n_examples}")
    print(f"Input Sequence :{input_len}")
    print(f"Input Features :{n_features}")
    print(f"Output Sequence :{n_future_steps}")

    ### Preprocess the data
    train_dataset = preprocess_data(trainX,input_len,n_future_steps,trainY)
    test_dataset = preprocess_data(testX,input_len,n_future_steps,testY,fit_scaler=False)


    model = LSTMModel(input_len, n_features, n_future_steps)
    # Create model instance
    model = TransformerModel(input_len, n_features, n_future_steps, use_attention=True)


    model.load_model()
    test_model(model,test_dataset)

    X = test_dataset[0][0].unsqueeze(0)
    y = test_dataset[0][1]
    with torch.no_grad():
        y_pred = model(X)

    y_pred,y

    #Here's the plan:

    # Fetch historical data for a stock.
    # Compute technical indicators (RSI, EMA, MACD, etc.) for each bar.
    # Compute candle stick patterns
    # Generate Buy/Sell signals based on the LSTM predictions.
    # Simulate trades: Track capital, entries, exits, and performance metrics.

    
    # # Target: % Profit after n minutes
    # n = 3  # minutes ahead
    # df["Future_Close"] = df["Close"].shift(-n)
    # df["Profit_Percent"] = (df["Future_Close"] - df["Close"]) / df["Close"] * 100
    # df.dropna(inplace=True)

    # # # Features for LSTM
    # # features = ["Open", "High", "Low", "Close", "Volume", "RSI", "EMA_9", "EMA_20",
    # #             "MACD_12_26_9", "MACDh_12_26_9", "Hammer", "Engulfing", "Doji"]
    # # X = df[features].values
    # # y = df["Profit_Percent"].values
    # model = LSTMModel()
    
    # # Run backtest
    # profit, trades = model.backtest_strategy(ticker="RELIANCE.NS", window=60, period="90d", n_minutes=3)
    # print(f"Backtest Profit: {profit:.2f}%")
    # for trade in trades:
    #     print(trade)







