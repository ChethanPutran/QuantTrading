import yfinance as yf
import ta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Configuration ---
ticker = "RELIANCE.NS"
interval = "1m"
period = "1d"
sequence_length = 30
future_minutes = 10

# --- Step 1: Fetch data ---
data = yf.download(ticker, interval=interval, period=period)
data.dropna(inplace=True)

# --- Step 2: Add technical indicators ---
data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
data['macd'] = ta.trend.MACD(data['Close']).macd()
data['ema'] = ta.trend.EMAIndicator(data['Close'], window=10).ema_indicator()
bb = ta.volatility.BollingerBands(data['Close'])
data['bb_high'] = bb.bollinger_hband()
data['bb_low'] = bb.bollinger_lband()

# --- Step 3: Calculate future profit ---
data['future_price'] = data['Close'].shift(-future_minutes)
data['profit'] = data['future_price'] - data['Close']
data.dropna(inplace=True)

# --- Step 4: Normalize features ---
features = ['Close', 'rsi', 'macd', 'ema', 'bb_high', 'bb_low']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features].fillna(0))

# --- Step 5: Create sequences ---
X, y = [], []
for i in range(len(data_scaled) - sequence_length - future_minutes):
    X.append(data_scaled[i:i+sequence_length])
    y.append(data['profit'].iloc[i+sequence_length])

X, y = np.array(X), np.array(y)

# --- Step 6: Build LSTM model ---
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# --- Step 7: Train model ---
model.fit(X, y, epochs=20, batch_size=32)

# --- Step 8: Predict latest ---
latest_seq = data_scaled[-sequence_length:]
predicted_profit = model.predict(latest_seq.reshape(1, sequence_length, -1))

print(f"Expected profit in next {future_minutes} minutes if you buy now: ₹{predicted_profit[0][0]:.2f}")
