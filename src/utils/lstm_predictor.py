from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn

from models.lstm import LSTMPredictorConfig


def add_basic_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight indicators without requiring optional TA packages."""

    frame = data.copy()
    close = frame["Close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)

    frame["rsi"] = 100 - (100 / (1 + rs))
    frame["ema"] = close.ewm(span=10, adjust=False).mean()
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    frame["macd"] = ema_fast - ema_slow
    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    frame["bb_high"] = rolling_mean + 2 * rolling_std
    frame["bb_low"] = rolling_mean - 2 * rolling_std
    return frame.bfill().ffill()


def create_profit_sequences(
    data: pd.DataFrame,
    config: LSTMPredictorConfig = LSTMPredictorConfig(),
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Create scaled LSTM inputs and future-profit labels."""

    if "Close" not in data:
        raise KeyError("data must include a Close column")

    frame = add_basic_indicators(data)
    frame["future_price"] = frame["Close"].shift(-config.future_steps)
    frame["profit"] = frame["future_price"] - frame["Close"]
    frame = frame.dropna()

    columns = feature_columns or ["Close", "rsi", "macd", "ema", "bb_high", "bb_low"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(frame[columns].astype(float))

    x_values, y_values = [], []
    limit = len(scaled) - config.sequence_length
    for idx in range(limit):
        x_values.append(scaled[idx : idx + config.sequence_length])
        y_values.append(frame["profit"].iloc[idx + config.sequence_length])

    return np.asarray(x_values), np.asarray(y_values, dtype=float), scaler




def train_lstm_profit_model(
    data: pd.DataFrame,
    config: LSTMPredictorConfig = LSTMPredictorConfig(),
):
    """Train a profit-forecasting LSTM and return ``(model, scaler)``."""

    x_values, y_values, scaler = create_profit_sequences(data, config)
    if len(x_values) == 0:
        raise ValueError("not enough rows to create LSTM sequences")

    model = build_lstm_model((x_values.shape[1], x_values.shape[2]), config)
    model.fit(
        x_values,
        y_values,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=0,
    )
    return model, scaler


def predict_latest_profit(
    data: pd.DataFrame,
    model,
    scaler: MinMaxScaler,
    config: LSTMPredictorConfig = LSTMPredictorConfig(),
    feature_columns: list[str] | None = None,
) -> float:
    """Predict future profit from the latest available sequence."""

    columns = feature_columns or ["Close", "rsi", "macd", "ema", "bb_high", "bb_low"]
    frame = add_basic_indicators(data).dropna()
    if len(frame) < config.sequence_length:
        raise ValueError("not enough rows for the configured sequence_length")

    latest = scaler.transform(frame[columns].astype(float))[-config.sequence_length :]
    prediction = model.predict(latest.reshape(1, config.sequence_length, -1), verbose=0)
    return float(np.asarray(prediction).reshape(-1)[0])




def full_feature_stock_model(
    features: dict[str, float], weights: dict[str, float], base_price: float
) -> float:
    """Deterministic nonlinear price simulator for a rich feature dictionary."""

    required = {
        "rsi",
        "macd",
        "macd_signal",
        "sma_9",
        "sma_21",
        "price",
        "vwap",
        "atr",
        "adx",
        "bollinger_width",
        "pe_ratio",
        "eps",
        "revenue_growth",
        "debt_to_equity",
        "profit_margin",
        "dividend_yield",
        "free_cash_flow",
        "news_sentiment",
        "social_volume",
        "social_sentiment",
        "insider_activity",
        "short_seller_activity",
        "analyst_rating_change",
        "guidance_change",
        "cpi",
        "treasury_yield",
        "crude_oil_price",
        "unemployment_rate",
        "geopolitical_risk",
        "usd_index",
        "volume_change",
        "implied_volatility",
        "order_book_depth",
        "put_call_ratio",
        "earnings_surprise",
        "mna_activity",
    }
    missing = sorted(required.difference(features))
    if missing:
        raise KeyError(f"missing required features: {missing}")

    safe_base = max(float(base_price), 1e-9)
    score = 0.0

    transformed = {
        "rsi": np.tanh((features["rsi"] - 50) / 20),
        "macd": features["macd"] - features["macd_signal"],
        "sma": np.tanh((features["sma_9"] - features["sma_21"]) / safe_base),
        "vwap": (features["price"] - features["vwap"]) / max(features["vwap"], 1e-9),
        "atr": np.tanh(features["atr"] / safe_base),
        "adx": np.tanh(features["adx"] / 50),
        "bollinger": features["bollinger_width"],
        "pe": 1 / (features["pe_ratio"] + 1e-6),
        "eps": np.log1p(max(features["eps"], -0.999999))
        * np.tanh(features["revenue_growth"]),
        "debt": -np.tanh(features["debt_to_equity"] / 2),
        "margin": np.tanh(features["profit_margin"]),
        "dividend": features["dividend_yield"],
        "fcf": np.tanh(features["free_cash_flow"]),
        "news": np.sign(features["news_sentiment"])
        * np.sqrt(abs(features["news_sentiment"])),
        "social": np.log1p(max(features["social_volume"], 0))
        * features["social_sentiment"],
        "insider": features["insider_activity"] - features["short_seller_activity"],
        "analyst": features["analyst_rating_change"],
        "guidance": np.tanh(features["guidance_change"]),
        "cpi": -np.exp(features["cpi"]) / 100,
        "yield": -np.tanh(features["treasury_yield"] / 10),
        "oil": -np.tanh(features["crude_oil_price"] / 100),
        "unemployment": -np.tanh(features["unemployment_rate"] / 10),
        "geopolitical": -features["geopolitical_risk"],
        "dxy": -np.tanh(features["usd_index"] / 100),
        "volume": np.log1p(max(features["volume_change"], -0.999999)),
        "iv": -np.tanh(features["implied_volatility"]),
        "liquidity": np.tanh(features["order_book_depth"] / 1000),
        "putcall": -features["put_call_ratio"],
        "earnings": np.tanh(features["earnings_surprise"]),
        "merger": features["mna_activity"],
    }

    for key, value in transformed.items():
        score += weights.get(key, 0.0) * value

    return round(safe_base * (1 + np.tanh(score)), 2)


def ultra_complex_price_model(
    features: dict[str, float], weights: dict[str, float], base_price: float
) -> float:
    """Generic weighted nonlinear price simulator."""

    score = 0.0
    for key, value in features.items():
        val = float(value)
        weight = float(weights.get(key, 0.0))

        if key.startswith(("rsi", "sentiment")):
            val = np.tanh((val - 50) / 20)
        elif key.startswith(("volatility", "iv")) or key.endswith("deviation"):
            val = np.tanh(val)
        elif "spread" in key or "yield" in key:
            val = -abs(val)
        elif key.endswith("score") or "surprise" in key:
            val = np.tanh(val)
        elif key.startswith("growth") or key.endswith(("change", "momentum")):
            val = np.log1p(max(val, -0.999999))
        elif any(token in key for token in ("penalty", "risk", "drag")):
            val = -np.tanh(val)
        elif key.endswith("_flag"):
            val = float(bool(val))

        score += weight * val

    return round(float(base_price) * (1 + np.tanh(score)), 2)


def fit_pca_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    regressor: Any | None = None,
    n_components: int = 5,
):
    """Fit scaler, PCA, and a regressor; return all three fitted objects."""

    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = regressor or RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("regressor", estimator),
        ]
    )
    pipeline.fit(np.asarray(features, dtype=float), np.asarray(targets, dtype=float))
    return pipeline



import numpy as np
import torch
import torch.nn.functional as F
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
            X[i] = data_scaled[i:i+input_len, :]
            y[i] = output_data[i+input_len]
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return TensorDataset(X_tensor, y_tensor)
        
    # Get the last input_len items
    X = data_scaled[-input_len:]
    return torch.tensor(X, dtype=torch.float32)

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



from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd


FeatureProvider = Callable[[str], float]


def predict_from_feature_vector(
    features: list[float] | np.ndarray,
    regressor: Any,
    scaler: Any | None = None,
    transformer: Any | None = None,
) -> float:
    """Predict with an estimator using optional scaler/PCA-style transformers."""

    x = np.asarray(features, dtype=float).reshape(1, -1)
    if scaler is not None:
        x = scaler.transform(x)
    if transformer is not None:
        x = transformer.transform(x)
    prediction = regressor.predict(x)
    return float(np.asarray(prediction).reshape(-1)[0])


def market_frame_to_features(stock_data: pd.DataFrame) -> list[float]:
    """Extract OHLCV features from an Alpha Vantage-style intraday frame."""

    required = ["4. close", "5. volume", "2. high", "3. low", "1. open"]
    missing = [column for column in required if column not in stock_data]
    if missing:
        raise KeyError(f"missing market data columns: {missing}")

    latest = stock_data.iloc[-1]
    return [float(latest[column]) for column in required]


def predict_real_time_stock_price(
    symbol: str,
    regressor: Any,
    scaler: Any,
    pca: Any,
    market_data_provider: Callable[[str], pd.DataFrame],
) -> float:
    """Predict a symbol price from a caller-supplied real-time data provider."""

    stock_data = market_data_provider(symbol)
    features = market_frame_to_features(stock_data)
    return predict_from_feature_vector(features, regressor, scaler=scaler, transformer=pca)


def predict_stock_price_with_advanced_features(
    symbol: str,
    regressor: Any,
    pca: Any,
    scaler: Any,
    market_data_provider: Callable[[str], pd.DataFrame],
    feature_providers: Mapping[str, FeatureProvider] | None = None,
) -> float:
    """Predict using OHLCV plus optional external sentiment/options features."""

    stock_data = market_data_provider(symbol)
    features = market_frame_to_features(stock_data)

    for provider in (feature_providers or {}).values():
        features.append(float(provider(symbol)))

    return predict_from_feature_vector(features, regressor, scaler=scaler, transformer=pca)






