def get_model(model_name='xgbr'):   
    from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
    # from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC,SVR
    from sklearn.naive_bayes import CategoricalNB
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor,XGBClassifier

    model = None
    if model_name == 'dtc':
        # Decision Tree Classifier
        model = DecisionTreeClassifier()
    elif model_name == 'dtr':
        # Decision Tree Regressor
        model = DecisionTreeRegressor()
    elif model_name == 'rtc':
        # Random Forest Classifier
        model = RandomForestClassifier()
    elif model_name == 'rtr':
        # Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'svc':
        # Support Vector Machine (SVM) Classifier
        model = SVC(kernel='rbf', C=1.0)
    elif model_name == 'svr':
        # Support Vector Machine (SVM) Regressor
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    elif model_name == 'xgbc':
        # XGB Classifier
        model = XGBClassifier()
    elif model_name == 'xgbr':
        # XGB Regressor
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    return model

def full_feature_stock_model(features, weights, base_price):
    """
    Simulates stock price with maximum realism using all known influential features.
    
    Parameters:
    - features: dict of all input features.
    - weights: dict mapping feature names to importance.
    - base_price: float, last known price.
    
    Returns:
    - float: simulated price
    """

    # --- Normalizations and Interactions ---
    rsi_score = np.tanh((features['rsi'] - 50) / 20)
    macd_score = features['macd'] - features['macd_signal']
    sma_trend = np.tanh((features['sma_9'] - features['sma_21']) / base_price)
    vwap_diff = (features['price'] - features['vwap']) / features['vwap']
    atr_score = np.tanh(features['atr'] / base_price)
    adx_trend_strength = np.tanh(features['adx'] / 50)

    pe_inverse = 1 / (features['pe_ratio'] + 1e-6)
    eps_momentum = np.log1p(features['eps']) * np.tanh(features['revenue_growth'])
    debt_penalty = -np.tanh(features['debt_to_equity'] / 2)
    margin_gain = np.tanh(features['profit_margin'])

    news_score = np.sign(features['news_sentiment']) * np.sqrt(abs(features['news_sentiment']))
    social_score = np.log1p(features['social_volume']) * features['social_sentiment']
    insider_score = features['insider_activity'] - features['short_seller_activity']

    inflation_penalty = -np.exp(features['cpi']) / 100
    yield_drag = -np.tanh(features['treasury_yield'] / 10)
    oil_penalty = -np.tanh(features['crude_oil_price'] / 100)
    unemployment_penalty = -np.tanh(features['unemployment_rate'] / 10)

    geopolitics_drag = -features['geopolitical_risk']
    earnings_surprise = np.tanh(features['earnings_surprise'])

    iv_penalty = -np.tanh(features['implied_volatility'])
    liquidity_signal = np.tanh(features['order_book_depth'] / 1000)

    rating_boost = features['analyst_rating_change']
    guidance_shift = np.tanh(features['guidance_change'])

    # --- Weighted Aggregation ---
    score = 0.0

    # Technical
    score += weights['rsi'] * rsi_score
    score += weights['macd'] * macd_score
    score += weights['sma'] * sma_trend
    score += weights['vwap'] * vwap_diff
    score += weights['atr'] * atr_score
    score += weights['adx'] * adx_trend_strength
    score += weights['bollinger'] * features['bollinger_width']

    # Fundamental
    score += weights['pe'] * pe_inverse
    score += weights['eps'] * eps_momentum
    score += weights['debt'] * debt_penalty
    score += weights['margin'] * margin_gain
    score += weights['dividend'] * features['dividend_yield']
    score += weights['fcf'] * np.tanh(features['free_cash_flow'])

    # Sentiment
    score += weights['news'] * news_score
    score += weights['social'] * social_score
    score += weights['insider'] * insider_score
    score += weights['analyst'] * rating_boost
    score += weights['guidance'] * guidance_shift

    # Macro
    score += weights['cpi'] * inflation_penalty
    score += weights['yield'] * yield_drag
    score += weights['oil'] * oil_penalty
    score += weights['unemployment'] * unemployment_penalty
    score += weights['geopolitical'] * geopolitics_drag
    score += weights['dxy'] * -np.tanh(features['usd_index'] / 100)

    # Microstructure
    score += weights['volume'] * np.log1p(features['volume_change'])
    score += weights['iv'] * iv_penalty
    score += weights['liquidity'] * liquidity_signal
    score += weights['putcall'] * -features['put_call_ratio']

    # Events
    score += weights['earnings'] * earnings_surprise
    score += weights['merger'] * features['mna_activity']

    # Final score
    price = base_price * (1 + np.tanh(score))
    return round(price, 2)

def ultra_complex_price_model(features: dict, weights: dict, base_price: float):
    """
    Simulate next stock price based on ultra-complex multi-domain features.
    
    Args:
    - features: Dictionary of feature values (numeric).
    - weights: Dictionary of feature weights (importance).
    - base_price: Last closing price or base price.

    Returns:
    - Simulated price
    """
    score = 0

    for key in features:
        val = features[key]
        w = weights.get(key, 0.0)

        # Nonlinear transformation
        if key.startswith('rsi') or key.startswith('sentiment'):
            val = np.tanh((val - 50) / 20)
        elif key.startswith('volatility') or key.startswith('iv') or key.endswith('deviation'):
            val = np.tanh(val)
        elif 'spread' in key or 'yield' in key:
            val = -np.abs(val)
        elif key.endswith('score') or 'surprise' in key:
            val = np.tanh(val)
        elif key.startswith('growth') or key.endswith('change') or key.endswith('momentum'):
            val = np.log1p(val)
        elif 'penalty' in key or 'risk' in key or 'drag' in key:
            val = -np.tanh(val)
        elif key.endswith('_flag'):
            val = float(val)  # 0 or 1
        else:
            val = val

        score += w * val

    impact = np.tanh(score)  # to keep price change bounded
    predicted_price = base_price * (1 + impact)
    return round(predicted_price, 2)


import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Simulated example features (replace this with actual data)
np.random.seed(42)
features = np.random.rand(1000, 20)  # 1000 data points, 20 features (can represent stock data)
prices = np.random.rand(1000) * 100  # Stock prices as target

# --- Step 1: Apply Standard Scaling ---
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --- Step 2: Apply PCA ---
from sklearn.decomposition import PCA
pca = PCA(n_components=5)  # Reduce to 5 principal components
features_pca = pca.fit_transform(features_scaled)

# --- Step 3: Train Regression Models ---
X_train, X_test, y_train, y_test = train_test_split(features_pca, prices, test_size=0.2, random_state=42)

# 1. Random Forest
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# 2. XGBoost
xg_regressor = xgb.XGBRegressor(n_estimators=100, random_state=42)
xg_regressor.fit(X_train, y_train)

# --- Step 4: Evaluate Models ---
rf_pred = rf_regressor.predict(X_test)
xg_pred = xg_regressor.predict(X_test)

print(f"Random Forest MAE: {mean_absolute_error(y_test, rf_pred)}")
print(f"XGBoost MAE: {mean_absolute_error(y_test, xg_pred)}")

import requests
import pandas as pd

def get_stock_data(symbol, api_key):
    """
    Fetches the real-time stock data for a given symbol using Alpha Vantage API.
    
    Parameters:
    - symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
    - api_key (str): Alpha Vantage API key
    
    Returns:
    - DataFrame: Stock data with date, open, high, low, close, volume
    """
    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': '5min',  # Can choose '1min', '5min', '15min', etc.
        'apikey': api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'Time Series (5min)' in data:
        time_series = data['Time Series (5min)']
        df = pd.DataFrame(time_series).T
        df = df.astype(float)
        return df
    else:
        raise ValueError(f"Error fetching data: {data.get('Note', 'Unknown error')}")

# Example usage:
api_key = "YOUR_ALPHA_VANTAGE_API_KEY"
symbol = "AAPL"  # Example: Apple Inc.
stock_data = get_stock_data(symbol, api_key)
print(stock_data.head())

def predict_real_time_stock_price(symbol, regressor, pca, scaler, api_key):
    """
    Predict stock price in real-time using trained model and real-time stock data.
    
    Parameters:
    - symbol (str): Stock symbol to predict
    - regressor (sklearn model): Trained regression model (e.g., RF or XGBoost)
    - pca (PCA): Fitted PCA model
    - scaler (StandardScaler): Fitted scaler
    - api_key (str): Alpha Vantage API key
    
    Returns:
    - float: Predicted stock price
    """
    # Fetch real-time stock data from Alpha Vantage
    stock_data = get_stock_data(symbol, api_key)
    
    # Select relevant features (e.g., closing price, volume, etc.)
    features_input = [
        stock_data['4. close'].iloc[-1],  # Latest closing price
        stock_data['5. volume'].iloc[-1],  # Latest volume
        stock_data['2. high'].iloc[-1],    # Latest high price
        stock_data['3. low'].iloc[-1],     # Latest low price
        stock_data['1. open'].iloc[-1]     # Latest open price
    ]
    
    # Standardize and apply PCA (same as model training)
    features_scaled = scaler.transform([features_input])
    features_pca = pca.transform(features_scaled)
    
    # Predict stock price using the trained regressor
    predicted_price = regressor.predict(features_pca)
    
    return predicted_price[0]

# Example usage:
predicted_price = predict_real_time_stock_price(symbol="AAPL", regressor=rf_regressor, 
                                                pca=pca, scaler=scaler, api_key=api_key)
print(f"Predicted real-time stock price: {predicted_price:.2f}")
