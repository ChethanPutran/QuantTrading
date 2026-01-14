Let's extend the model further by integrating more advanced machine learning algorithms and using real-time data from APIs. We'll:

1. **Integrate More Machine Learning Algorithms**: We'll explore how we can integrate **Random Forests** and **XGBoost**, which are more robust than linear regression, and perform **hyperparameter tuning** for better performance.

2. **Real-Time Data from APIs**: We will use APIs like **Alpha Vantage** or **Yahoo Finance** to get real-time stock data.

---

### 🛠 **1. Machine Learning Algorithms Extension**

We'll use **Random Forest** and **XGBoost** as an alternative to Linear Regression for stock price prediction.

#### **Random Forest**: This model is a powerful ensemble method that works well with tabular data and can handle non-linear relationships better than linear regression.

#### **XGBoost**: A more advanced gradient-boosting algorithm that often provides state-of-the-art performance for regression tasks.

---

### 📊 **Step 1: Random Forest and XGBoost Implementation**

```python
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
```

### **Key Improvements**:

* **Random Forest** and **XGBoost** are ensemble methods that improve model accuracy by reducing variance and bias.
* **Mean Absolute Error (MAE)** is used to evaluate the models. You can replace this with other metrics like RMSE (Root Mean Square Error) or R² depending on your needs.

---

### 🛠 **2. Integrating Real-Time Stock Data from APIs**

We'll now integrate **Alpha Vantage** to fetch real-time stock data. You need an **API key** from Alpha Vantage (you can sign up [here](https://www.alphavantage.co/support/#api-key)).

---

### 📈 **Step 2: Fetch Real-Time Data Using Alpha Vantage**

```python
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
```

This function pulls the latest 5-minute interval data for a given stock symbol and returns it as a **Pandas DataFrame**. You can then use this data as input to your stock price prediction model.

---

### 🛠 **3. Putting Everything Together**

Now we’ll modify our stock prediction function to use **real-time data** and **trained models** (Random Forest or XGBoost).

```python
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
```

---

### 📚 **Real-Time Data Features to Use**:

* **Closing Price**: The last traded price of the stock.
* **Volume**: The number of shares traded.
* **Open, High, Low**: The opening, highest, and lowest prices for the stock.

These features can be fed into the machine learning model to predict the next potential price.

---

### 🚀 **Future Enhancements**:

1. **Real-Time Model Retraining**: Implement a system that retrains the model periodically with new data to adapt to changing market conditions.

2. **Web Scraping or Sentiment Analysis**: Integrate sentiment analysis of news articles or social media to predict stock movements based on news events.

3. **Additional APIs**: Use **Yahoo Finance**, **Polygon**, or **Quandl** for more detailed financial and fundamental data.

Would you like to see how to implement live stock predictions with backtesting or expand to include more advanced features (like news sentiment or options flow)?
