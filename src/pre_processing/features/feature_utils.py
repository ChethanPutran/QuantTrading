To replace the handcrafted weights with those learned via regression or machine learning, and to reduce the feature space using **PCA** (Principal Component Analysis), we can follow a step-by-step process.

Here's a breakdown of the steps:

---

### 1. **Use PCA for Dimensionality Reduction**

We will reduce the feature space by applying PCA to extract the most important features.

### 2. **Train a Regression Model (e.g., Linear Regression or Random Forest)**

Once we reduce the features, we can use a regression model to learn the weights from historical data.

---

### 📊 **Steps for Implementation**:

#### **Step 1: Perform PCA on the feature set**

First, we will apply PCA to the features before feeding them into a regression model. This helps reduce the dimensionality and can capture latent variables that explain most of the variance in the data.

#### **Step 2: Train a Regression Model**

We will use **linear regression** as an example, but you could replace it with any model (e.g., Random Forest, XGBoost, or Neural Networks).

#### **Step 3: Use the Learned Model for Stock Price Prediction**

After training the regression model, we can use it to predict the stock price by feeding in new data points.

---

### 🛠 **Code Implementation**

Here is how you can implement this:

#### **1. PCA and Regression Model Setup**

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulated example features (replace this with actual data)
np.random.seed(42)
features = np.random.rand(1000, 20)  # 1000 data points, 20 features (can represent stock data)
prices = np.random.rand(1000) * 100  # Stock prices as target

# --- Step 1: Apply Standard Scaling ---
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --- Step 2: Apply PCA ---
pca = PCA(n_components=5)  # Reduce to 5 principal components
features_pca = pca.fit_transform(features_scaled)

# --- Step 3: Train a Regression Model ---
X_train, X_test, y_train, y_test = train_test_split(features_pca, prices, test_size=0.2, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# --- Step 4: Evaluate the Model ---
y_pred = regressor.predict(X_test)
print(f"Model R^2 score: {regressor.score(X_test, y_test)}")
```

#### **2. Predict Stock Price with the Trained Model**

Now that we have trained the model, we can integrate it into the stock price function and predict the stock price based on the learned regression weights.

```python
def complex_stock_price_model_with_ml(features, base_price, regressor, pca, scaler):
    """
    Predict stock price based on features using a PCA and Regression model.

    Parameters:
    - features (dict): Dictionary of all input features
    - base_price (float): Previous day's price or starting point
    - regressor (sklearn model): Trained regression model
    - pca (PCA): Fitted PCA model
    - scaler (StandardScaler): Fitted scaler
    
    Returns:
    - float: Predicted stock price
    """
    
    # Convert input features to the format used for training
    features_input = np.array([
        features['rsi'], features['macd'], features['macd_signal'], features['bollinger_width'],
        features['news_sentiment'], features['social_volume'], features['social_sentiment'],
        features['pe_ratio'], features['eps'], features['roe'], features['inflation_rate'],
        features['vwap'], features['price'], features['volume_change'], features['insider_activity'],
        features['short_ratio'], features['analyst_rating_change'], features['crude_oil_price'],
        features['treasury_yield_10yr']
    ])
    
    # Apply the scaler and PCA (standardize and reduce dimensionality)
    features_scaled = scaler.transform([features_input])  # Standardize input
    features_pca = pca.transform(features_scaled)  # Apply PCA
    
    # Predict the stock price using the trained regressor
    predicted_price = regressor.predict(features_pca)
    
    return predicted_price[0]

# Example features (from real-time data or new input)
new_features = {
    'rsi': 62,
    'macd': 1.5,
    'macd_signal': 1.2,
    'bollinger_width': 0.04,
    'news_sentiment': 0.6,
    'social_volume': 8000,
    'social_sentiment': 0.7,
    'pe_ratio': 22,
    'eps': 3.2,
    'roe': 0.15,
    'inflation_rate': 0.032,
    'vwap': 98,
    'price': 100,
    'volume_change': 0.12,
    'insider_activity': 0.03,
    'short_ratio': 0.18,
    'analyst_rating_change': 0.4,
    'crude_oil_price': 84,
    'treasury_yield_10yr': 0.045,
}

# Predicting stock price for new data
predicted_price = complex_stock_price_model_with_ml(new_features, base_price=100.0, 
                                                   regressor=regressor, pca=pca, scaler=scaler)
print(f"Predicted stock price: {predicted_price:.2f}")
```

---

### 🔍 **Explanation of Steps:**

1. **Standard Scaling**: The input features are standardized to have zero mean and unit variance. This is important when working with PCA and regression models.

2. **PCA**: We apply PCA to reduce the dimensionality of the feature set. In this example, we keep the top 5 principal components, but you can adjust this based on the explained variance.

3. **Linear Regression**: The regression model learns the relationship between the features (after PCA reduction) and the stock price. You can also try other models, such as Random Forest, XGBoost, or even Neural Networks.

4. **Prediction**: For new stock data (represented as the `new_features` dictionary), the model predicts the stock price.

---

### 🚀 **Future Steps:**

* **Feature Selection**: Before applying PCA, you could use techniques like **L1 regularization** (Lasso) to perform feature selection.

* **Cross-Validation**: Use cross-validation (e.g., K-fold) to ensure the model generalizes well to unseen data.

* **Advanced Models**: Replace the linear regressor with more complex models like **XGBoost** or **Random Forests**.

Would you like me to extend the model further by using other machine learning algorithms or integrating real-time data from APIs?
