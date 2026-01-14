

# Define MPC Controller class
class MPCController:
    def __init__(self, model, time_horizon, initial_balance=10000):
        self.model = model
        self.time_horizon = time_horizon
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = 0  # Current stock position (number of shares held)
        
    def objective_function(self, actions, historical_features,charges=20):
        """
        Objective function to maximize the cumulative reward (profit).
        actions: List of buy/sell/hold actions
        historical_prices: Stock prices for the time horizon
        """
        balance = self.initial_balance
        positions = 0
        total_reward = 0
        for t in range(self.time_horizon):
            action = np.round(actions[t])
            features = historical_features[t]

            # Model predicts profit if we act now
            predicted_profit = self.predict_profit(features)
            
            if action == 1:  # Buy
                num_shares = balance // features[0]  # Assume price is the first feature
                balance -= num_shares * features[0] + charges
                positions += num_shares
            elif action == -1 and positions > 0:  # Sell only if holding shares
                balance += positions * features[0] - charges
                positions = 0

            # Portfolio Value: balance + value of held shares
            total_reward = balance + positions * features[0] + predicted_profit
        
        return -total_reward  # Minimize negative of total expected reward

            if action == 1:  # Buy
                positions += balance // price
                balance -= positions * price - charges
            elif action == -1:  # Sell
                balance += positions * price - charges
                positions = 0

            # Reward: The value of the portfolio
            total_reward = balance + positions * price
        
        return -total_reward  # We minimize the negative reward

        """
        Objective: Maximize the expected cumulative profit based on model predictions.
        actions: [buy(1)/sell(-1)/hold(0)] for each time step
        historical_features: technical indicators for time_horizon
        """
        balance = self.initial_balance
        positions = 0
        total_reward = 0
        
        for t in range(self.time_horizon):
            action = np.round(actions[t])  # round to 0, 1, or -1
            features = historical_features[t]

            # Model predicts profit if we act now
            predicted_profit = self.predict_profit(features)

           

    import numpy as np
import torch
from scipy.optimize import minimize

class MPCController:
    def __init__(self, model, time_horizon, initial_balance=10000, device="cpu"):
        self.model = model
        self.time_horizon = time_horizon
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = 0
        self.device = device
        
    def predict_profit(self, input_features):
        """
        Use the trained LSTM model to predict profit after 'n' minutes.
        input_features: tensor of shape (seq_len, features)
        """
        self.model.eval()
        with torch.no_grad():
            input_features = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(self.device)  # (batch, seq_len, features)
            output = self.model(input_features)
            predicted_profit = output.item()
        return predicted_profit

    def objective_function(self, actions, historical_features, charges=20):
        
    def optimize_actions(self, historical_features):
        # Start with all hold actions
        initial_actions = np.zeros(self.time_horizon)
        
        bounds = [(-1, 1) for _ in range(self.time_horizon)]  # -1: sell, 0: hold, 1: buy
        
        result = minimize(self.objective_function, initial_actions, args=(historical_features,),
                          bounds=bounds, method='SLSQP')
        
        optimal_actions = np.round(result.x)  # final action sequence rounded to integers
        return optimal_actions

    def optimize_actions(self, historical_prices):
        # Initial guess for actions (0: Hold, 1: Buy, -1: Sell)
        initial_actions = np.zeros(self.time_horizon)

        # Constraints: Ensure that we do not buy/sell more than the available balance
        bounds = [(0, 1) for _ in range(self.time_horizon)]  # Buy or hold actions
        
        # Optimize the actions using the objective function
        result = minimize(self.objective_function, initial_actions, args=(historical_prices,),
                          bounds=bounds, method='SLSQP')
        
        return result.x  # Optimal actions (0: Hold, 1: Buy, -1: Sell)

"""
This LSTM model takes the last 60 days of closing prices to predict the next day's closing price.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load stock data (for example, AAPL stock data)
stock_data = pd.read_csv('AAPL_data.csv', date_parser=True, index_col='Date')

# Preprocess the stock data (scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# Function to create dataset for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Create dataset with a time window of 60 days
X, y = create_dataset(scaled_data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape data for LSTM input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(units=50, return_sequences=False),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the trained model
model.save('lstm_stock_model.h5')



### Step 2: Set Up MPC to Optimize Buy/Sell/Hold
"""

We will now set up an **MPC controller** that uses the
 **LSTM predictions** for stock prices to decide on **buy/sell/hold** actions.
   The MPC will optimize the future reward, which is based on the predicted stock price.

"""
from scipy.optimize import minimize
import numpy as np
import tensorflow as tf

# Load the trained LSTM model
model = tf.keras.models.load_model('lstm_stock_model.h5')

# Function to predict the next stock price using the LSTM model
def predict_stock_price(data):
    prediction = model.predict(data)
    return prediction[0][0]

# Define MPC Controller class
class MPCController:
    def __init__(self, model, time_horizon, initial_balance=10000):
        self.model = model
        self.time_horizon = time_horizon
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = 0  # Current stock position (number of shares held)
        
    def objective_function(self, actions, historical_prices):
        """
        Objective function to maximize the cumulative reward (profit).
        `actions`: List of buy/sell/hold actions
        `historical_prices`: Stock prices for the time horizon
        """
        balance = self.initial_balance
        positions = 0
        total_reward = 0
        for t in range(self.time_horizon):
            action = actions[t]
            price = historical_prices[t]

            if action == 1:  # Buy
                positions += balance // price
                balance -= positions * price
            elif action == -1:  # Sell
                balance += positions * price
                positions = 0

            # Reward: The value of the portfolio
            total_reward = balance + positions * price
        
        return -total_reward  # We minimize the negative reward
    
    def optimize_actions(self, historical_prices):
        # Initial guess for actions (0: Hold, 1: Buy, -1: Sell)
        initial_actions = np.zeros(self.time_horizon)

        # Constraints: Ensure that we do not buy/sell more than the available balance
        bounds = [(0, 1) for _ in range(self.time_horizon)]  # Buy or hold actions
        
        # Optimize the actions using the objective function
        result = minimize(self.objective_function, initial_actions, args=(historical_prices,),
                          bounds=bounds, method='SLSQP')
        
        return result.x  # Optimal actions (0: Hold, 1: Buy, -1: Sell)

# Function to simulate and get buy/sell/hold signals
def simulate_mpc(historical_prices, mpc_controller):
    # Get optimal actions for the given prices
    optimal_actions = mpc_controller.optimize_actions(historical_prices)

    # Map actions to Buy/Sell/Hold
    actions = []
    for action in optimal_actions:
        if action > 0.5:
            actions.append("BUY")
        elif action < -0.5:
            actions.append("SELL")
        else:
            actions.append("HOLD")
    
    return actions

# Example: Use MPC to decide buy/sell/hold for a given period
historical_prices = np.array([120, 122, 125, 130, 128, 132, 135, 137, 138])  # Example stock prices

# Create MPC controller
mpc_controller = MPCController(model=model, time_horizon=5)

# Simulate MPC and get actions
actions = simulate_mpc(historical_prices, mpc_controller)
print("MPC Buy/Sell/Hold actions:", actions)

