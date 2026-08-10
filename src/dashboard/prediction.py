class TaxApplicability:
    TYPE_BUY_ONLY = 1
    TYPE_SELL_ONLY = 2
    TYPE_BUY_SELL = 3
    EXCHANGE_NSE = "NSE"
    EXCHANGE_BSE = "BSE"
    TRANSACTION_BUY = "buy"
    TRANSACTION_SELL = "sell"
    GST = 18
    ExchangeTransactionChargesNSE = 0.00297
    ExchangeTransactionChargesBSE = 0.00375
    InvestorProtectionFundTrustChargeNSE = 0.0001
    InvestorProtectionFundTrustChargeBSE = 0
    SEBITurnoverCharges = 0.0001
    StampDuty = 0.003
    STT = 0.025
    BROKERAGE = 0.1
    MIN_BROKERAGE = 2
    MAX_BROKERAGE = 20
    
    def __init__(self,exchange=EXCHANGE_NSE,transaction_type=TRANSACTION_BUY):
        self.exchange = exchange
        self.transaction=transaction_type
    def get_exchange_transaction_charges(self):
        if self.exchange == self.EXCHANGE_NSE:
            return self.ExchangeTransactionChargesNSE*self.turnover/100
        else:
            return self.ExchangeTransactionChargesBSE*self.turnover/100
    def set_trasaction_info(self,n_stocks,u_price):
        self.turnover = n_stocks*u_price
    def sebi_turnover_charges(self):
        return self.SEBITurnoverCharges*self.turnover/100
    def investor_protection_fund_trust_charge(self):
        if self.exchange == self.EXCHANGE_NSE:
            return self.InvestorProtectionFundTrustChargeNSE*self.turnover/100
        else:
            return self.InvestorProtectionFundTrustChargeBSE*self.turnover/100
    def gst(self,tax_amount):
        return self.GST*tax_amount/100
    def stamp_duty(self):
        if self.transaction == self.TRANSACTION_BUY:
            return self.StampDuty*self.turnover/100
        return 0
    def brokerage(self):
        return max(min(self.MAX_BROKERAGE,self.BROKERAGE*self.turnover/100),self.MIN_BROKERAGE)
    def get_stt(self):
        if self.transaction == self.TRANSACTION_SELL:
            return self.STT**self.turnover/100
        return 0
    def calculate_tax(self):
        tax = self.get_stt() + self.brokerage() + self.stamp_duty() + self.investor_protection_fund_trust_charge() \
        + self.sebi_turnover_charges() + self.get_exchange_transaction_charges()
        return tax
    def calculate_total_tax(self):
        tax = self.calculate_tax()
        return tax + self.gst(tax)
    
def get_signal(symbol, interval='1m', lookback=60, plot=False):
    # Get the stock data
    # Fetch the live market data (last available data)
    data = get_last_n_min_data(symbol,minutes=lookback,interval=interval)
    # data = yf.download(tickers=symbol, interval=interval, period=lookback, progress=False)
    data.dropna(inplace=True)

    # Technical indicators for scalping
    data['EMA9'] = ta.trend.ema_indicator(data['Close'], window=9)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=7).rsi()
    data['Support'] = data['Low'].rolling(window=15).min()
    data['Resistance'] = data['High'].rolling(window=15).max()

    last = data.iloc[-1]
    prev = data.iloc[-2]

    signal = "HOLD"
    if last['Close'] > last['EMA9'] and prev['RSI'] < 30 and last['RSI'] > 30 and last['Close'] >= last['Support']:
        signal = "BUY"
    elif last['Close'] < last['EMA9'] and prev['RSI'] > 70 and last['RSI'] < 70 and last['Close'] <= last['Resistance']:
        signal = "SELL"

    if plot:
        fig = go.Figure(data=[
            go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                           low=data['Low'], close=data['Close'], name="Candles"),
            go.Scatter(x=data.index, y=data['EMA9'], mode='lines', name='EMA9', line=dict(color='blue')),
            go.Scatter(x=data.index, y=data['Support'], mode='lines', name='Support', line=dict(color='green', dash='dot')),
            go.Scatter(x=data.index, y=data['Resistance'], mode='lines', name='Resistance', line=dict(color='red', dash='dot')),
        ])
        fig.update_layout(title=f"{symbol} - 1-Min Scalping Chart", xaxis_title="Time", yaxis_title="Price")
        fig.show()

    return {
        "Symbol": symbol,
        "Close": round(last['Close'], 2),
        "RSI": round(last['RSI'], 2),
        "EMA9": round(last['EMA9'], 2),
        "Support": round(last['Support'], 2),
        "Resistance": round(last['Resistance'], 2),
        "Signal": signal
    }


def update_plot(frame, df, ax1, ax2):
    """Update the plot in real-time."""
    # Simulate new data arriving every frame
    new_data = generate_new_data(df['Datetime'].iloc[-1])
    df = df.append(new_data, ignore_index=True)
    
    # Re-calculate indicators
    df = capture_trend(df)
    df = capture_bought_status(df)
    
    # Clear previous plot
    ax1.clear()
    ax2.clear()

    # Plot Close price and Bollinger Bands
    ax1.plot(df['Datetime'], df['Close'], label='Close Price', color='blue')
    ax1.plot(df['Datetime'], df['bollinger_upper'], label='Bollinger Upper Band', color='red', linestyle='--')
    ax1.plot(df['Datetime'], df['bollinger_lower'], label='Bollinger Lower Band', color='green', linestyle='--')
    
    # Plot Buy and Sell signals
    buy_signals = df[df['RSI'] < 30]
    sell_signals = df[df['RSI'] > 70]
    ax1.scatter(buy_signals['Datetime'], buy_signals['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
    ax1.scatter(sell_signals['Datetime'], sell_signals['Close'], marker='v', color='r', label='Sell Signal', alpha=1)
    
    # Formatting for price chart
    ax1.set_xlabel('Datetime')
    ax1.set_ylabel('Price')
    ax1.set_title('Real-Time Stock Data with Buy/Sell Signals')
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.legend(loc='best')

    # Plot RSI on secondary axis
    ax2.plot(df['Datetime'], df['RSI'], label='RSI', color='orange', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.legend(loc='upper left')

    # Rotate the x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()

    return df
def generate_new_data(last_time):
    """Generate new data as if it were coming from a live source."""
    new_time = last_time + timedelta(minutes=1)
    new_close = np.random.normal(loc=1125, scale=2)  # Random close price around 1125
    new_data = {
        'Datetime': new_time,
        'Close': new_close,
        'Volume': np.random.randint(1000, 2000),  # Random volume
    }
    return pd.DataFrame([new_data])

def capture_trend(df, short_window=5, long_window=20):
    """
    Capture the trend using SMA and EMA.
    Args:
    df (DataFrame): DataFrame containing the stock price data.
    short_window (int): The period for the short-term SMA and EMA.
    long_window (int): The period for the long-term SMA and EMA.
    
    Returns:
    df (DataFrame): The original DataFrame with trend-related columns.
    """
    # Simple Moving Average (SMA)
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()  # Short-term SMA
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()    # Long-term SMA
    
    # Exponential Moving Average (EMA)
    df['EMA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()  # Short-term EMA
    df['EMA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()   # Long-term EMA

    return df

def capture_momentum(df):
    """
    Capture momentum using MACD, Stochastic Oscillator, and Volume.
    Args:
    df (DataFrame): DataFrame containing the stock price data.
    
    Returns:
    df (DataFrame): The original DataFrame with momentum-related columns.
    """
    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal line

    # Stochastic Oscillator
    df['14_low'] = df['Low'].rolling(window=14).min()
    df['14_high'] = df['High'].rolling(window=14).max()
    df['stochastic'] = 100 * (df['Close'] - df['14_low']) / (df['14_high'] - df['14_low'])
    
    # Volume: Using the actual Volume data
    df['volume_avg'] = df['Volume'].rolling(window=20).mean()  # 20-period moving average of volume

    return df

def capture_bought_status(df, rsi_period=14, bollinger_window=20, std_dev=2):
    """
    Capture buy status using RSI and Bollinger Bands.
    Args:
    df (DataFrame): DataFrame containing the stock price data.
    rsi_period (int): The period for calculating RSI.
    bollinger_window (int): The period for calculating Bollinger Bands.
    std_dev (int): The number of standard deviations for Bollinger Bands.
    
    Returns:
    df (DataFrame): The original DataFrame with buy status-related columns.
    """
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))  # RSI formula

    # Bollinger Bands
    df['bollinger_middle'] = df['Close'].rolling(window=bollinger_window).mean()
    df['bollinger_std'] = df['Close'].rolling(window=bollinger_window).std()
    df['bollinger_upper'] = df['bollinger_middle'] + (df['bollinger_std'] * std_dev)
    df['bollinger_lower'] = df['bollinger_middle'] - (df['bollinger_std'] * std_dev)

    return df
    
def decision_signal(df):
    """
    Decides Buy, Sell, or Hold based on the technical indicators.
    """
    signals = []

    for i in range(len(df)):
        buy_signal = False
        sell_signal = False

        # Buy Conditions
        if df['Close'].iloc[i] > df['EMA_short'].iloc[i] and df['RSI'].iloc[i] < 30:
            buy_signal = True
        if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and df['Close'].iloc[i] < df['bollinger_lower'].iloc[i]:
            buy_signal = True

        # Sell Conditions
        if df['Close'].iloc[i] < df['EMA_short'].iloc[i] and df['RSI'].iloc[i] > 70:
            sell_signal = True
        if df['MACD'].iloc[i] < df['MACD_signal'].iloc[i] and df['Close'].iloc[i] > df['bollinger_upper'].iloc[i]:
            sell_signal = True

        # Hold Conditions: If no buy or sell signal
        if not buy_signal and not sell_signal:
            signals.append('Hold')
        elif buy_signal:
            signals.append('Buy')
        elif sell_signal:
            signals.append('Sell')

    df['Signal'] = signals
    return df

def live_plot():
    # Initialize the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Initial data frame with one data point
    initial_data = {
        'Datetime': [datetime.now()],
        'Close': [1125],
        'Volume': [1000],
    }
    df = pd.DataFrame(initial_data)
    
    # Set up the animation
    ani = animation.FuncAnimation(fig, update_plot, fargs=(df, ax1, ax2), interval=5000)  # Update every 5 seconds
    
    # Show the plot
    plt.show()

def plot_signal(df2):
    df = df2.copy()
    # Visualization
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    df.index = df.index.strftime('%H:%M') 
    
    # Plot Close price and Bollinger Bands
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.75)
    ax1.plot(df.index, df['bollinger_upper'], label='Bollinger Upper Band', color='red', linestyle='--', alpha=0.5)
    ax1.plot(df.index, df['bollinger_lower'], label='Bollinger Lower Band', color='green', linestyle='--', alpha=0.5)
    
    # Plot Buy and Sell signals
    buy_signals = df[df['Signal'] == 'Buy']
    sell_signals = df[df['Signal'] == 'Sell']
    ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
    ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal', alpha=1)
    
    # Formatting
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.set_title('Price Chart with Buy/Sell Signals and Bollinger Bands')
    ax1.legend(loc='best')
    # ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Plot RSI on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['RSI'], label='RSI', color='orange', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.legend(loc='upper left')
    
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()