import matplotlib.pyplot as plt
from matplotlib import animation
import plotly.graph_objs as go

def update_plot(frame, df, ax1, ax2,data_cb):
    """Update the plot in real-time."""
    df = data_cb()
    
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

def live_plot(init_dfd,data_cb):
    # Initialize the plot
    def update(frame,df,ax1,ax2):
        update_plot(frame,df,ax1,ax2,data_cb)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Set up the animation
    ani = animation.FuncAnimation(fig, update, fargs=(init_dfd, ax1, ax2), interval=5000)  # Update every 5 seconds
    
    # Show the plot
    plt.show()

def plot_signal_go(data):
    fig = go.Figure(data=[
        go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'], name="Candles"),
        go.Scatter(x=data.index, y=data['EMA9'], mode='lines', name='EMA9', line=dict(color='blue')),
        go.Scatter(x=data.index, y=data['Support'], mode='lines', name='Support', line=dict(color='green', dash='dot')),
        go.Scatter(x=data.index, y=data['Resistance'], mode='lines', name='Resistance', line=dict(color='red', dash='dot')),
    ])
    fig.update_layout(title=f"{data.symbol} - 1-Min Scalping Chart", xaxis_title="Time", yaxis_title="Price")
    fig.show()

   

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
 

# Visualize the candlestick chart with detected patterns
def plot_candlestick_with_patterns(df):
    fig = go.Figure(data=[
        go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'], name="Candles"),
        # go.Scatter(x=data.index, y=data['EMA_9'], mode='lines', name='EMA9', line=dict(color='blue')),
        # go.Scatter(x=data.index, y=data['Support'], mode='lines', name='Support', line=dict(color='green', dash='dot')),
        # go.Scatter(x=data.index, y=data['Resistance'], mode='lines', name='Resistance', line=dict(color='red', dash='dot')),
        #  go.Scatter(x=data.index, y=data['Bullish'], marker=dict(symbol='triangle-up-open',color='green')),
        # go.Scatter(x=data.index, y=data['Bearish'], marker=dict(symbol='triangle-down-open',color='red')),

    ])
    # fig.update_layout(title=f"{data.symbol} - 1-Min Scalping Chart", xaxis_title="Time", yaxis_title="Price")
    fig.show()
   
    return
    # Mark patterns on the chart
    for pattern in pattern_results.columns:
        for i in range(len(pattern_results)):
            if pattern_results[pattern][i] > 0:  # Bullish pattern
                plt.scatter(i, df['low'][i] - 2, marker='^', color='green', s=100)
            elif pattern_results[pattern][i] < 0:  # Bearish pattern
                plt.scatter(i, df['high'][i] + 2, marker='v', color='red', s=100)
    
    plt.title('Candlestick Chart with Detected Patterns')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(df)), df.index.strftime('%Y-%m-%d'), rotation=45)
    plt.tight_layout()
    plt.show()

def load():
    import pandas as pd
    from pattern_detector import capture_technical_indicators,get_signal_from_candlestick_pattern

    df = pd.read_csv("data.csv",keep_date_col=True,index_col="Datetime")
    df.index = pd.to_datetime(df.index.to_list())
    df.index.name = "Datetime"
    data = df.head(100)

    data_with_ti = capture_technical_indicators(data).bfill().ffill()
    data_with_cd_sig = get_signal_from_candlestick_pattern(data)
    final_data = pd.concat([data,data_with_ti,data_with_cd_sig],axis=1)
    final_data.to_csv("data_with_tp.csv")
    # plot_candlestick_with_patterns(data)



if __name__ == "__main__":
    import pandas as pd
    from pattern_detector import capture_technical_indicators,get_signal_from_candlestick_pattern

    df = pd.read_csv("data_with_tp.csv",keep_date_col=True,index_col="Datetime")
    df.index = pd.to_datetime(df.index.to_list())
    df.index.name = "Datetime"
    data = df.head(100)

    plot_candlestick_with_patterns(data)