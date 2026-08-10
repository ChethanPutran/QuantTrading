import pandas as pd
import numpy as np
try:
    from dash import Dash, dcc, html
    from dash.dependencies import Input, Output
    DASH_AVAILABLE = True
except Exception:
    DASH_AVAILABLE = False
from datetime import datetime, timedelta
import random

# Dashboard integration helper
from .integration import start_system_in_thread, get_stats, process_tick

if DASH_AVAILABLE:
    import plotly.graph_objs as go

    # Initialize the Dash app
    app = Dash(__name__)
    app.title = "Real-Time Trading Dashboard"

    # Starting dataframe
    df = pd.DataFrame({
        "Datetime": [datetime.now()],
        "Close": [1125.0],
        "Volume": [random.randint(1000, 2000)]
    })

    # Indicator functions
    def capture_trend(df, short_window=5, long_window=20):
        df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
        df['EMA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df['EMA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
        return df

    def capture_momentum(df):
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        low_14 = df['Close'].rolling(window=14).min()
        high_14 = df['Close'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        return df

    def capture_bought_status(df, rsi_period=14, bollinger_window=20, std_dev=2):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['bollinger_middle'] = df['Close'].rolling(window=bollinger_window).mean()
        df['bollinger_std'] = df['Close'].rolling(window=bollinger_window).std()
        df['bollinger_upper'] = df['bollinger_middle'] + (df['bollinger_std'] * std_dev)
        df['bollinger_lower'] = df['bollinger_middle'] - (df['bollinger_std'] * std_dev)
        return df

    def get_trade_signal(row):
        if row['RSI'] < 30 and row['MACD'] > row['Signal_Line']:
            return 'BUY'
        elif row['RSI'] > 70 and row['MACD'] < row['Signal_Line']:
            return 'SELL'
        else:
            return 'HOLD'

    def generate_new_row(last_time):
        new_time = last_time + timedelta(minutes=1)
        new_close = np.random.normal(loc=1125, scale=2)
        new_volume = random.randint(1000, 2000)
        return pd.DataFrame([{
            'Datetime': new_time,
            'Close': new_close,
            'Volume': new_volume
        }])

    # Layout
    app.layout = html.Div([
        html.H1("Real-Time Buy/Sell Trading Signals", style={'textAlign': 'center'}),
        dcc.Graph(id='live-graph', style={"height": "70vh"}),
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # 5 seconds
            n_intervals=0
        )
    ])

    # Update graph callback
    @app.callback(
        Output('live-graph', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_graph(n):
        global df
        new_data = generate_new_row(df['Datetime'].iloc[-1])
        df = pd.concat([df, new_data], ignore_index=True)
        df = capture_trend(df)
        df = capture_momentum(df)
        df = capture_bought_status(df)
        df['Signal'] = df.apply(get_trade_signal, axis=1)

        fig = go.Figure()

        # Close price line
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Close'],
                                 mode='lines+markers', name='Close Price',
                                 line=dict(color='blue')))

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['bollinger_upper'],
                                 name='Upper Band', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['bollinger_lower'],
                                 name='Lower Band', line=dict(color='green', dash='dot')))

        # Buy/Sell signals
        buy_signals = df[df['Signal'] == 'BUY']
        sell_signals = df[df['Signal'] == 'SELL']

        fig.add_trace(go.Scatter(x=buy_signals['Datetime'], y=buy_signals['Close'],
                                 mode='markers', name='Buy Signal',
                                 marker=dict(symbol='triangle-up', color='lime', size=12)))

        fig.add_trace(go.Scatter(x=sell_signals['Datetime'], y=sell_signals['Close'],
                                 mode='markers', name='Sell Signal',
                                 marker=dict(symbol='triangle-down', color='red', size=12)))

        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_dark',
            legend=dict(x=0, y=1.1, orientation='h'),
            margin=dict(t=50, b=20),
        )

        return fig

# Run app
if __name__ == '__main__':
    # Start the trading system in background for live data (best-effort)
    start_system_in_thread()
    if DASH_AVAILABLE:
        app.run(debug=True)
    else:
        print('Dash is not installed in the environment. Install `dash` to run the dashboard.')
