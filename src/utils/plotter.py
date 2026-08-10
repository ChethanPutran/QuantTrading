import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import pandas_ta as ta
from plotly.subplots import make_subplots

def plot_stok_info(data):

    ticker= "NIFTY-50"

    # --- STEP 3: Create Subplots ---
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.4,0.05, 0.15, 0.15, 0.15],
                        subplot_titles=('Price Chart', 'RSI', 'MACD', 'Stochastic Oscillator'))

    # --- Row 1: Candlestick Chart ---
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Candlestick'),
                row=1, col=1)

    # --- Row 2: RSI ---
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='orange'), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', row=3, col=1)

    # --- Row 3: MACD ---
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], line=dict(color='blue'), name='MACD'), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACDs'], line=dict(color='red'), name='Signal'), row=4, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['MACDh'], name='Histogram'), row=4, col=1)

    # --- Row 4: Stochastic Oscillator ---
    fig.add_trace(go.Scatter(x=data.index, y=data['STOCHk'], line=dict(color='blue'), name='%K'), row=5, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['STOCHd'], line=dict(color='orange'), name='%D'), row=5, col=1)
    fig.add_hline(y=80, line_dash='dash', line_color='red', row=5, col=1)
    fig.add_hline(y=20, line_dash='dash', line_color='green', row=5, col=1)

    # --- Layout Settings ---
    fig.update_layout(height=1000, width=1200, showlegend=False,
                    title_text=f'Technical Analysis Dashboard - {ticker}',
                    xaxis4_rangeslider_visible=False)

    fig.show()

