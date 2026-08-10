import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from datetime import datetime, timedelta
import random
from data_fetcher import get_ticker,get_last_n_min_data,load_stock_data
from pattern_detector import capture_trend,capture_momentum,capture_bought_status,decision_signal,get_signal




# Initialize the Dash app
app = Dash(__name__)
app.title = "Real-Time Trading Dashboard"

status,data = load_stock_data("RELIANCE.NS")
data['Time'] = data.index.strftime('%H:%M')

# Initialize figure outside the callback
fig = go.Figure()

# Initialize the figure with Close Price trace
fig.add_trace(go.Scatter(x=[], y=[],
                         mode='lines+markers', name='Close Price',
                         line=dict(color='blue')))

# Initialize data store
data_store = dcc.Store(id='data_store', data=[])

# Interval to update data
interval = dcc.Interval(id='data_interval', interval=1000, n_intervals=0)  # update every 5 second


# Main layout of the application
app.layout = html.Div([
    data_store,
    interval,
    html.H1("Real-Time Buy/Sell Trading Signals", style={'textAlign': 'center'}),
    html.Button("Start TradingSystem", id='start_system_btn', n_clicks=0),
    html.Div(id='system_status'),
    dcc.Graph(id='realtime_graph',style={"height": "70vh"}),
    html.Div(id='data_display')  # Display data for verification
])

# Update graph callback
@app.callback(
    Output('data_store', 'data'),
    Input('data_store', 'data'),
    Input('data_interval', 'n_intervals')
)
def update_data(old_data,n_intervals):
    global data
    old_data.append(data.iloc[[n_intervals]].to_dict(orient='records')[0])
    return old_data[-50:]  # Keep last 50 points


# Update graph callback
@app.callback(
    Output('realtime_graph', 'figure'),
    Input('data_store', 'data')
)
def update_graph(data_json):
    data = pd.DataFrame(data_json)

    # Close price line
    
    # Bollinger Bands
    # fig.add_trace(go.Scatter(x=df.index, y=df['bollinger_upper'],
    #                          name='Upper Band', line=dict(color='red', dash='dot')))
    # fig.add_trace(go.Scatter(x=df.index, y=df['bollinger_lower'],
    #                          name='Lower Band', line=dict(color='green', dash='dot')))

    # # Buy/Sell signals
    # buy_signals = df[df['Signal'] == 'BUY']
    # sell_signals = df[df['Signal'] == 'SELL']

    # fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
    #                          mode='markers', name='Buy Signal',
    #                          marker=dict(symbol='triangle-up', color='lime', size=12)))

    # fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
    #                          mode='markers', name='Sell Signal',
    #                          marker=dict(symbol='triangle-down', color='red', size=12)))

    # Update the figure's trace data

    fig.data[0].x = data['Time']
    fig.data[0].y = data['Close']
    
    # Update layout if needed (optional)
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_dark',
        legend=dict(x=0, y=1.1, orientation='h'),
        margin=dict(t=50, b=20),
        xaxis=dict(
            tickformat='%H:%M',  # Format the x-axis as H:Min (hours:minutes)
            showgrid=True,
            title='Time'
        ),
        yaxis=dict(
            title='Price'
        ),
    )
    return fig


# --- Integration with backend TradingSystem (optional) ---
try:
    from dashboard.integration import start_system_in_thread, get_stats
except Exception:
    start_system_in_thread = None
    get_stats = None


@app.callback(Output('system_status', 'children'), Input('start_system_btn', 'n_clicks'))
def handle_start(n_clicks):
    if n_clicks and start_system_in_thread is not None:
        started = start_system_in_thread()
        return 'TradingSystem started' if started else 'Failed to start TradingSystem'
    return 'TradingSystem not started'


@app.callback(Output('data_display', 'children'), Input('data_interval', 'n_intervals'))
def show_system_stats(n):
    # Display simulator stats if running
    if get_stats is None:
        return 'No TradingSystem integration available.'
    stats = get_stats()
    if not stats:
        return 'TradingSystem not running.'
    sim = stats.get('simulator', {})
    return f"NAV: {sim.get('portfolio_value', sim.get('nav', 'N/A'))} | PnL: {sim.get('final_pnl', 'N/A')}"


# Run app
if __name__ == '__main__':
    app.run(debug=True)
