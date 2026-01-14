from data_fetcher import get_ticker,get_last_n_min_data,load_stock_data
from features.pattern_detector import capture_trend,capture_momentum,capture_bought_status,decision_signal,get_signal
from graph_plotter import plot_signal
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go


tickers = get_ticker("SBI Life")
ticker = tickers.iloc[0].ticker 


df = load_stock_data(ticker)

# Capture trend, momentum, and buy status with the provided sample data
df = capture_trend(df)
df = capture_momentum(df)
df = capture_bought_status(df)
df = decision_signal(df)

plot_signal(df.tail(10))

results = get_signal(ticker, interval='1m', lookback=60, plot=True)
results

from features.feature_builder import FeatureBuilder

if __name__ == '__main__':
    # ticker = "TCS.NS"
    # feature_builder = FeatureBuilder(ticker)  

    # features_fundm = feature_builder.get_fundamental_features()
    # features_fundm_df = pd.DataFrame(features_fundm,index=[ticker])

    # features_macro = feature_builder.get_macroeconomic_features()
    # features_macro_df = pd.DataFrame(features_macro).bfill().ffill()
    # features_macro_item = pd.DataFrame(features_macro_df.iloc[-1]).T
    # features_macro_item.index = [ticker]

    # fetures_df = pd.concat([features_fundm_df,features_macro_item],axis=1)

    from stock_selection.pattern_detector import Tool

    tool = Tool()
    
    tool.add_technical_params()