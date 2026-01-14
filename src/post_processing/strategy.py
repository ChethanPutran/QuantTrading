import yfinance as yf
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
from pattern_detector import Tool

df = pd.read_csv("RELIANCE.NS_data.csv")

data = df.head(300)


tool = Tool()

data_with_tp = tool.add_technical_params(data)

resistance =  data_with_tp['Resistance']
support = data_with_tp['Support']


if 




