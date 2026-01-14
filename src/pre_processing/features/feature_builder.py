import yfinance as yf
import pandas_ta as ta
import pandas as pd
from dotenv import load_dotenv
import os
import openbb
import numpy as np
from fredapi import Fred



load_dotenv("../.env")
FRED_API_KEY = os.environ.get("FRED_API_KEY")


class FeatureBuilder:
    def __init__(self,ticker):
        self.ticker = ticker
        self.fred = Fred(api_key=FRED_API_KEY)
        self.features = {}
        
    def get_technical_features(self):
        df = yf.download(self.ticker, period='6mo', interval='1d')

        # Compute indicators
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.sma(length=9, append=True)
        df.ta.sma(length=21, append=True)
        df.ta.ema(length=12, append=True)
        df.ta.bbands(append=True)
        df.ta.adx(append=True)
        df.ta.atr(append=True)
        df.ta.cci(append=True)
        df.ta.mfi(append=True)
        df.ta.obv(append=True)
        df.ta.stoch(append=True)
        df.ta.ichimoku(append=True)
        df.ta.mom(append=True)
        df.ta.roc(append=True)
        
        # Custom signals
        df['golden_cross_flag'] = (df['SMA_50'] > df['EMA_12']).astype(int)
        df['death_cross_flag'] = (df['SMA_50'] < df['EMA_12']).astype(int)
        df['bollinger_band_width'] = df['BBU_20_2.0'] - df['BBL_20_2.0']
        df['bollinger_%b'] = (df['Close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        
        # Latest values
        latest = df.iloc[-1]
        features = {
            'rsi': latest['RSI_14'],
            'macd': latest['MACD_12_26_9'],
            'macd_signal': latest['MACDs_12_26_9'],
            'sma_50': latest['SMA_50'],
            'ema_12': latest['EMA_12'],
            'bollinger_band_width': latest['bollinger_band_width'],
            'bollinger_%b': latest['bollinger_%b'],
            'adx': latest['ADX_14'],
            'atr': latest['ATRr_14'],
            'cci': latest['CCI_14_0.015'],
            'mfi': latest['MFI_14'],
            'obv': latest['OBV'],
            'stochastic_k': latest['STOCHk_14_3_3'],
            'stochastic_d': latest['STOCHd_14_3_3'],
            'ichimoku_cloud': latest['ISA_9'],
            'momentum': latest['MOM_10'],
            'roc': latest['ROC_10'],
            'golden_cross_flag': latest['golden_cross_flag'],
            'death_cross_flag': latest['death_cross_flag'],
        }
        
        # Filter out NaNs
        features_cleaned = {k: v for k, v in features.items() if pd.notna(v)}
        
        return features_cleaned
        
    def add_technical_features(self,features):
        self.features = {**self.features,**features}
    
    def get_fundamental_features(self):
        stock = yf.Ticker(self.ticker)
        info = stock.info
        
        fundamentals = {
            'market_cap':info.get("marketCap"),
            'eps_ttm': info.get('trailingEps'),
            'pe_ratio': info.get('trailingPE'),
            'peg_ratio': info.get('pegRatio'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'ev_ebitda': info.get('enterpriseToEbitda'),
            'revenue_growth_yoy': info.get('revenueGrowth'),
            'net_margin': info.get('netMargins'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'dividend_yield': info.get('dividendYield'),
            'free_cash_flow_yield': info.get('freeCashflow') / info.get('marketCap', 1),
            'buyback_yield': info.get('buyBackYield'),
            'debt_to_equity': info.get('debtToEquity'),
            'interest_coverage': info.get('ebitda') / info.get('interestExpense') if info.get('interestExpense') else None,
            'institutional_holdings': info.get('heldPercentInstitutions'),
            'insider_holdings': info.get('heldPercentInsiders'),
        } 
        return fundamentals

    def add_fundamental_features(self,features):
        self.features = {**self.features,**features}

    def get_macroeconomic_features(self):
        macro_features = {
        'cpi': self.fred.get_series('CPIAUCSL'),
        'core_inflation': self.fred.get_series('CPILFESL'),
        'unemployment_rate': self.fred.get_series('UNRATE'),
        'gdp_growth': self.fred.get_series('A191RL1Q225SBEA'),
        'fed_rate': self.fred.get_series('FEDFUNDS'),
        'real_interest_rate': self.fred.get_series('INTDSRUSM193N'),
        'yield_curve_spread': self.fred.get_series('T10Y2Y'),
        'dxy': self.fred.get_series('DTWEXBGS'),
        'oil_price': self.fred.get_series('DCOILWTICO'),
        'gold_price': self.fred.get_series('IR14270'),
        'vix': self.fred.get_series('VIXCLS'),
        'retail_sales_growth': self.fred.get_series('MRTSSM44X72USS'),
        'trade_balance': self.fred.get_series('NETEXP'),
        }
        return macro_features
        
    def add_macroeconomic_features(sdelf,features):
        self.features = {**self.features,**features}
        
    def get_fred_data(self,series_id):
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'sort_order': 'desc'
        }
        r = requests.get(url, params=params)
        data = r.json()
        return float(data['observations'][0]['value'])

    def get_market_sentiments_features(self):
        ### Sentiment, News, and Options Flow
        news_sentiment = openbb.stocks.sia.sentiment(self.ticker)
        ### Options Flow & Greeks
        # Options flow and open interest
        options_oi = openbb.stocks.options.oi(self.ticker)
        options_flow = openbb.stocks.options.unusual(self.ticker)

        sentiment_features = {
            'options_oi':options_oi,
            'options_flow': options_flow,
            'news_sentiment': news_sentiment
        }
        return sentiment_features

    def add_market_sentiments_features(self,features):
        self.features = {**self.features,**features}

    def build_full_feature_vector(self):
        features = self.get_fundamental_features()
        self.add_fundamental_features(features)

        features = self.get_technical_features()
        self.add_technical_features(features)

        features = self.get_macroeconomic_features()
        self.add_macroeconomic_features(features)
        
        features = self.get_market_sentiments_features()
        self.add_market_sentiments_features(features)

    def get_features(self):
        return self.features


def compute_quarterly_metrics(ticker,start_date,end_date):
    stock = yf.Ticker(ticker)

    # Pull quarterly statements
    income = stock.quarterly_income_stmt.T
    balance = stock.quarterly_balance_sheet.T
    info = stock.info
    
    price = info.get("currentPrice", None)
    shares = info.get("sharesOutstanding", None)
    market_cap = info.get("marketCap", price * shares)
    
    df = pd.concat([income,balance],axis=1).sort_index(ascending=False)
    df = df[(df.index >= start_date) & (df.index <= end_date)].astype('float').bfill().ffill()
    
    # Compute Profitability Ratios
    df['EBITDA Margin'] = df['EBITDA'] / df['Total Revenue']
    df['EBIT Margin'] = df['EBIT'] / df['Total Revenue']
    df['Net Profit Margin'] = df['Net Income'] / df['Total Revenue']
    df['Basic EPS'] = df['Net Income'] / df['Basic Average Shares']
    
    # Compute Liquidity Ratios
    df['Current Ratio'] = df['Current Assets'] / df['Current Liabilities']
    df['Quick Ratio'] = (df['Current Assets'] - df['Inventory']) / df['Current Liabilities']
    
    # Compute Leverage Ratios
    df['Debt-to-Equity'] = df['Total Debt'] / df['Stockholders Equity']
    df['Debt Ratio'] = df['Total Debt'] / df['Total Assets']
    # df['Long-Term Debt-to-Equity'] = df['Long Term Debt'] / df['Stockholders Equity']
    
    # Compute Efficiency Ratios
    df['Asset Turnover'] = df['Total Revenue'] / df['Total Assets']
    df['Inventory Turnover'] = df['Cost Of Revenue'] / df['Inventory']
    
    # Compute Additional Fundamental Ratios
    df['EPS'] = df['Net Income'] / df['Basic Average Shares']  # Earnings per Share
    df['P/E Ratio'] = price / df['EPS']  # Price to Earnings Ratio
    df['P/B Ratio'] = price  / (df['Stockholders Equity'] / shares ) # Price to Book Ratio
    df['P/S Ratio'] = price  / (df['Total Revenue'] / shares)  # Price to Sales Ratio
    df['ROE'] = df['Net Income'] / df['Stockholders Equity']  # Return on Equity
    df['Net Margin'] = df['Net Income'] / df['Total Revenue']  # Net Profit Margin
    
    # Compute EV/EBITDA - Requires Market Cap and Cash equivalents
    df['EV'] = market_cap + df['Total Debt'] - df['Cash And Cash Equivalents']  # Enterprise Value
    df['EV/EBITDA'] = df['EV'] / df['EBITDA']  # EV/EBITDA
    
    # Compute Revenue and Net Income - Already included, but can explicitly display:
    df['Revenue'] = df['Total Revenue']
    df['Net Income'] = df['Net Income']  # You may already have this column
    return df

if __name__ == '__main__':
    ticker = "TCS.NS"
    feature_builder = FeatureBuilder(ticker)  

    features_fundm = feature_builder.get_fundamental_features()
    features_fundm_df = pd.DataFrame(features_fundm,index=[ticker])

    features_macro = feature_builder.get_macroeconomic_features()
    features_macro_df = pd.DataFrame(features_macro).bfill().ffill()
    features_macro_item = pd.DataFrame(features_macro_df.iloc[-1]).T
    features_macro_item.index = [ticker]

    fetures_df = pd.concat([features_fundm_df,features_macro_item],axis=1)