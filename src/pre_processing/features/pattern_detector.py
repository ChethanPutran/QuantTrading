import pandas as pd
import pandas_ta as ta
# from data_fetcher import load_stock_data
# import talib


class Tool:
    def weighted_sum(self,df_, weights,normalize=True):
        df = df_.bfill().ffill()
        ws = sum(df[column] * weight for column, weight in weights.items())
        if normalize:
            return (ws - ws.min()) / (ws.max() - ws.min())
        return ws
    def capture_trend(self, data, ema_short_length=5, ema_long_length=20, adx_length=14, supt_length=10, supt_multi=3.0,single=False,normalize=True):
        # Technical indicators for scalping
        """
        Trend Indicators
        These show the direction and strength of the market trend.

        Moving Averages (MA, EMA, VWAP)
            SMA: Simple average of closing prices.
            EMA: Weighted for recent data, more responsive.
            VWAP: Volume Weighted Average Price - crucial for institutions.

        MACD (Moving Average Convergence Divergence)
            Shows trend + momentum changes.
            Crossovers of signal line give buy/sell signals.

        ADX (Average Directional Index)
            Measures trend strength (not direction).
            ADX > 25 usually implies a strong trend.

        Supertrend
            Used to determine the direction of the market
        """
        self.weights_trend = {
        'EMAs': 0.1,
        'EMAl': 0.15,
        'ADX': 0.2,
        'DMP': 0.1,
        'DMN': 0.1,
        'SUPERT': 0.1,
        'SUPERTd': 0.05,
        'SUPERTl': 0.05,
        'SUPERTs': 0.05
        }
        trend_params = pd.DataFrame(index=data.index)
        # 📊 Trend Indicators
        trend_params['EMAs'] = data.ta.ema(
            length=ema_short_length)
        trend_params['EMAl'] = data.ta.ema(
            length=ema_long_length)
        trend_params[["ADX", "DMP",
                      "DMN"]] = data.ta.adx(length=adx_length)
        trend_params[["SUPERT", "SUPERTd", "SUPERTl",
                      # ← SUPER INDICATOR
                      "SUPERTs"]] = data.ta.supertrend(length=supt_length, multiplier=supt_multi)
        if single:
            return self.weighted_sum(trend_params,self.weights_trend,normalize)
        return trend_params

    def capture_momentum(self, data, rsi_length=14, stoc_osc_length=14, cci_length=20, mcda_length=12,single=False,normalize=True,append=False):
        """
        Momentum Indicators
        These help spot trend reversals or continuations.

        RSI (Relative Strength Index)
            0-100 scale; >70 = overbought, <30 = oversold.
            Useful for mean-reversion strategies.

        Stochastic Oscillator
            Detects overbought/oversold conditions using high/low ranges.

        CCI (Commodity Channel Index)
            Identifies cyclical trends, breakouts, and reversals.

        MACD (Moving Average Convergence Divergence)
            Helps traders identify changes in the strength, direction, momentum, and duration of a trend in a stock's price
        """
        
        self.weights_momentum = {
            'RSI': 0.15,
            'STOCHk': 0.1,
            'STOCHd': 0.1,
            'CCI': 0.1,
            'MACD': 0.2,
            'MACDh': 0.1,
            'MACDs': 0.15
        }
        momentum_params = pd.DataFrame(index=data.index)
        # ⚡ Momentum Indicators
        momentum_params['RSI'] = data.ta.rsi(length=rsi_length)
        momentum_params[["STOCHk", "STOCHd"]] = data.ta.stoch(
            length=stoc_osc_length)
        momentum_params['CCI'] = data.ta.cci(length=cci_length)
        momentum_params[["MACD", "MACDh",
                         "MACDs"]] = data.ta.macd(length=mcda_length)
        if single:
            return self.weighted_sum(momentum_params,self.weights_momentum,normalize)
        
        if append:
            return pd.concat([data,momentum_params])
        return momentum_params

    def capture_volatility_params(self, data, bbl_length=20, atr_length=14,single=False,normalize=True):
        """
        Volatility Indicators
            They measure how much price is moving.

        Bollinger Bands
            Shows upper and lower bounds based on volatility.
            Price touching bands = possible reversal or breakout.
        ATR (Average True Range)
            Measures market volatility.

        Used for stop loss placement in intraday trades.

        """
        self.weights_volatility = {
            'BBL': 0.1,
            'BBM': 0.1,
            'BBU': 0.1,
            'BBB': 0.1,
            'BBP': 0.1,
            'ATR': 0.5
        }
         
        volatility_params = pd.DataFrame(index=data.index)
        # 📉 Volatility
        volatility_params[["BBL", "BBM", "BBU","BBB", "BBP"]] = data.ta.bbands(length=bbl_length)
        volatility_params['ATR'] = data.ta.atr(length=atr_length)
        if single:
            return self.weighted_sum(volatility_params,self.weights_volatility,normalize)
        return volatility_params

    def capture_volume_params(self, data, vma_length=20,single=False,normalize=True):
        """
        Volume-Based Indicators
        These confirm the strength behind price movements.

        OBV (On-Balance Volume)
            Combines price and volume to detect hidden strength.

        Volume Profile / VWAP
            VWAP is heavily used by day traders to identify fair value areas.

        """
        self.weights_volume = {
            'VMA': 0.25,
            'VWAP': 0.25,
            'OBV': 0.25,
            'VS': 0.25
        }
        volume_params = pd.DataFrame(index=data.index)
        # 📈 Volume
        volume_params['VMA'] = data['Volume'].rolling(window=vma_length).mean()
        volume_params['VWAP'] = data.ta.vwap()
        volume_params['OBV'] = data.ta.obv()

        # Define a Volume Spike (Volume > 2x the 20-period Moving Average)
        volume_params['VS'] = (data['Volume'] > 2 * volume_params['VMA']).astype(int)
        if single:
            return self.weighted_sum(volume_params,self.weights_volume,normalize)
        return volume_params

    def capture_price_action(self, data, support_length=12, resistance_length=15,single=False,normalize=True):
        """
        Price Action Tools (Non-indicator but critical)
        Candlestick Patterns
        E.g., Doji, Hammer, Engulfing, etc.
        Often combined with indicators for confirmation.

        Support & Resistance
        Key levels where price may reverse.

        Often used with breakout strategies.

        """
        
        self.weights_price_action = {
            'SUP': 0.5,
            'RES': 0.5
        }
    
        price_action_params = pd.DataFrame(index=data.index)
        price_action_params['SUP'] = data['Low'].rolling(
            window=support_length).min()
        price_action_params['RES'] = data['High'].rolling(
            window=resistance_length).max()
        if single:
            return self.weighted_sum(price_action_params,self.weights_price_action,normalize)
        return price_action_params

    def decision_signal(self, df):
        """
        Decides Buy, Sell, or Hold based on the technical indicators.
        """
        signals = []

        for i in range(len(df)):
            buy_signal = False
            sell_signal = False

            # Buy Conditions
            if df['Close'].iloc[i] > df['EMAs'].iloc[i] and df['RSI'].iloc[i] < 30:
                buy_signal = True
            if df['MACD'].iloc[i] > df['MACDs'].iloc[i] and df['Close'].iloc[i] < df['BBL'].iloc[i]:
                buy_signal = True

            # Sell Conditions
            if df['Close'].iloc[i] < df['EMAs'].iloc[i] and df['RSI'].iloc[i] > 70:
                sell_signal = True
            if df['MACD'].iloc[i] < df['MACDs'].iloc[i] and df['Close'].iloc[i] > df['BBU'].iloc[i]:
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

    def generate_trading_signals(self, df):
        # Initialize signal column (0 = no action, 1 = buy, -1 = sell)
        df['Signal'] = 0

        # Buy Signal conditions
        buy_condition = (
            (df['RSI'] < 30) &  # Oversold condition
            # EMA crossover (short-term above long-term)
            (df['EMAs'] > df['EMAl']) &
            (df['MACD'] > df['MACDs']) &  # MACD crossover
            (df['SUPERT'] > df['Close']) &  # Supertrend bullish
            (df['VS'] == 1) &  # Volume spike
            (df["Close"] >= df["SUP"]) &  # Close at or above Support
            (df['ADX'] > 25)  # Strong trend
        )

        # Sell Signal conditions
        sell_condition = (
            (df['RSI'] > 70) &  # Overbought condition
            # EMA crossover (short-term below long-term)
            (df['EMAs'] < df['EMAl']) &
            (df['MACD'] < df['MACDs']) &  # MACD crossover
            (df['SUPERT'] < df['Close']) &  # Supertrend bearish
            (df['VS'] == 1) &  # Volume spike
            (df["Close"] <= df["RES"]) &  # Close at or below Resistance
            (df['ADX'] > 25)  # Strong trend
        )

        # Apply conditions to generate signals
        df.loc[buy_condition, 'Signal'] = 1  # Buy signal
        df.loc[sell_condition, 'Signal'] = -1  # Sell signal

        return df

    def capture_technical_indicators(self, df, trend=True, momentum=True, volatility=True, price_action=True, volume=True):
        trend_df = []
        momentum_df = []
        price_action_df = []
        volatility_df = []
        volume_df = []

        if trend:
            trend_df = self.capture_trend(df)
        if momentum:
            momentum_df = self.capture_momentum(df)
        if price_action:
            price_action_df = self.capture_price_action(df)
        if volatility:
            volatility_df = self.capture_volatility_params(df)

        if volume:
            volume_df = self.capture_volume_params(df)

        return pd.concat([trend_df, momentum_df, price_action_df, volatility_df, volume_df], axis=1).bfill().ffill()

    # def find_candlestick_pattern(self, data):
    #     # Initialize the DataFrame to hold the patterns with the same index as the input data
    #     cd_patterns = pd.DataFrame(index=data.index)

    #     # Get the list of candlestick pattern functions from TA-Lib
    #     patterns = talib.get_function_groups()['Pattern Recognition']

    #     # Loop through each pattern and calculate its value
    #     for pattern in patterns:
    #         # Get the corresponding TA-Lib function for the pattern
    #         func = getattr(talib, pattern)

    #         # Call the function for the candlestick pattern and assign it as a new column
    #         cd_patterns[pattern] = func(
    #             data['Open'], data['High'], data['Low'], data['Close'])
            
        

    #     return cd_patterns
    
    def find_candlestick_pattern(self,data):
        """
        Detect candlestick patterns using pandas_ta library instead of TA-Lib.

        Args:
            data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', 'Close' columns.

        Returns:
            pd.DataFrame: DataFrame with candlestick pattern detection results.
        """
        # Initialize the DataFrame to hold patterns
        cd_patterns = pd.DataFrame(index=data.index)

        # List of available candlestick patterns in pandas-ta
        # pattern_funcs = [
        #     'cdl_doji', 'cdl_engulfing', 'cdl_hammer', 'cdl_invertedhammer',
        #     'cdl_morningstar', 'cdl_morningdojistar', 'cdl_eveningstar',
        #     'cdl_eveningdojistar', 'cdl_shootingstar', 'cdl_hangingman',
        #     'cdl_piercing', 'cdl_darkcloudcover', 'cdl_threewhitesoldiers',
        #     'cdl_threeblackcrows', 'cdl_marubozu', 'cdl_spinningtop'
        # ]

        # Loop through each pattern and calculate it
        # for pattern in pattern_funcs:
        #     # pandas-ta pattern functions need open, high, low, close explicitly
        #     cd_patterns[pattern] = getattr(ta, pattern)(
        #         open_=data['Open'], high=data['High'], low=data['Low'], close=data['Close']
        #     )

        return ta.cdl_pattern(open_=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])
        
    def add_technical_params(self, data_):
        data = data_.copy()
        df_cd = self.find_candlestick_pattern(data_)
        # Get the bullish & bear technical indicator
        data['Bullish'] = df_cd[df_cd > 0].sum(axis=1)
        data['Bearish'] = df_cd[df_cd < 0].sum(axis=1)
        df_technical_params = self.capture_technical_indicators(
            data_).bfill().ffill()
        return pd.concat([data, df_technical_params], axis=1)

    def get_signal_from_candlestick_pattern(self, data_):
        df_cd = self.find_candlestick_pattern(data_)
        data = pd.DataFrame(index=data_.index)
        # Get the bullish & bear technical indicator
        data['Bullish'] = df_cd[df_cd > 0].sum(axis=1)
        data['Bearish'] = df_cd[df_cd < 0].sum(axis=1)
        return data


if __name__ == "__main__":
    from data.data_fetcher import load_stock_data
    data = load_stock_data("RELIANCE.NS")

    tool = Tool()
    df = tool.add_technical_params(
        data, trend=True, momentum=True, volatility=True, price_action=True,)
