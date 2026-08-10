import pandas as pd


def _require_pandas_ta():
    try:
        import pandas_ta as ta
    except ImportError as exc:
        raise ImportError(
            "pandas_ta is required for candlestick pattern calculations"
        ) from exc
    return ta


def find_candlestick_pattern(data: pd.DataFrame) -> pd.DataFrame:
    ta = _require_pandas_ta()
    return ta.cdl_pattern(
        open_=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
    )


def get_signal_from_candlestick_pattern(data: pd.DataFrame) -> pd.DataFrame:
    df_cd = find_candlestick_pattern(data)
    signals = pd.DataFrame(index=data.index)
    signals["Bullish"] = df_cd[df_cd > 0].sum(axis=1)
    signals["Bearish"] = df_cd[df_cd < 0].sum(axis=1)
    return signals


def add_candlestick_features(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    pattern_signals = get_signal_from_candlestick_pattern(data)
    output["Bullish"] = pattern_signals["Bullish"]
    output["Bearish"] = pattern_signals["Bearish"]
    return output


def find_candlestick_pattern_talib(data: pd.DataFrame) -> pd.DataFrame:
    try:
        import talib
    except ImportError as exc:
        raise ImportError("TA-Lib is required for this pattern detector") from exc

    cd_patterns = pd.DataFrame(index=data.index)
    patterns = talib.get_function_groups()["Pattern Recognition"]

    for pattern in patterns:
        func = getattr(talib, pattern)
        cd_patterns[pattern] = func(
            data["Open"],
            data["High"],
            data["Low"],
            data["Close"],
        )

    return cd_patterns


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["RSI"] = compute_rsi(output["Close"])
    output["MACD"], output["MACD_Signal"] = compute_macd(output["Close"])
    output["SMA_20"] = output["Close"].rolling(window=20).mean()
    output["SMA_50"] = output["Close"].rolling(window=50).mean()
    output["EMA_12"] = output["Close"].ewm(span=12, adjust=False).mean()
    output["EMA_26"] = output["Close"].ewm(span=26, adjust=False).mean()
    return output


def add_macro_features(df: pd.DataFrame, macro_data: dict | None = None) -> pd.DataFrame:
    output = df.copy()
    macro_data = macro_data or {"CPI": 0.0, "Interest_Rate": 0.0}
    for name, value in macro_data.items():
        output[name] = value
    return output


def add_sentiment_features(
    df: pd.DataFrame,
    sentiment_data: dict | None = None,
) -> pd.DataFrame:
    output = df.copy()
    sentiment_data = sentiment_data or {}
    for name, value in sentiment_data.items():
        output[name] = value
    return output


def prepare_features(
    df: pd.DataFrame,
    sentiment_data: dict | None = None,
    macro_data: dict | None = None,
) -> pd.DataFrame:
    output = calculate_technical_indicators(df)
    output = add_macro_features(output, macro_data)
    output = add_sentiment_features(output, sentiment_data)
    return output.fillna(0)


import pandas as pd

from .base import BaseFeatureTransformer


def _require_pandas_ta():
    try:
        import pandas_ta as ta
    except ImportError as exc:
        raise ImportError(
            "pandas_ta is required for technical feature calculations"
        ) from exc
    return ta


def weighted_sum(df_: pd.DataFrame, weights: dict[str, float], normalize: bool = True):
    df = df_.bfill().ffill()
    ws = sum(df[column] * weight for column, weight in weights.items())
    if normalize:
        span = ws.max() - ws.min()
        if span == 0:
            return ws * 0
        return (ws - ws.min()) / span
    return ws


def capture_trend(
    data: pd.DataFrame,
    ema_short_length: int = 5,
    ema_long_length: int = 20,
    adx_length: int = 14,
    supt_length: int = 10,
    supt_multi: float = 3.0,
    single: bool = False,
    normalize: bool = True,
) -> pd.DataFrame | pd.Series:
    _require_pandas_ta()
    weights = {
        "EMAs": 0.1,
        "EMAl": 0.15,
        "ADX": 0.2,
        "DMP": 0.1,
        "DMN": 0.1,
        "SUPERT": 0.1,
        "SUPERTd": 0.05,
        "SUPERTl": 0.05,
        "SUPERTs": 0.05,
    }
    trend_params = pd.DataFrame(index=data.index)
    trend_params["EMAs"] = data.ta.ema(length=ema_short_length)
    trend_params["EMAl"] = data.ta.ema(length=ema_long_length)
    trend_params[["ADX", "DMP", "DMN"]] = data.ta.adx(length=adx_length)
    trend_params[["SUPERT", "SUPERTd", "SUPERTl", "SUPERTs"]] = data.ta.supertrend(
        length=supt_length,
        multiplier=supt_multi,
    )
    if single:
        return weighted_sum(trend_params, weights, normalize)
    return trend_params


def capture_momentum(
    data: pd.DataFrame,
    rsi_length: int = 14,
    stoc_osc_length: int = 14,
    cci_length: int = 20,
    mcda_length: int = 12,
    single: bool = False,
    normalize: bool = True,
    append: bool = False,
) -> pd.DataFrame | pd.Series:
    _require_pandas_ta()
    weights = {
        "RSI": 0.15,
        "STOCHk": 0.1,
        "STOCHd": 0.1,
        "CCI": 0.1,
        "MACD": 0.2,
        "MACDh": 0.1,
        "MACDs": 0.15,
    }
    momentum_params = pd.DataFrame(index=data.index)
    momentum_params["RSI"] = data.ta.rsi(length=rsi_length)
    momentum_params[["STOCHk", "STOCHd"]] = data.ta.stoch(
        length=stoc_osc_length
    )
    momentum_params["CCI"] = data.ta.cci(length=cci_length)
    momentum_params[["MACD", "MACDh", "MACDs"]] = data.ta.macd(length=mcda_length)
    if single:
        return weighted_sum(momentum_params, weights, normalize)
    if append:
        return pd.concat([data, momentum_params], axis=1)
    return momentum_params


def capture_volatility_params(
    data: pd.DataFrame,
    bbl_length: int = 20,
    atr_length: int = 14,
    single: bool = False,
    normalize: bool = True,
) -> pd.DataFrame | pd.Series:
    _require_pandas_ta()
    weights = {
        "BBL": 0.1,
        "BBM": 0.1,
        "BBU": 0.1,
        "BBB": 0.1,
        "BBP": 0.1,
        "ATR": 0.5,
    }
    volatility_params = pd.DataFrame(index=data.index)
    volatility_params[["BBL", "BBM", "BBU", "BBB", "BBP"]] = data.ta.bbands(
        length=bbl_length
    )
    volatility_params["ATR"] = data.ta.atr(length=atr_length)
    if single:
        return weighted_sum(volatility_params, weights, normalize)
    return volatility_params


def capture_volume_params(
    data: pd.DataFrame,
    vma_length: int = 20,
    single: bool = False,
    normalize: bool = True,
) -> pd.DataFrame | pd.Series:
    _require_pandas_ta()
    weights = {
        "VMA": 0.25,
        "VWAP": 0.25,
        "OBV": 0.25,
        "VS": 0.25,
    }
    volume_params = pd.DataFrame(index=data.index)
    volume_params["VMA"] = data["Volume"].rolling(window=vma_length).mean()
    volume_params["VWAP"] = data.ta.vwap()
    volume_params["OBV"] = data.ta.obv()
    volume_params["VS"] = (data["Volume"] > 2 * volume_params["VMA"]).astype(int)
    if single:
        return weighted_sum(volume_params, weights, normalize)
    return volume_params


def capture_price_action(
    data: pd.DataFrame,
    support_length: int = 12,
    resistance_length: int = 15,
    single: bool = False,
    normalize: bool = True,
) -> pd.DataFrame | pd.Series:
    weights = {
        "SUP": 0.5,
        "RES": 0.5,
    }
    price_action_params = pd.DataFrame(index=data.index)
    price_action_params["SUP"] = data["Low"].rolling(window=support_length).min()
    price_action_params["RES"] = data["High"].rolling(window=resistance_length).max()
    if single:
        return weighted_sum(price_action_params, weights, normalize)
    return price_action_params


def capture_technical_indicators(
    df: pd.DataFrame,
    trend: bool = True,
    momentum: bool = True,
    volatility: bool = True,
    price_action: bool = True,
    volume: bool = True,
) -> pd.DataFrame:
    frames = []

    if trend:
        frames.append(capture_trend(df))
    if momentum:
        frames.append(capture_momentum(df))
    if price_action:
        frames.append(capture_price_action(df))
    if volatility:
        frames.append(capture_volatility_params(df))
    if volume:
        frames.append(capture_volume_params(df))

    if not frames:
        return pd.DataFrame(index=df.index)
    return pd.concat(frames, axis=1).bfill().ffill()


def latest_technical_features(df: pd.DataFrame) -> dict[str, float]:
    enriched = capture_technical_indicators(df)
    latest = enriched.iloc[-1]
    return {
        str(name): float(value)
        for name, value in latest.items()
        if pd.notna(value)
    }


class TechnicalFeatureTransformer(BaseFeatureTransformer):
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return capture_technical_indicators(data)
