import pandas as pd


def decision_signal(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    signals = []

    for i in range(len(output)):
        buy_signal = False
        sell_signal = False

        if output["Close"].iloc[i] > output["EMAs"].iloc[i] and output["RSI"].iloc[i] < 30:
            buy_signal = True
        if output["MACD"].iloc[i] > output["MACDs"].iloc[i] and output["Close"].iloc[i] < output["BBL"].iloc[i]:
            buy_signal = True

        if output["Close"].iloc[i] < output["EMAs"].iloc[i] and output["RSI"].iloc[i] > 70:
            sell_signal = True
        if output["MACD"].iloc[i] < output["MACDs"].iloc[i] and output["Close"].iloc[i] > output["BBU"].iloc[i]:
            sell_signal = True

        if buy_signal:
            signals.append("Buy")
        elif sell_signal:
            signals.append("Sell")
        else:
            signals.append("Hold")

    output["Signal"] = signals
    return output


def generate_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["Signal"] = 0

    buy_condition = (
        (output["RSI"] < 30)
        & (output["EMAs"] > output["EMAl"])
        & (output["MACD"] > output["MACDs"])
        & (output["SUPERT"] > output["Close"])
        & (output["VS"] == 1)
        & (output["Close"] >= output["SUP"])
        & (output["ADX"] > 25)
    )

    sell_condition = (
        (output["RSI"] > 70)
        & (output["EMAs"] < output["EMAl"])
        & (output["MACD"] < output["MACDs"])
        & (output["SUPERT"] < output["Close"])
        & (output["VS"] == 1)
        & (output["Close"] <= output["RES"])
        & (output["ADX"] > 25)
    )

    output.loc[buy_condition, "Signal"] = 1
    output.loc[sell_condition, "Signal"] = -1
    return output
