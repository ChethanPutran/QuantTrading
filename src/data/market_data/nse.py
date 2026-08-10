import pandas as pd


NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}


def _nse_session():
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required for NSE data") from exc

    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    session.get("https://www.nseindia.com", timeout=10)
    return session


def get_nse_stock_data(symbol: str) -> dict:
    symbol = symbol.upper()
    session = _nse_session()
    response = session.get(
        f"https://www.nseindia.com/api/quote-equity?symbol={symbol}",
        timeout=10,
        headers={
            **NSE_HEADERS,
            "Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}",
        },
    )
    response.raise_for_status()
    data = response.json()
    price_info = data["priceInfo"]
    return {
        "symbol": symbol,
        "lastPrice": price_info["lastPrice"],
        "previousClose": price_info.get("previousClose"),
        "open": price_info.get("open"),
        "dayHigh": price_info.get("intraDayHighLow", {}).get("max"),
        "dayLow": price_info.get("intraDayHighLow", {}).get("min"),
        "volume": price_info.get("totalTradedVolume"),
    }


def get_most_traded_nse_stocks() -> pd.DataFrame:
    session = _nse_session()
    response = session.get(
        "https://www.nseindia.com/api/live-analysis-most-active-securities?index=volume",
        timeout=10,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = df[["symbol", "lastPrice", "volume", "value", "pChange"]]
    df.columns = ["Symbol", "Last Price", "Volume", "Turnover", "% Change"]
    return df
