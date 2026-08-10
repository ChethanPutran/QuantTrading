from collections.abc import Mapping
import os
from typing import Any

import pandas as pd

from .base import BaseFeatureProvider
from .macro import FRED_SERIES


class YFinancePriceProvider(BaseFeatureProvider):
    def __init__(self, period: str = "6mo", interval: str = "1d") -> None:
        self.period = period
        self.interval = interval

    def fetch(self, ticker: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required for price data") from exc

        return yf.download(ticker, period=self.period, interval=self.interval)


class YFinanceFundamentalProvider(BaseFeatureProvider):
    def fetch(self, ticker: str) -> Mapping[str, Any]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required for fundamental data") from exc

        return yf.Ticker(ticker).info


class FredMacroProvider(BaseFeatureProvider):
    def __init__(
        self,
        api_key: str | None = None,
        series_map: Mapping[str, str] | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.series_map = dict(series_map or FRED_SERIES)

    def fetch(self, ticker: str = "") -> Mapping[str, Any]:
        try:
            from fredapi import Fred
        except ImportError as exc:
            raise ImportError("fredapi is required for macro data") from exc

        fred = Fred(api_key=self.api_key)
        return {
            name: fred.get_series(series_id)
            for name, series_id in self.series_map.items()
        }


class OpenBBSentimentProvider(BaseFeatureProvider):
    def fetch(self, ticker: str) -> Mapping[str, Any]:
        try:
            import openbb
        except ImportError as exc:
            raise ImportError("openbb is required for sentiment data") from exc

        return {
            "news_sentiment": openbb.stocks.sia.sentiment(ticker),
            "options_oi": openbb.stocks.options.oi(ticker),
            "options_flow": openbb.stocks.options.unusual(ticker),
        }
