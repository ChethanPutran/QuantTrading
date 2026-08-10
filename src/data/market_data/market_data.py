from datetime import datetime, timedelta
import os
import time

import pandas as pd

from ..base import BaseMarketDataProvider
from .transforms import clean_ohlcv, safe_concat


class YFinanceMarketDataProvider(BaseMarketDataProvider):
    def fetch(
        self,
        symbol: str,
        interval: str = "1m",
        period: str = "1d",
        start: str | None = None,
        end: str | None = None,
        drop_ticker: bool = True,
        timezone: str = "Asia/Kolkata",
        **kwargs,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required for Yahoo market data") from exc

        if start and end:
            data = yf.download(
                tickers=symbol,
                interval=interval,
                start=start,
                end=end,
                auto_adjust=True,
                **kwargs,
            )
        else:
            data = yf.download(
                tickers=symbol,
                interval=interval,
                period=period,
                auto_adjust=True,
                **kwargs,
            )

        if data.empty:
            return pd.DataFrame()

        return clean_ohlcv(data, drop_ticker=drop_ticker, timezone=timezone)


def load_stock_data(
    ticker: str,
    interval: str = "1m",
    period: str = "1d",
    start: str | None = None,
    end: str | None = None,
    drop_ticker: bool = True,
    save: bool = False,
    file_path: str = "yfinance_data.pkl",
    refresh: bool = False,
) -> tuple[bool, pd.DataFrame]:
    if not refresh and os.path.exists(file_path):
        return True, pd.read_pickle(file_path)

    provider = YFinanceMarketDataProvider()
    data = provider.fetch(
        ticker,
        interval=interval,
        period=period,
        start=start,
        end=end,
        drop_ticker=drop_ticker,
    )

    if data.empty:
        return False, data

    if save:
        data.to_pickle(file_path)

    return True, data


def get_last_n_min_data(
    ticker: str,
    minutes: int = 60,
    interval: str = "1m",
) -> pd.DataFrame:
    try:
        import pytz
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance and pytz are required for intraday data") from exc

    data = yf.Ticker(ticker).history(period="1d", interval=interval)
    ist = pytz.timezone("Asia/Kolkata")
    cutoff = datetime.now(ist) - timedelta(minutes=minutes)
    return data[data.index >= cutoff]


def get_todays_nifty_data() -> tuple[bool, pd.DataFrame]:
    return load_stock_data("^NSEI", refresh=True)


class DataGenerator:
    def __init__(
        self,
        tickers: list[str] | None = None,
        total_days: int = 28,
        interval: str = "1m",
        max_days_per_request: int = 7,
        output_dir: str = "data",
    ) -> None:
        self.total_days = total_days
        self.tickers = tickers or []
        self.interval = interval
        self.max_days_per_request = max_days_per_request
        self.output_dir = output_dir

    def generate(self, wait_time: int = 3) -> list[str]:
        file_names = []

        for ticker in self.tickers:
            data_list = []
            end_date = datetime.now()
            remaining_days = self.total_days

            while remaining_days > 0:
                days_to_fetch = min(remaining_days, self.max_days_per_request)
                start_date = end_date - timedelta(days=days_to_fetch)
                status, data = load_stock_data(
                    ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval=self.interval,
                    refresh=True,
                )

                if status and not data.empty:
                    data_list.insert(0, data)

                remaining_days -= days_to_fetch
                end_date = start_date - timedelta(days=1)
                time.sleep(wait_time)

            combined_df = safe_concat(data_list)
            if combined_df.empty:
                continue

            combined_df = combined_df[~combined_df.index.duplicated()]
            output_path = os.path.join(self.output_dir, f"{ticker}_data.csv")
            combined_df.to_csv(output_path)
            file_names.append(output_path)

        return file_names
