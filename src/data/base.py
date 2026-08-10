from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
@dataclass
class EventEnvelope:
    event_id: str

    timestamp: int

    event_type: str

    payload: dict

    source: str

    version: int


class BaseDataCollector(ABC):
    """Base class for data collectors."""

    @abstractmethod
    async def run(self):
        pass
    
class BaseMarketDataProvider(ABC):
    """Fetches OHLCV or quote data for a symbol."""

    @abstractmethod
    def fetch(self, symbol: str, **kwargs) -> pd.DataFrame | Mapping[str, Any]:
        pass


class BaseNewsProvider(ABC):
    """Fetches news items for a symbol, company, or market query."""

    @abstractmethod
    def fetch(self, query: str, **kwargs) -> pd.DataFrame:
        pass


class BaseSentimentAnalyzer(ABC):
    """Scores text or news rows."""

    @abstractmethod
    def score(self, text: str) -> Mapping[str, float]:
        pass


class BaseTickerResolver(ABC):
    """Resolves company names to exchange tickers."""

    @abstractmethod
    def resolve(self, company_name: str, **kwargs) -> str | None:
        pass
