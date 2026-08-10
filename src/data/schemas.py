"""Shared data schemas used by collectors and replay engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
import numpy as np

@dataclass
class BaseEvent:
    timestamp: int          # epoch ms
    source: str             # provider/source
    symbol: str             # asset or topic
    data: Dict[str, Any]    # payload


@dataclass(frozen=True)
class DataRequest:
    symbol: str
    interval: str = "1m"
    period: str = "1d"
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class Quote:
    symbol: str
    last_price: float
    previous_close: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None


@dataclass(frozen=True)
class NewsItem:
    title: str
    summary: str = ""
    link: str = ""
    published_at: datetime | None = None
    source: str | None = None
    company: str | None = None


@dataclass(frozen=True)
class SentimentScore:
    negative: float = 0.0
    neutral: float = 0.0
    positive: float = 0.0
    compound: float = 0.0

@dataclass(slots=True)
class TickSnapshot:
    timestamp: float
    symbol: str
    price: float
    volume: float | None = None
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    imbalance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)



@dataclass(slots=True)
class OptionChainSnapshot:
    timestamp: float
    symbol: str
    expiry: str | None
    underlying_price: float
    implied_volatility: float | None = None
    open_interest: float | None = None
    gamma_exposure: float | None = None
    call_put_imbalance: float | None = None
    iv_skew: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StateSnapshot:
    timestamp: float
    symbol: str
    timeframe: str
    features: np.ndarray
    filtered_price: float
    volatility: float
    regime_probs: np.ndarray
    hidden_state: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
