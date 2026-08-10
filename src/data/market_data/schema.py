from dataclasses import dataclass

# Tick Event
@dataclass
class TickEvent:
    timestamp: int

    symbol: str

    bid: float
    ask: float
    last: float

    volume: float

    spread: float

    exchange: str

# OHLCV Candle
@dataclass
class CandleEvent:
    timestamp: int

    symbol: str
    timeframe: str

    open: float
    high: float
    low: float
    close: float

    volume: float
    trades: int


# Order Book Snapshot
@dataclass
class OrderBookEvent:
    timestamp: int

    symbol: str

    best_bid: float
    best_ask: float

    bid_volume: float
    ask_volume: float

    imbalance: float

    depth_levels: list