"""Replay loader utilities to stream TickEvent objects from stored data.

Provides:
- load_prices_from_csv(path) -> List[float]
- async replay_price_series(price_series, delay_per_tick=0.01) -> AsyncGenerator[TickEvent]
- replay_from_csv(path, delay_per_tick=0.01) -> AsyncGenerator[TickEvent]

The functions are intentionally lightweight and dependency-free (pandas optional).
"""
from typing import List, AsyncGenerator, Optional
import asyncio
import csv
import os
from datetime import datetime

try:
    from smart.trading_system.comm.events import TickEvent
except Exception:
    # Fallback local import for direct module execution
    from ..utils.events import TickEvent  # type: ignore


def load_prices_from_csv(path: str, price_column: str = 'Close') -> List[float]:
    """Load a list of prices from a CSV file.

    Args:
        path: Path to CSV file.
        price_column: Column name to use for prices (case-sensitive).

    Returns:
        List of float prices (empty list on error).
    """
    if not os.path.exists(path):
        return []

    prices: List[float] = []
    try:
        # Try pandas for convenience if available
        import pandas as pd

        df = pd.read_csv(path)
        if price_column in df.columns:
            prices = df[price_column].astype(float).tolist()
        else:
            # fallback to first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                prices = df[numeric_cols[0]].astype(float).tolist()
    except Exception:
        # Fallback to csv module
        try:
            with open(path, 'r', newline='') as fh:
                reader = csv.DictReader(fh)
                # try price_column, else first numeric
                for row in reader:
                    if price_column in row and row[price_column] not in (None, ''):
                        try:
                            prices.append(float(row[price_column]))
                        except Exception:
                            continue
                    else:
                        # pick first numeric-looking column
                        for v in row.values():
                            try:
                                prices.append(float(v))
                                break
                            except Exception:
                                continue
        except Exception:
            return []

    return prices


async def replay_price_series(
    price_series: List[float],
    delay_per_tick: float = 0.01,
    volume_series: Optional[List[float]] = None,
) -> AsyncGenerator[TickEvent, None]:
    """Async generator that yields TickEvent from price_series.

    Args:
        price_series: List of prices to stream.
        delay_per_tick: Seconds to await between yields.
        volume_series: Optional list of volumes matching price_series.
    """
    for i, price in enumerate(price_series):
        ts = datetime.now().timestamp()
        vol = None
        if volume_series is not None and i < len(volume_series):
            vol = float(volume_series[i])
        yield TickEvent(price=float(price), timestamp=ts, volume=vol)
        await asyncio.sleep(delay_per_tick)


async def replay_from_csv(path: str, delay_per_tick: float = 0.01) -> AsyncGenerator[TickEvent, None]:
    """Convenience: read CSV and stream TickEvent.

    Args:
        path: CSV path
        delay_per_tick: delay between ticks
    """
    prices = load_prices_from_csv(path)
    async for tick in replay_price_series(prices, delay_per_tick=delay_per_tick):
        yield tick
