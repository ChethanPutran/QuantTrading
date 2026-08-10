"""Backtesting helper that uses the TradingSystem integration.

This module provides a simple entry to run a price-replay backtest
using the existing `TradingSystem` via `dashboard.integration`.
"""
from typing import List, Optional
from .integration import create_system, start_system_in_thread


def run_replay_backtest(price_series: List[float], delay: float = 0.0, **system_kwargs) -> bool:
    """Start a TradingSystem in replay mode with provided `price_series`.

    Returns True if the system was started.
    """
    system = create_system(**system_kwargs)
    if system is None:
        return False
    return start_system_in_thread(price_data=price_series, delay=delay)


if __name__ == '__main__':
    # Example quick test (random walk)
    import numpy as np
    np.random.seed(0)
    prices = [100.0]
    for _ in range(200):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))
    run_replay_backtest(prices, delay=0.001)