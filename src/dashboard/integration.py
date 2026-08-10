from threading import Thread
import asyncio
from typing import Optional, List

try:
    from app import TradingSystem
except Exception:
    TradingSystem = None

_system: Optional[TradingSystem] = None
_thread: Optional[Thread] = None


def create_system(**kwargs) -> Optional[TradingSystem]:
    global _system
    if TradingSystem is None:
        return None
    if _system is None:
        _system = TradingSystem(**kwargs)
    return _system


def _run_system(system: TradingSystem, price_data: Optional[List[float]] = None, delay: float = 0.01):
    try:
        asyncio.run(system.run(price_data=price_data, delay_per_tick=delay))
    except Exception:
        # best-effort runner for dashboard; exceptions are swallowed here
        pass


def start_system_in_thread(price_data: Optional[List[float]] = None, delay: float = 0.01, **kwargs) -> bool:
    """Start TradingSystem in a background thread. Returns True if started."""
    global _system, _thread
    if TradingSystem is None:
        return False
    if _system is None:
        _system = create_system(**kwargs)

    if _thread is not None and _thread.is_alive():
        return True

    _thread = Thread(target=_run_system, args=(_system, price_data, delay), daemon=True)
    _thread.start()
    return True


def get_stats() -> dict:
    if _system is None:
        return {}
    try:
        return _system.get_stats()
    except Exception:
        return {}


def process_tick(price: float) -> None:
    if _system is None:
        return
    try:
        _system.process_tick(price)
    except Exception:
        pass


def stop_system() -> None:
    # No graceful stop implemented in TradingSystem; placeholder
    global _thread, _system
    _thread = None
