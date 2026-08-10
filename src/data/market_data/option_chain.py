from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, cast

import pandas as pd


OPTION_CHAIN_COLUMNS = (
    "contractSymbol",
    "lastTradeDate",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "change",
    "percentChange",
    "volume",
    "openInterest",
    "impliedVolatility",
    "inTheMoney",
    "contractSize",
    "currency",
)


def _load_yfinance_ticker(symbol: str):
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("yfinance is required for option chain data") from exc

    return yf.Ticker(symbol)


def _underlying_price(ticker) -> float | None:
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info is not None:
        for field in ("lastPrice", "last_price", "regularMarketPrice"):
            value = fast_info.get(field) if hasattr(fast_info, "get") else None
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

    info = getattr(ticker, "info", {}) or {}
    for field in ("regularMarketPrice", "currentPrice", "lastPrice"):
        value = info.get(field)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    try:
        history = ticker.history(period="5d", interval="1d")
        if not history.empty:
            return float(history["Close"].dropna().iloc[-1])
    except Exception:
        return None

    return None


def _normalize_chain_frame(
    frame: pd.DataFrame,
    side: str,
    underlying_price: float | None,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    output = frame.copy()
    if "strike" in output.columns:
        output = output.sort_values("strike")
        if underlying_price is not None:
            output = output.assign(
                _distance=(output["strike"] - underlying_price).abs()
            ).sort_values(["_distance", "strike"])

    selected_columns = [name for name in OPTION_CHAIN_COLUMNS if name in output.columns]
    records = output[selected_columns].copy()
    if "lastTradeDate" in records.columns:
        records["lastTradeDate"] = records["lastTradeDate"].astype(str)

    normalized: list[dict[str, Any]] = [
        cast(dict[str, Any], dict(record))
        for record in records.to_dict(orient="records")
    ]
    for item in normalized:
        item["side"] = side

    return normalized


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _synthetic_option_chain_frames(
    symbol: str,
    underlying_price: float,
    selected_expiry: str,
    strike_count: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a lightweight synthetic option chain around the underlying price.

    This is used when Yahoo does not expose options for an index symbol.
    The goal is not market realism; it is to keep replay simulations flowing.
    """
    center = max(1.0, float(underlying_price))
    strike_step = max(1.0, round(center * 0.01, 2))
    half_width = strike_count // 2
    strike_prices = [round(center + (i - half_width) * strike_step, 2) for i in range(strike_count)]

    def build_row(strike: float, side: str) -> dict[str, Any]:
        distance = abs(strike - center)
        intrinsic = max(0.0, center - strike) if side == "call" else max(0.0, strike - center)
        time_value = max(0.25, center * 0.015 * math.exp(-distance / max(center * 0.03, 1.0)))
        last_price = round(intrinsic + time_value, 2)
        implied_volatility = round(0.12 + min(0.9, distance / max(center, 1.0) * 1.5), 4)
        open_interest = int(max(25, 2000 - distance * 12))
        volume = int(max(5, 250 - distance * 2))

        return {
            "contractSymbol": f"{symbol}_{selected_expiry}_{side.upper()}_{strike:.0f}",
            "lastTradeDate": pd.Timestamp.now(tz="UTC"),
            "strike": strike,
            "lastPrice": last_price,
            "bid": round(max(0.01, last_price - 0.05), 2),
            "ask": round(last_price + 0.05, 2),
            "change": 0.0,
            "percentChange": 0.0,
            "volume": volume,
            "openInterest": open_interest,
            "impliedVolatility": implied_volatility,
            "inTheMoney": intrinsic > 0,
            "contractSize": "REGULAR",
            "currency": "INR",
        }

    calls = pd.DataFrame([build_row(strike, "call") for strike in strike_prices])
    puts = pd.DataFrame([build_row(strike, "put") for strike in strike_prices])
    return calls, puts


def get_index_option_chain_data(
    symbol: str,
    expiry: str | None = None,
) -> Dict[str, Any]:
    """Fetch a normalized option-chain snapshot for an index symbol."""

    ticker = _load_yfinance_ticker(symbol)
    expiries = list(getattr(ticker, "options", []) or [])
    if not expiries:
        raise ValueError(f"No option expiries available for {symbol}")

    selected_expiry = expiry or expiries[0]
    if selected_expiry not in expiries:
        raise ValueError(
            f"Expiry {selected_expiry!r} is not available for {symbol}; "
            f"available expiries: {expiries}"
        )

    chain = ticker.option_chain(selected_expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    underlying_price = _underlying_price(ticker)

    summary: dict[str, Any] = {
        "symbol": symbol,
        "expiry": selected_expiry,
        "underlying_price": underlying_price,
        "available_expiries": len(expiries),
        "calls_count": int(len(calls)),
        "puts_count": int(len(puts)),
        "calls_open_interest": _sum_column(calls, "openInterest"),
        "puts_open_interest": _sum_column(puts, "openInterest"),
        "calls_volume": _sum_column(calls, "volume"),
        "puts_volume": _sum_column(puts, "volume"),
    }

    combined_frames = []
    if not calls.empty and "strike" in calls.columns:
        combined_frames.append(calls[["strike"]])
    if not puts.empty and "strike" in puts.columns:
        combined_frames.append(puts[["strike"]])

    if underlying_price is not None and combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True).dropna(subset=["strike"])
        if not combined.empty:
            distances = (combined["strike"] - underlying_price).abs()
            nearest_index = distances.idxmin()
            nearest_strike = float(combined.loc[nearest_index, "strike"])
            summary["atm_strike"] = nearest_strike
            summary["atm_distance"] = abs(nearest_strike - underlying_price)

    return {
        "symbol": symbol,
        "expiry": selected_expiry,
        "available_expiries": expiries,
        "underlying_price": underlying_price,
        "summary": summary,
        "calls": _normalize_chain_frame(calls, "call", underlying_price),
        "puts": _normalize_chain_frame(puts, "put", underlying_price),
    }


def _build_snapshot_for_underlying(
    symbol: str,
    selected_expiry: str,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    underlying_price: float | None,
    expiries: list[str],
) -> dict[str, Any]:
    """Build a snapshot dict for a given underlying price using the provided chains.

    This uses the current option strikes/vols from `calls`/`puts` but substitutes
    the provided `underlying_price` so snapshots can be replayed for simulation.
    """
    summary: dict[str, Any] = {
        "symbol": symbol,
        "expiry": selected_expiry,
        "underlying_price": underlying_price,
        "available_expiries": len(expiries),
        "calls_count": int(len(calls)),
        "puts_count": int(len(puts)),
        "calls_open_interest": _sum_column(calls, "openInterest"),
        "puts_open_interest": _sum_column(puts, "openInterest"),
        "calls_volume": _sum_column(calls, "volume"),
        "puts_volume": _sum_column(puts, "volume"),
    }

    # compute ATM strike relative to provided underlying_price
    combined_frames = []
    if not calls.empty and "strike" in calls.columns:
        combined_frames.append(calls[["strike"]])
    if not puts.empty and "strike" in puts.columns:
        combined_frames.append(puts[["strike"]])

    if underlying_price is not None and combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True).dropna(subset=["strike"])
        if not combined.empty:
            distances = (combined["strike"] - underlying_price).abs()
            nearest_index = distances.idxmin()
            nearest_strike = float(combined.loc[nearest_index, "strike"])
            summary["atm_strike"] = nearest_strike
            summary["atm_distance"] = abs(nearest_strike - underlying_price)

    return {
        "symbol": symbol,
        "expiry": selected_expiry,
        "available_expiries": expiries,
        "underlying_price": underlying_price,
        "summary": summary,
        "calls": _normalize_chain_frame(calls, "call", underlying_price),
        "puts": _normalize_chain_frame(puts, "put", underlying_price),
    }


def get_index_option_chain_history(
    symbol: str,
    expiry: str | None = None,
    days: int = 30,
    interval: str = "1d",
) -> list[dict[str, Any]]:
    """Return a list of option-chain snapshots for the past `days` days.

    Note: yfinance does not provide historical option-chain snapshots. This
    function reuses the current option-chain (for the chosen expiry) and
    replays it across historical underlying prices fetched from yfinance.
    Each returned snapshot contains a `date` key with the historical timestamp.
    """
    ticker = _load_yfinance_ticker(symbol)
    expiries = list(getattr(ticker, "options", []) or [])
    selected_expiry = expiry or (expiries[0] if expiries else "SYNTHETIC")

    synthetic_mode = not expiries
    if expiries:
        if selected_expiry not in expiries:
            raise ValueError(
                f"Expiry {selected_expiry!r} is not available for {symbol}; "
                f"available expiries: {expiries}"
            )

        chain = ticker.option_chain(selected_expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
    else:
        calls = pd.DataFrame()
        puts = pd.DataFrame()

    assert calls.empty or puts.empty, "Empty calls or puts frame expected for synthetic mode"

    history = ticker.history(period=f"{days}d", interval=interval)
    if history is None or history.empty:
        return []

    closes = history["Close"].dropna()
    snapshots: list[dict[str, Any]] = []
    for ts, close in closes.items():
        try:
            underlying = float(close)
        except (TypeError, ValueError):
            continue

        if synthetic_mode:
            calls, puts = _synthetic_option_chain_frames(
                symbol=symbol,
                underlying_price=underlying,
                selected_expiry=selected_expiry,
            )

        snap = _build_snapshot_for_underlying(
            symbol=symbol,
            selected_expiry=selected_expiry,
            calls=calls,
            puts=puts,
            underlying_price=underlying,
            expiries=expiries,
        )
        snap["date"] = ts.isoformat()
        snap["synthetic"] = synthetic_mode
        snapshots.append(snap)

    return snapshots


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(cast(Any, value))
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and print a normalized yfinance option-chain snapshot."
    )
    parser.add_argument(
        "symbol",
        help="Index symbol to inspect, for example ^NSEI, ^NSEBANK, or ^BSESN",
        
    )
    parser.add_argument(
        "--expiry",
        default=None,
        help="Optional expiry date from the available yfinance expiries",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Return historical snapshots for the past N days instead of a single snapshot",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=30,
        help="Number of past days to fetch underlying history for (used with --history)",
    )
    parser.add_argument(
        "--history-interval",
        default="1d",
        help="Interval for historical underlying prices (e.g. '1d', '1h')",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for the printed output",
    )
    args = parser.parse_args()

    try:
        if args.history:
            snapshots = get_index_option_chain_history(
                args.symbol, expiry=args.expiry, days=args.history_days, interval=args.history_interval
            )
            print(json.dumps(snapshots, indent=args.indent, default=_json_default))
        else:
            snapshot = get_index_option_chain_data(args.symbol, expiry=args.expiry)
            print(json.dumps(snapshot, indent=args.indent, default=_json_default))
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())