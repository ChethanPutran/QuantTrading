from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


def to_float(value: Any) -> float | None:
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (bool, int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None

    return None


def is_sequence(value: Any) -> bool:
    return isinstance(value, (Sequence, np.ndarray)) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def flatten_numeric(data: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}

    for key, value in data.items():
        flatten_value(str(key).lower(), value, values)

    return values


def flatten_value(key: str, value: Any, values: dict[str, float]) -> None:
    number = to_float(value)
    if number is not None:
        values[key] = number
        return

    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            flatten_value(f"{key}.{str(child_key).lower()}", child_value, values)
        return

    if is_sequence(value):
        for index, item in enumerate(value):
            flatten_value(f"{key}.{index}", item, values)


def safe_ratio(
    numerator: float | None,
    denominator: float | None,
    fill_value: float = 0.0,
) -> float:
    if numerator is None or denominator in (None, 0.0):
        return fill_value
    return (numerator / denominator) - 1.0


def range_position(
    price: float | None,
    low: float | None,
    high: float | None,
    fill_value: float = 0.0,
) -> float:
    if price is None or low is None or high is None or high == low:
        return fill_value
    return (price - low) / (high - low)


def finite_or_fill(value: float | None, fill_value: float = 0.0) -> float:
    if value is None or not math.isfinite(value):
        return fill_value
    return value


def clean_feature_dict(features: Mapping[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}

    for name, value in features.items():
        number = to_float(value)
        if number is not None and math.isfinite(number):
            cleaned[name] = number

    return cleaned

import numpy as np
import pandas as pd


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if isinstance(output.columns, pd.MultiIndex):
        output.columns = output.columns.droplevel(-1)
    return output


def ensure_timezone(
    df: pd.DataFrame,
    timezone: str = "Asia/Kolkata",
) -> pd.DataFrame:
    output = df.copy()
    if not isinstance(output.index, pd.DatetimeIndex):
        return output

    if output.index.tz is None:
        output.index = output.index.tz_localize("UTC").tz_convert(timezone)
    else:
        output.index = output.index.tz_convert(timezone)

    return output


def clean_ohlcv(
    df: pd.DataFrame,
    drop_ticker: bool = True,
    timezone: str = "Asia/Kolkata",
) -> pd.DataFrame:
    output = df.copy()
    if drop_ticker:
        output = flatten_yfinance_columns(output)
    output = ensure_timezone(output, timezone)
    return output.sort_index()


def create_sequences(
    data: np.ndarray,
    seq_length: int = 50,
    output_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = []
    y_values = []

    for i in range(len(data) - seq_length):
        x_values.append(data[i : i + seq_length])
        if output_idx is None:
            y_values.append(data[i + seq_length])
        else:
            y_values.append(data[i + seq_length, output_idx])

    return np.asarray(x_values), np.asarray(y_values)


def safe_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    return pd.concat(usable).sort_index()
