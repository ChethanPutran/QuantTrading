"""Feature engineering pipeline for multi-timeframe trading state vectors.

The pipeline is designed for online/event-driven processing.  Each call to
``update`` consumes one TickEvent and updates all rolling statistics without
using future information.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Tuple

import numpy as np
from pydantic_settings import BaseSettings

from config.settings import FeaturePipelineConfig
from filters.kalman import KalmanFilter1D
from .base import BaseFeaturePipeline
from .transforms import finite_or_fill, flatten_numeric, range_position, safe_ratio
from ..core.events import StateVector, TickEvent





class FeaturePipeline(BaseFeaturePipeline):
    """Build a fixed-size online feature vector from market ticks.

    Feature order is:

    0.  return
    1.  rolling volatility
    2.  momentum
    3.  RSI
    4.  MACD
    5.  EMA fast
    6.  SMA
    7.  ATR-like tick range
    8.  VWAP
    9.  Bollinger-band position
    10. Bollinger-band width
    11. Price z-score
    12. spread
    13. order-book imbalance

    Volume is maintained internally and can be added to the feature vector
    later without changing the online state-management design.
    """

    FEATURE_NAMES = (
        "return",
        "rolling_volatility",
        "momentum",
        "rsi",
        "macd",
        "ema_fast",
        "sma",
        "atr",
        "vwap",
        "bb_position",
        "bb_width",
        "zscore",
        "spread",
        "imbalance",
    )

    DEFAULT_MARKET_FIELDS = ("open", "high", "low", "close", "volume")
    FALLBACK_MARKET_FIELDS = ("bid", "ask", "last", "price")

    def __init__(
        self,
        symbol: str = "",
        timeframe: str = "1d",
        window: int | None = None,
        feature_names: Sequence[str] | None = None,
        include_derived: bool = True,
        fill_value: float = 0.0,
        config: FeaturePipelineConfig | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.include_derived = include_derived
        self.fill_value = float(fill_value)
        self.config = config or FeaturePipelineConfig()

        self.window = int(window or self.config.rolling_window)
        self.window = max(2, self.window)

        # Keep the public feature-name API, but enforce the fixed online
        # feature vector used by update().
        self.feature_names = (
            tuple(feature_names)
            if feature_names is not None
            else self.FEATURE_NAMES
        )

        if len(self.FEATURE_NAMES) != self.config.feature_dim:
            raise ValueError(
                f"feature_dim={self.config.feature_dim} does not match "
                f"{len(self.FEATURE_NAMES)} implemented features"
            )

        self.kalman = KalmanFilter1D()
        self.price_history: deque[float] = deque(maxlen=max(256, self.window + 2))
        self.return_history: deque[float] = deque(maxlen=max(256, self.window + 2))
        self.volume_history: deque[float] = deque(maxlen=max(256, self.window + 2))
        self.price_volume_history: deque[float] = deque(
            maxlen=max(256, self.window + 2)
        )

        self._last_price: float | None = None
        self.volatility_ewma = 0.0
        self.update_count = 0

        # Exponentially weighted running statistics for online normalization.
        self.feature_mean = np.zeros(self.config.feature_dim, dtype=float)
        self.feature_var = np.ones(self.config.feature_dim, dtype=float)
        self.feature_std = np.ones(self.config.feature_dim, dtype=float)

    # ------------------------------------------------------------------
    # Generic mapping transformation
    # ------------------------------------------------------------------

    def transform(self, data: Mapping[str, Any]) -> np.ndarray:
        """Convert a market mapping into a numeric feature vector.

        This method is useful for non-TickEvent callers. For the live
        event-driven path, prefer ``update(TickEvent)`` because it maintains
        rolling state.
        """
        if not isinstance(data, Mapping):
            raise TypeError(
                "FeaturePipeline.transform expects a mapping of market data"
            )

        numeric_values = flatten_numeric(data)
        if not numeric_values:
            raise ValueError(
                "FeaturePipeline.transform received no numeric values"
            )

        if self.feature_names is None:
            market_fields = [
                name
                for name in self.DEFAULT_MARKET_FIELDS
                if name in numeric_values
            ]
            if not market_fields:
                market_fields = [
                    name
                    for name in self.FALLBACK_MARKET_FIELDS
                    if name in numeric_values
                ]
            self.feature_names = tuple(market_fields or sorted(numeric_values))

        base_features = [
            numeric_values.get(name, self.fill_value)
            for name in self.feature_names
        ]

        derived_features = (
            self._derived_features(numeric_values)
            if self.include_derived
            else []
        )

        features = np.asarray(base_features + derived_features, dtype=float)
        features[~np.isfinite(features)] = self.fill_value
        return features

    def _derived_features(self, values: Mapping[str, float]) -> list[float]:
        """Compute simple stateless features from a market mapping."""
        open_price = values.get("open")
        high = values.get("high")
        low = values.get("low")
        close = values.get("close")
        volume = values.get("volume")
        bid = values.get("bid")
        ask = values.get("ask")
        price = self._price_from(values)

        mid_price = (
            (bid + ask) / 2.0
            if bid is not None and ask is not None
            else price
        )
        spread = (
            ask - bid
            if bid is not None and ask is not None
            else self.fill_value
        )
        log_volume = (
            math.log1p(max(float(volume), 0.0))
            if volume is not None
            else self.fill_value
        )
        price_return = (
            safe_ratio(price, self._last_price, self.fill_value) - 1.0
            if price is not None and self._last_price is not None
            else self.fill_value
        )

        return [
            finite_or_fill(mid_price, self.fill_value),
            finite_or_fill(spread, self.fill_value),
            safe_ratio(close, open_price, self.fill_value),
            safe_ratio(high, low, self.fill_value),
            range_position(close, low, high, self.fill_value),
            finite_or_fill(log_volume, self.fill_value),
            finite_or_fill(price_return, self.fill_value),
        ]

    @staticmethod
    def _price_from(values: Mapping[str, float]) -> float | None:
        """Extract the best available price from a market mapping."""
        for name in ("close", "last", "price"):
            value = values.get(name)
            if value is not None and np.isfinite(value):
                return float(value)
        return None

    # ------------------------------------------------------------------
    # Rolling calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _as_array(values: deque[float] | Sequence[float]) -> np.ndarray:
        return np.asarray(list(values), dtype=float)

    @staticmethod
    def _safe_std(values: np.ndarray) -> float:
        return float(np.std(values)) if values.size else 0.0

    @staticmethod
    def _ema(values: np.ndarray, span: int) -> float:
        if values.size == 0:
            return 0.0

        span = max(1, int(span))
        alpha = 2.0 / (span + 1.0)
        ema = float(values[0])

        for value in values[1:]:
            ema = alpha * float(value) + (1.0 - alpha) * ema

        return float(ema)

    def _rsi(self, returns: np.ndarray) -> float:
        """Compute RSI from recent returns."""
        period = self.config.rsi_period

        if returns.size < period:
            return 50.0

        recent = returns[-period:]
        gains = np.clip(recent, 0.0, None)
        losses = np.clip(-recent, 0.0, None)

        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))

        if avg_loss <= 1e-12:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    def _bollinger_features(self, prices: np.ndarray) -> tuple[float, float]:
        """Return band position and normalized band width."""
        period = self.config.bb_period

        if prices.size < period:
            return 0.5, 0.0

        recent = prices[-period:]
        sma = float(np.mean(recent))
        std = float(np.std(recent))
        current_price = float(recent[-1])

        upper = sma + 2.0 * std
        lower = sma - 2.0 * std
        width = upper - lower

        position = (
            (current_price - lower) / width
            if width > 1e-12
            else 0.5
        )
        position = float(np.clip(position, 0.0, 1.0))

        width_normalized = (
            width / abs(current_price)
            if abs(current_price) > 1e-12
            else 0.0
        )

        return position, float(width_normalized)

    # ------------------------------------------------------------------
    # Tick processing
    # ------------------------------------------------------------------

    def update(self, tick: TickEvent) -> StateVector:
        """Process one tick and return the current StateVector.

        No future observations are used. All rolling features only use data
        observed at or before this tick.
        """
        raw_price = self._extract_tick_price(tick)
        if raw_price <= 0.0:
            raise ValueError(f"Invalid tick price: {raw_price}")

        filtered_price, velocity = self.kalman.update(raw_price)
        filtered_price = float(filtered_price)
        velocity = float(velocity)

        self.price_history.append(filtered_price)

        # Return from the filtered price series.
        if len(self.price_history) >= 2:
            previous_price = self.price_history[-2]
            current_return = (
                (filtered_price - previous_price) / abs(previous_price)
                if abs(previous_price) > 1e-12
                else 0.0
            )
        else:
            current_return = 0.0

        self.return_history.append(float(current_return))

        volume = self._extract_float(tick, "volume", 0.0)
        volume = max(volume, 0.0)
        self.volume_history.append(volume)
        self.price_volume_history.append(filtered_price * volume)

        # EWMA volatility.
        halflife = max(float(self.config.volatility_halflife), 1e-6)
        decay = 1.0 - math.exp(-math.log(2.0) / halflife)
        self.volatility_ewma = (
            (1.0 - decay) * self.volatility_ewma
            + decay * abs(current_return)
        )

        prices = self._as_array(self.price_history)
        returns = self._as_array(self.return_history)

        recent_prices = prices[-self.window:]
        recent_returns = returns[-self.window:]

        # 1. Return.
        feature_return = float(current_return)

        # 2. Rolling volatility.
        rolling_vol = self._safe_std(recent_returns)

        # 3. Momentum.
        momentum_period = max(1, self.config.momentum_period)
        if prices.size > momentum_period:
            momentum = float(
                prices[-1] - prices[-1 - momentum_period]
            )
        else:
            momentum = 0.0

        # 4. RSI.
        rsi = self._rsi(returns) / 100.0

        # 5. MACD.
        macd = self._ema(
            prices,
            self.config.ema_fast_period,
        ) - self._ema(
            prices,
            self.config.ema_slow_period,
        )

        # 6. EMA fast.
        ema_fast = self._ema(
            recent_prices,
            self.config.ema_fast_period,
        )

        # 7. SMA.
        sma = (
            float(np.mean(recent_prices))
            if recent_prices.size
            else filtered_price
        )

        # 8. Tick-based ATR approximation.
        if prices.size >= 2:
            abs_changes = np.abs(np.diff(recent_prices))
            atr = float(np.mean(abs_changes)) if abs_changes.size else 0.0
        else:
            atr = 0.0

        # 9. VWAP.
        recent_volumes = self._as_array(self.volume_history)[-self.window:]
        price_volume = self._as_array(self.price_volume_history)[-self.window:]
        volume_sum = float(np.sum(recent_volumes))

        if volume_sum > 1e-12:
            vwap = float(np.sum(price_volume) / volume_sum)
        else:
            vwap = filtered_price

        # 10-11. Bollinger position and width.
        bb_position, bb_width = self._bollinger_features(prices)

        # 12. Price z-score.
        price_std = self._safe_std(recent_prices)
        zscore = (
            (filtered_price - sma) / price_std
            if price_std > 1e-12
            else 0.0
        )

        # 13. Spread.
        spread = self._extract_spread(tick, filtered_price)

        # 14. Order-book imbalance.
        imbalance = self._extract_float(tick, "imbalance", 0.0)
        imbalance = float(np.clip(imbalance, -1.0, 1.0))

        features = np.asarray(
            [
                feature_return,
                rolling_vol,
                momentum,
                rsi,
                macd,
                ema_fast,
                sma,
                atr,
                vwap,
                bb_position,
                bb_width,
                zscore,
                spread,
                imbalance,
            ],
            dtype=np.float64,
        )

        features[~np.isfinite(features)] = self.fill_value

        # Online normalization.
        normalized_features = self._update_normalization(features)

        self.update_count += 1
        self._last_price = filtered_price

        # These are placeholders until the GMM/HMM modules are connected.
        # They are intentionally explicit rather than pretending that fixed
        # probabilities are learned regime probabilities.
        regime_probs = np.full(3, 1.0 / 3.0, dtype=np.float64)

        hidden_state = np.asarray(
            [
                momentum,
                rolling_vol,
                imbalance,
                spread,
            ],
            dtype=np.float64,
        )

        metadata = self._tick_metadata(tick)

        return StateVector(
            timestamp=float(tick.timestamp),
            symbol=str(tick.symbol),
            timeframe=self.timeframe,
            features=normalized_features,
            filtered_price=filtered_price,
            volatility=float(rolling_vol),
            regime_probs=regime_probs,
            hidden_state=hidden_state,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_normalization(self, features: np.ndarray) -> np.ndarray:
        """Update online mean/variance and return normalized features."""
        alpha = float(np.clip(self.config.normalization_alpha, 1e-6, 1.0))

        if self.update_count == 0:
            self.feature_mean = features.copy()
            self.feature_var = np.ones_like(features)
            self.feature_std = np.ones_like(features)
            return features.copy()

        delta = features - self.feature_mean
        self.feature_mean += alpha * delta

        # EWMA variance update.
        self.feature_var = (
            (1.0 - alpha) * self.feature_var
            + alpha * delta * (features - self.feature_mean)
        )
        self.feature_var = np.maximum(self.feature_var, 1e-12)
        self.feature_std = np.sqrt(self.feature_var)

        if self.update_count < self.config.warmup_updates:
            return features.copy()

        return (features - self.feature_mean) / (self.feature_std + 1e-8)

    @staticmethod
    def _extract_float(tick: TickEvent, name: str, default: float) -> float:
        value = getattr(tick, name, None)
        if value is None:
            return float(default)

        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)

        return value if np.isfinite(value) else float(default)

    def _extract_tick_price(self, tick: TickEvent) -> float:
        """Extract a valid price from TickEvent."""
        for name in ("price", "last", "close"):
            value = getattr(tick, name, None)
            if value is None:
                continue

            try:
                price = float(value)
            except (TypeError, ValueError):
                continue

            if np.isfinite(price) and price > 0.0:
                return price

        bid = self._extract_float(tick, "bid", 0.0)
        ask = self._extract_float(tick, "ask", 0.0)

        if bid > 0.0 and ask > 0.0:
            return (bid + ask) / 2.0

        raise ValueError("TickEvent does not contain a valid market price")

    def _extract_spread(self, tick: TickEvent, price: float) -> float:
        spread = getattr(tick, "spread", None)

        if spread is not None:
            try:
                spread = float(spread)
                if np.isfinite(spread) and spread >= 0.0:
                    return spread
            except (TypeError, ValueError):
                pass

        bid = self._extract_float(tick, "bid", 0.0)
        ask = self._extract_float(tick, "ask", 0.0)

        if bid > 0.0 and ask > 0.0 and ask >= bid:
            return ask - bid

        return 0.0

    @staticmethod
    def _tick_metadata(tick: TickEvent) -> dict[str, Any]:
        metadata = getattr(tick, "metadata", None)
        return dict(metadata) if isinstance(metadata, Mapping) else {}

    def get_state(self) -> Tuple[float, float]:
        """Return the current Kalman state."""
        return self.kalman.get_state()

    def reset(self, initial_price: float = 0.0) -> None:
        """Reset all online state."""
        self.kalman.reset(float(initial_price))

        self.price_history.clear()
        self.return_history.clear()
        self.volume_history.clear()
        self.price_volume_history.clear()

        self._last_price = None
        self.volatility_ewma = 0.0
        self.feature_mean = np.zeros(
            self.config.feature_dim,
            dtype=float,
        )
        self.feature_var = np.ones(
            self.config.feature_dim,
            dtype=float,
        )
        self.feature_std = np.ones(
            self.config.feature_dim,
            dtype=float,
        )
        self.update_count = 0