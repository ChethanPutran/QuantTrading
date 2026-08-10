"""Volatility-aware position sizing and exposure control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskEngine:
    max_position_fraction: float = 0.03
    max_drawdown_fraction: float = 0.15
    confidence_floor: float = 0.55
    liquidity_penalty: float = 0.25

    def size_position(self, *, portfolio_value: float, confidence: float, volatility: float, liquidity: float) -> float:
        confidence_scale = max(0.0, confidence - self.confidence_floor) / max(1e-12, 1.0 - self.confidence_floor)
        volatility_scale = 1.0 / (1.0 + 10.0 * max(0.0, volatility))
        liquidity_scale = 1.0 / (1.0 + self.liquidity_penalty * max(0.0, 1.0 - liquidity))
        raw_fraction = self.max_position_fraction * confidence_scale * volatility_scale * liquidity_scale
        return max(0.0, portfolio_value * raw_fraction)
