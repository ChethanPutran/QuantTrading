"""Application configuration for the trading intelligence system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_settings import BaseSettings


@dataclass(slots=True)
class ReplaySettings:
    symbol: str = "^NSEI"
    lookback_days: int = 180
    simulate_days: int = 30
    interval: str = "1d"
    delay_per_tick: float = 0.0
    deterministic_seed: int = 42


@dataclass(slots=True)
class RiskSettings:
    max_position_fraction: float = 0.03
    max_drawdown_fraction: float = 0.15
    min_confidence: float = 0.55
    max_gross_exposure: float = 1.0
    liquidity_penalty: float = 0.25



@dataclass(slots=True)
class PortfolioSettings(BaseSettings):
    initial_cash: float = 100_000.0
    transaction_cost_rate: float = 0.0005
    slippage_rate: float = 0.0002
    initial_position: float = 0.0
    initial_price: float = 100.0
    max_position: float = 1.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    index_symbol: str = "^NSEI"


class MPCSettings(BaseSettings):
    mpc_horizon: int = 5


class RegimeModelSettings(BaseSettings):
    feature_dim: int = 8
    n_regimes: int = 3

class RuntimeSettings(BaseSettings):
    queue_size: int = 1024
    event_timeout_seconds: float = 2.0
    state_store_path: Path = Path("results/state_store")
    replay_store_path: Path = Path("results/replay_store")
    redis_url: str | None = None
    log_level: str = "INFO"
    enable_latency_tracking: bool = True


class FeaturePipelineSettings(BaseSettings):
    """Configuration for the online feature pipeline."""

    feature_dim: int = 14
    kalman_process_noise: float = 1e-5
    kalman_measurement_noise: float = 1e-2
    volatility_halflife: float = 20.0
    rsi_period: int = 14
    bb_period: int = 20
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    momentum_period: int = 5
    rolling_window: int = 64
    warmup_updates: int = 100
    normalization_alpha: float = 0.01



class PipelineSettings(BaseSettings):
    """Configuration for the entire trading pipeline."""
    replay: ReplaySettings = ReplaySettings()
    risk: RiskSettings = RiskSettings()
    portfolio: PortfolioSettings = PortfolioSettings()
    mpc: MPCSettings = MPCSettings()
    regime_model: RegimeModelSettings = RegimeModelSettings()
    runtime: RuntimeSettings = RuntimeSettings()
    feature_pipeline: FeaturePipelineSettings = FeaturePipelineSettings()

@dataclass(slots=True)
class AppSettings():
    portfolio: PortfolioSettings = field(default_factory=PortfolioSettings)
    regime_model: RegimeModelSettings = field(default_factory=RegimeModelSettings)
    mpc: MPCSettings = field(default_factory=MPCSettings)
    replay: ReplaySettings = field(default_factory=ReplaySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    feature_pipeline: FeaturePipelineSettings = field(default_factory=FeaturePipelineSettings)