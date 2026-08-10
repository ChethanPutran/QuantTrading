
from dataclasses import dataclass



@dataclass
class VolatilityEvent:
    timestamp: int

    symbol: str

    realized_vol: float
    implied_vol: float

    vix: float

    volatility_regime: str