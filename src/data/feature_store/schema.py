
from attr import dataclass



@dataclass
class FeatureVector:
    timestamp: int

    symbol: str

    features: dict

    regime_probs: list

    hidden_state: list