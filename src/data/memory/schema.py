
from attr import dataclass



@dataclass
class PatternNode:
    pattern_id: str

    count: int

    avg_reward: float

    success_rate: float

    hidden_state: list

    regime_distribution: list

    child_patterns: list