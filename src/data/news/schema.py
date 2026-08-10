
from dataclasses import dataclass



@dataclass
class NewsEvent:
    timestamp: int

    headline: str

    source: str

    sentiment: float
    uncertainty: float

    relevance: float

    topic: str

    entities: list

    embedding: list