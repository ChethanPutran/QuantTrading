"""
Pattern Database - stores and retrieves pattern memories.
"""

from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, Dict, Optional, Set
import numpy as np
from memory.pattern import PatternNode
from trading_system.memory.patterns import PatternRecord
from hashlib import sha1

class BasePatternDB(ABC):
    """Abstract pattern memory."""

    @abstractmethod
    def get(self, key: Any)-> PatternNode:
        pass

    @abstractmethod
    def update(self, key: Any, value: Any) -> None:
        pass


def _lsh_hash(vector: np.ndarray, buckets: int = 8) -> tuple[int, ...]:
    scaled = np.floor(np.asarray(vector, dtype=float) * buckets).astype(int)
    return tuple(int(value) for value in scaled[: min(32, scaled.size)])


class PatternDB(BasePatternDB):
    """
    Pattern memory database.
    
    Stores experience indexed by pattern key.
    Enables experience reuse and pattern-based learning.
    """
    patterns: dict[str, PatternRecord] = field(default_factory=dict)
    hash_index: dict[tuple[int, ...], list[str]] = field(default_factory=dict)

    def __init__(self, max_patterns: Optional[int] = None):
        """
        Initialize pattern database.
        
        Args:
            max_patterns: Maximum number of patterns to store (None for unlimited)
        """
        self.db: Dict[Any, PatternNode] = {}
        self.max_patterns = max_patterns
        self.access_count = {}
        self.creation_order = []

    
    def _allocate_id(self, feature_snapshot: np.ndarray) -> str:
        payload = np.asarray(feature_snapshot, dtype=float).tobytes()
        return sha1(payload).hexdigest()[:16]

    def upsert(self, record: PatternRecord) -> PatternRecord:
        self.patterns[record.pattern_id] = record
        key = _lsh_hash(record.feature_snapshot)
        self.hash_index.setdefault(key, []).append(record.pattern_id)
        return record

    def get_or_create(self, feature_snapshot: np.ndarray, hidden_state: np.ndarray, regime_probs: np.ndarray) -> PatternRecord:
        pattern_id = self._allocate_id(feature_snapshot)
        existing = self.patterns.get(pattern_id)
        if existing is not None:
            return existing
        record = PatternRecord(
            pattern_id=pattern_id,
            feature_snapshot=np.asarray(feature_snapshot, dtype=float),
            hidden_state=np.asarray(hidden_state, dtype=float),
            regime_probs=np.asarray(regime_probs, dtype=float),
            confidence=0.55,
        )
        return self.upsert(record)

    def retrieve(self, query: np.ndarray, top_k: int = 5) -> list[PatternRecord]:
        query = np.asarray(query, dtype=float)
        candidates = list(self.patterns.values())
        if not candidates:
            return []
        bucket = _lsh_hash(query)
        if bucket in self.hash_index:
            candidates = [self.patterns[pattern_id] for pattern_id in self.hash_index[bucket]]
        ranked = sorted(candidates, key=lambda record: record.score(query), reverse=True)
        return ranked[:top_k]

    def branch(self, parent: PatternRecord, feature_snapshot: np.ndarray, hidden_state: np.ndarray, regime_probs: np.ndarray) -> PatternRecord:
        branch_id = f"{parent.pattern_id[:8]}-{len(parent.reward_history)}"
        branch = PatternRecord(
            pattern_id=branch_id,
            feature_snapshot=np.asarray(feature_snapshot, dtype=float),
            hidden_state=np.asarray(hidden_state, dtype=float),
            regime_probs=np.asarray(regime_probs, dtype=float),
            confidence=max(0.1, parent.confidence * 0.8),
            parent_pattern_id=parent.pattern_id,
        )
        return self.upsert(branch)

    def update_feedback(self, pattern_id: str, reward: float, trajectory: np.ndarray, success: bool) -> None:
        record = self.patterns[pattern_id]
        record.reward_history.append(float(reward))
        record.trajectory_history.append(np.asarray(trajectory, dtype=float))
        record.confidence = float(np.clip(record.confidence + (0.05 if success else -0.05), 0.05, 0.99))
        if success:
            record.success_count += 1
        else:
            record.failure_count += 1

    
    def get(self, key: Any) -> PatternNode:
        """
        Get or create pattern node.
        
        Args:
            key: Pattern key
        
        Returns:
            PatternNode for this key
        """
        if key not in self.db:
            # Create new node
            self.db[key] = PatternNode(key=key)
            self.creation_order.append(key)
        
        # Track access
        self.access_count[key] = self.access_count.get(key, 0) + 1
        
        return self.db[key]
    
    def update(self, key: Any, value: PatternNode) -> None:
        """
        Update or add pattern node.
        
        Args:
            key: Pattern key
            node: Updated pattern node
        """
        self.db[key] = value
        if key not in self.access_count:
            self.access_count[key] = 0
            self.creation_order.append(key)
    
    def remove(self, key: Any) -> bool:
        """
        Remove a pattern from database.
        
        Args:
            key: Pattern key
        
        Returns:
            True if removed, False if not found
        """
        if key in self.db:
            del self.db[key]
            if key in self.access_count:
                del self.access_count[key]
            return True
        return False
    
    def get_node(self, key: Any) -> Optional[PatternNode]:
        """
        Get pattern node without creating if doesn't exist.
        
        Args:
            key: Pattern key
        
        Returns:
            PatternNode or None
        """
        return self.db.get(key)
    
    def get_all_keys(self) -> Set[Any]:
        """Get all pattern keys in database."""
        return set(self.db.keys())
    
    def get_size(self) -> int:
        """Get number of patterns in database."""
        return len(self.db)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.db:
            return {
                'n_patterns': 0,
                'total_count': 0,
                'mean_count': 0.0,
                'most_frequent': None
            }
        
        counts = [node.count for node in self.db.values()]
        
        most_frequent_key = max(
            self.db.keys(),
            key=lambda k: self.db[k].count
        )
        
        return {
            'n_patterns': len(self.db),
            'total_count': sum(counts),
            'mean_count': np.mean(counts),
            'max_count': max(counts),
            'min_count': min(counts),
            'most_frequent': most_frequent_key,
            'most_frequent_count': self.db[most_frequent_key].count
        }
    
    def prune_least_frequent(self, keep_ratio: float = 0.8) -> int:
        """
        Remove least frequent patterns to stay under memory limit.
        
        Args:
            keep_ratio: Ratio of patterns to keep
        
        Returns:
            Number of patterns removed
        """
        if self.max_patterns is None:
            return 0
        
        n_to_remove = max(0, len(self.db) - int(self.max_patterns * keep_ratio))
        
        if n_to_remove == 0:
            return 0
        
        # Sort by count (ascending)
        sorted_keys = sorted(
            self.db.keys(),
            key=lambda k: self.db[k].count
        )
        
        # Remove least frequent
        for key in sorted_keys[:n_to_remove]:
            self.remove(key)
        
        return n_to_remove
    
    def clear(self) -> None:
        """Clear all patterns from database."""
        self.db.clear()
        self.access_count.clear()
        self.creation_order.clear()
