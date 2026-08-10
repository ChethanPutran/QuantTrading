"""
Trainer for online learning and model updates.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
from memory.pattern_db import PatternDB
from memory.encoder import PatternEncoder


class BaseTrainer(ABC):
    """Handles learning updates."""

    @abstractmethod
    def update(self, experience: dict) -> None:
        pass


class OnlineTrainer(BaseTrainer):
    """
    Online learning trainer.
    
    Updates:
    - Prediction model weights
    - Hidden state
    - Pattern memory statistics
    - Triggers branching on high error
    """
    
    def __init__(
        self,
        model,
        hidden_state_updater,
        pattern_db: PatternDB,
        pattern_encoder: PatternEncoder,
        error_threshold: float = 1.0,
        branching_enabled: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: Prediction model
            hidden_state_updater: Hidden state updater
            pattern_db: Pattern database
            pattern_encoder: Pattern encoder
            error_threshold: Threshold for branching
            branching_enabled: Whether to enable pattern branching
        """
        self.model = model
        self.hidden_state_updater = hidden_state_updater
        self.pattern_db = pattern_db
        self.pattern_encoder = pattern_encoder
        self.error_threshold = error_threshold
        self.branching_enabled = branching_enabled
        
        # Statistics
        self.update_count = 0
        self.total_error = 0.0
        self.errors_over_threshold = 0
        self.branches_created = 0
    
    def update(
        self,
        experience: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update models based on experience.
        
        Experience dictionary should contain:
        - features: feature vector
        - regime_probs: regime probabilities
        - prediction: predicted return
        - actual_return: actual realized return
        - hidden_state: current hidden state
        
        Returns:
            Update statistics
        """
        # Extract from experience
        features = experience.get('features', None)
        regime_probs = experience.get('regime_probs', None)
        prediction = experience.get('prediction', 0.0)
        actual_return = experience.get('actual_return', 0.0)
        hidden_state = experience.get('hidden_state', None)
        
        # Compute error
        error = actual_return - prediction
        abs_error = abs(error)
        
        # Update prediction model
        if self.model is not None and features is not None:
            self.model.update(features, actual_return)
        
        # Update hidden state
        if hidden_state is not None:
            if self.hidden_state_updater is not None:
                hidden_state = self.hidden_state_updater.update(
                    hidden_state,
                    error,
                    features
                )
        
        # Update pattern memory
        if regime_probs is not None and self.pattern_encoder is not None:
            pattern_key = self.pattern_encoder.encode(features, regime_probs)
            pattern_node = self.pattern_db.get(pattern_key)
            pattern_node.update(actual_return, abs_error, hidden_state)
            
            # Check for branching
            if self.branching_enabled and abs_error > self.error_threshold:
                # High error persists - create branch
                if pattern_node.count > 10:  # Only branch after seeing pattern enough times
                    high_error_count = sum(
                        1 for err in pattern_node.hidden_states_history
                        if err is not None
                    )
                    
                    if high_error_count > len(pattern_node.hidden_states_history) // 3:
                        # More than 1/3 of observations have high error
                        branch_name = f"error_{self.branches_created}"
                        pattern_node.create_branch(branch_name)
                        self.branches_created += 1
        
        # Track statistics
        self.update_count += 1
        self.total_error += abs_error
        
        if abs_error > self.error_threshold:
            self.errors_over_threshold += 1
        
        return {
            'error': error,
            'abs_error': abs_error,
            'model_updated': self.model is not None,
            'hidden_state_updated': hidden_state is not None,
            'pattern_updated': regime_probs is not None,
            'branch_created': (
                abs_error > self.error_threshold and
                self.branching_enabled
            ),
            'hidden_state': hidden_state
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'update_count': self.update_count,
            'mean_error': self.total_error / max(self.update_count, 1),
            'errors_over_threshold': self.errors_over_threshold,
            'error_ratio': (
                self.errors_over_threshold / max(self.update_count, 1)
            ),
            'branches_created': self.branches_created,
            'n_patterns': self.pattern_db.get_size()
        }
    
    def reset(self) -> None:
        """Reset trainer statistics."""
        self.update_count = 0
        self.total_error = 0.0
        self.errors_over_threshold = 0
        self.branches_created = 0