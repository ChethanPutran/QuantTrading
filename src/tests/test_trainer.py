import numpy as np

from learning.trainer import OnlineTrainer
from learning.hidden_state import HiddenStateUpdater
from memory.pattern_db import PatternDB
from memory.encoder import PatternEncoder


class DummyModel:
    def __init__(self):
        self.updates = 0

    def update(self, x, y):
        self.updates += 1


def test_trainer_updates_pattern_and_stats():
    model = DummyModel()
    hsu = HiddenStateUpdater(hidden_dim=4)
    pdb = PatternDB()
    encoder = PatternEncoder(feature_bin_size=0.1, regime_dim=2)

    trainer = OnlineTrainer(
        model=model,
        hidden_state_updater=hsu,
        pattern_db=pdb,
        pattern_encoder=encoder,
        error_threshold=0.01,
        branching_enabled=False
    )

    features = np.array([0.1, 0.2, 0.3, 0.4])
    regime = np.array([0.6, 0.4])
    experience = {
        'features': features,
        'regime_probs': regime,
        'prediction': 0.0,
        'actual_return': 0.05,
        'hidden_state': np.zeros(4)
    }

    stats = trainer.update(experience)
    assert 'error' in stats
    s = trainer.get_statistics()
    assert s['update_count'] >= 1
    assert pdb.get_size() >= 1
