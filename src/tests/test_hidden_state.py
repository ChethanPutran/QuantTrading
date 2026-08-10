import numpy as np

from learning.hidden_state import HiddenStateUpdater


def test_hidden_state_update_and_reset():
    hsu = HiddenStateUpdater(hidden_dim=4, decay=0.9, learning_rate=0.1)
    hidden = np.zeros(4)
    updated = hsu.update(hidden, error=0.5, features=np.array([1.0, 0.5, 0.2]))
    assert updated.shape == (4,)
    # bounded
    assert (updated <= 10.0).all()
    reset = hsu.reset()
    assert (reset == 0).all()
