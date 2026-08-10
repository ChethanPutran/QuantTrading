import numpy as np

from control.mpc import MPCController


class DummyModel:
    def predict(self, features):
        return 0.01


def test_mpc_basic_decision_and_stats():
    model = DummyModel()
    mpc = MPCController(model=model, horizon=3, action_space=[-1, 0, 1])
    state = np.zeros(5)
    action = mpc.decide(state, features=np.zeros(1))
    assert isinstance(action, int)
    mpc.set_position(0.5)
    stats = mpc.get_stats()
    assert 'current_position' in stats
    mpc.reset()
    assert mpc.current_position == 0.0
