import numpy as np

from regime.gmm import GaussianMixtureRegimeModel
from regime.hmm import MarkovTransitionModel


def test_gmm_hmm_basic():
    gmm = GaussianMixtureRegimeModel(n_regimes=3, feature_dim=4, learning_rate=0.5)
    x = np.random.randn(4)
    probs = gmm.update(x)
    assert probs.shape == (3,)
    assert abs(probs.sum() - 1.0) < 1e-6

    hmm = MarkovTransitionModel(n_regimes=3, smoothing=0.5)
    out = hmm.update(probs)
    assert out.shape == (3,)
    assert abs(out.sum() - 1.0) < 1e-6
