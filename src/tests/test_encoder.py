import numpy as np

from memory.encoder import PatternEncoder


def test_encoder_basic_and_confidence():
    enc = PatternEncoder(feature_bin_size=0.1, regime_dim=3)
    features = np.array([0.2, -0.1, 0.05])
    regime = np.array([0.1, 0.8, 0.1])
    key = enc.encode(features, regime)
    assert isinstance(key, tuple)
    key2, conf = enc.encode_with_confidence(features, regime)
    assert isinstance(conf, float)
    assert conf == float(np.max(regime))
