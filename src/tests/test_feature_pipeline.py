import numpy as np

from features.feature_pipeline import FeaturePipeline


def test_feature_pipeline_shapes_and_reset():
    p = FeaturePipeline(feature_dim=8)
    feats = p.update(100.0)
    assert feats.shape == (8,)
    # multiple updates
    for price in [100.5, 101.0, 99.8, 100.2]:
        f = p.update(price)
        assert f.shape == (8,)
        assert np.isfinite(f).all()

    # reset
    p.reset(100.0)
    s = p.get_state()
    assert isinstance(s, tuple)
