import numpy as np
import pytest

from features import FeaturePipeline


def test_transform_extracts_stable_market_features():
    pipeline = FeaturePipeline(include_derived=False)

    first = pipeline.transform(
        {
            "close": 101,
            "open": 100,
            "high": 102,
            "low": 99,
            "volume": 1_000,
        }
    )
    second = pipeline.transform(
        {
            "extra": 999,
            "volume": 1_100,
            "low": 100,
            "high": 103,
            "open": 101,
            "close": 102,
        }
    )

    np.testing.assert_array_equal(first, np.array([100, 102, 99, 101, 1000.0]))
    np.testing.assert_array_equal(second, np.array([101, 103, 100, 102, 1100.0]))


def test_transform_adds_derived_features_and_price_return():
    pipeline = FeaturePipeline()

    first = pipeline.transform(
        {
            "open": 100,
            "high": 110,
            "low": 90,
            "close": 105,
            "volume": 99,
            "bid": 104,
            "ask": 106,
        }
    )
    second = pipeline.transform(
        {
            "open": 105,
            "high": 115,
            "low": 95,
            "close": 110,
            "volume": 99,
            "bid": 109,
            "ask": 111,
        }
    )

    assert len(first) == 12
    assert first[-7] == 105
    assert first[-6] == 2
    assert first[-1] == 0
    assert second[-1] == pytest.approx((110 / 105) - 1)


def test_transform_flattens_numeric_fallback_fields():
    pipeline = FeaturePipeline(include_derived=False)

    features = pipeline.transform(
        {
            "meta": "ignored",
            "signals": {"alpha": "0.25", "beta": 2},
            "levels": [10, 20],
        }
    )

    np.testing.assert_array_equal(features, np.array([10, 20, 0.25, 2]))


def test_transform_rejects_non_numeric_data():
    pipeline = FeaturePipeline()

    with pytest.raises(ValueError, match="no numeric values"):
        pipeline.transform({"symbol": "BTCUSD"})
