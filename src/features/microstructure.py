from collections.abc import Mapping
from typing import Any

from .base import BaseFeatureTransformer
from .transforms import clean_feature_dict, safe_ratio


def calculate_microstructure_features(raw: Mapping[str, Any]) -> dict[str, float]:
    bid = raw.get("bid")
    ask = raw.get("ask")
    price = raw.get("price") or raw.get("last") or raw.get("close")
    volume = raw.get("volume")
    previous_volume = raw.get("previous_volume")
    vwap = raw.get("vwap")

    spread = ask - bid if bid is not None and ask is not None else None
    mid = (bid + ask) / 2 if bid is not None and ask is not None else None

    return clean_feature_dict(
        {
            "bid": bid,
            "ask": ask,
            "mid_price": mid,
            "ask_bid_spread": spread,
            "volume_change_pct": safe_ratio(volume, previous_volume),
            "vwap_deviation": safe_ratio(price, vwap),
            "short_ratio": raw.get("short_ratio"),
            "short_interest_pct": raw.get("short_interest_pct"),
            "put_call_ratio": raw.get("put_call_ratio"),
            "dark_pool_pct": raw.get("dark_pool_pct"),
        }
    )


class MicrostructureFeatureTransformer(BaseFeatureTransformer):
    def transform(self, data: Mapping[str, Any]) -> dict[str, float]:
        return calculate_microstructure_features(data)
