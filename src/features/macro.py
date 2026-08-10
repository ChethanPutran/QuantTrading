from collections.abc import Mapping
from typing import Any

from .base import BaseFeatureTransformer
from .transforms import clean_feature_dict


FRED_SERIES = {
    "cpi": "CPIAUCSL",
    "core_inflation": "CPILFESL",
    "unemployment_rate": "UNRATE",
    "gdp_growth": "A191RL1Q225SBEA",
    "fed_rate": "FEDFUNDS",
    "real_interest_rate": "INTDSRUSM193N",
    "yield_curve_spread": "T10Y2Y",
    "dxy": "DTWEXBGS",
    "oil_price": "DCOILWTICO",
    "gold_price": "IR14270",
    "vix": "VIXCLS",
    "retail_sales_growth": "MRTSSM44X72USS",
    "trade_balance": "NETEXP",
}


def latest_macro_features(series_by_name: Mapping[str, Any]) -> dict[str, float]:
    latest_values = {}

    for name, series in series_by_name.items():
        if hasattr(series, "dropna") and hasattr(series, "iloc"):
            clean = series.dropna()
            latest_values[name] = clean.iloc[-1] if not clean.empty else None
        else:
            latest_values[name] = series

    return clean_feature_dict(latest_values)


class MacroFeatureTransformer(BaseFeatureTransformer):
    def transform(self, data: Mapping[str, Any]) -> dict[str, float]:
        return latest_macro_features(data)
