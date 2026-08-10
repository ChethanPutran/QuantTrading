from collections.abc import Mapping
from typing import Any

import pandas as pd

from .base import BaseFeatureTransformer
from .transforms import clean_feature_dict


def calculate_fundamental_features(info: Mapping[str, Any]) -> dict[str, float]:
    market_cap = info.get("marketCap") or 1
    interest_expense = info.get("interestExpense")

    features = {
        "market_cap": info.get("marketCap"),
        "eps_ttm": info.get("trailingEps"),
        "pe_ratio": info.get("trailingPE"),
        "peg_ratio": info.get("pegRatio"),
        "pb_ratio": info.get("priceToBook"),
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "net_margin": info.get("netMargins"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "dividend_yield": info.get("dividendYield"),
        "free_cash_flow_yield": (info.get("freeCashflow") or 0) / market_cap,
        "buyback_yield": info.get("buyBackYield"),
        "debt_to_equity": info.get("debtToEquity"),
        "interest_coverage": (
            info.get("ebitda") / interest_expense
            if interest_expense
            else None
        ),
        "institutional_holdings": info.get("heldPercentInstitutions"),
        "insider_holdings": info.get("heldPercentInsiders"),
    }
    return clean_feature_dict(features)


def compute_quarterly_metrics(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    info: Mapping[str, Any],
    start_date,
    end_date,
) -> pd.DataFrame:
    price = info.get("currentPrice")
    shares = info.get("sharesOutstanding")
    market_cap = info.get("marketCap") or ((price or 0) * (shares or 0))

    df = pd.concat([income.T, balance.T], axis=1).sort_index(ascending=False)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df = df.astype("float").bfill().ffill()

    df["EBITDA Margin"] = df["EBITDA"] / df["Total Revenue"]
    df["EBIT Margin"] = df["EBIT"] / df["Total Revenue"]
    df["Net Profit Margin"] = df["Net Income"] / df["Total Revenue"]
    df["Basic EPS"] = df["Net Income"] / df["Basic Average Shares"]
    df["Current Ratio"] = df["Current Assets"] / df["Current Liabilities"]
    df["Quick Ratio"] = (
        df["Current Assets"] - df["Inventory"]
    ) / df["Current Liabilities"]
    df["Debt-to-Equity"] = df["Total Debt"] / df["Stockholders Equity"]
    df["Debt Ratio"] = df["Total Debt"] / df["Total Assets"]
    df["Asset Turnover"] = df["Total Revenue"] / df["Total Assets"]
    df["Inventory Turnover"] = df["Cost Of Revenue"] / df["Inventory"]
    df["EPS"] = df["Net Income"] / df["Basic Average Shares"]
    df["P/E Ratio"] = price / df["EPS"]
    df["P/B Ratio"] = price / (df["Stockholders Equity"] / shares)
    df["P/S Ratio"] = price / (df["Total Revenue"] / shares)
    df["ROE"] = df["Net Income"] / df["Stockholders Equity"]
    df["Net Margin"] = df["Net Income"] / df["Total Revenue"]
    df["EV"] = market_cap + df["Total Debt"] - df["Cash And Cash Equivalents"]
    df["EV/EBITDA"] = df["EV"] / df["EBITDA"]
    df["Revenue"] = df["Total Revenue"]
    return df


class FundamentalFeatureTransformer(BaseFeatureTransformer):
    def transform(self, data: Mapping[str, Any]) -> dict[str, float]:
        return calculate_fundamental_features(data)
