from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from data.market_data.option_chain import get_index_option_chain_history
from regime.gmm import GaussianMixtureRegimeModel
from regime.hmm import MarkovTransitionModel
from utils.lstm_predictor import (
    LSTMPredictorConfig,
    predict_latest_profit,
    train_lstm_profit_model,
)


@dataclass
class RunSummary:
    symbol: str
    total_rows: int
    train_rows: int
    simulate_rows: int
    used_lstm: bool
    used_xgboost: bool
    final_portfolio_value: float
    final_pnl: float
    buy_trades: int
    sell_trades: int


def _snapshots_to_frame(snapshots: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        summary = snap.get("summary", {}) or {}
        calls_oi = float(summary.get("calls_open_interest", 0.0) or 0.0)
        puts_oi = float(summary.get("puts_open_interest", 0.0) or 0.0)
        calls_vol = float(summary.get("calls_volume", 0.0) or 0.0)
        puts_vol = float(summary.get("puts_volume", 0.0) or 0.0)

        rows.append(
            {
                "date": pd.to_datetime(snap.get("date")),
                "close": float(snap.get("underlying_price", np.nan)),
                "atm_distance": float(summary.get("atm_distance", 0.0) or 0.0),
                "calls_open_interest": calls_oi,
                "puts_open_interest": puts_oi,
                "calls_volume": calls_vol,
                "puts_volume": puts_vol,
                "put_call_oi_ratio": puts_oi / (calls_oi + 1e-9),
                "put_call_vol_ratio": puts_vol / (calls_vol + 1e-9),
                "synthetic": 1.0 if snap.get("synthetic") else 0.0,
            }
        )

    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return frame


def _add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"].astype(float)

    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["momentum_10"] = close / close.shift(10) - 1.0
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["vol_10"] = out["ret_1"].rolling(10).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    out["target_return_1d"] = out["ret_1"].shift(-1)
    return out.dropna().reset_index(drop=True)


def _feature_columns() -> list[str]:
    return [
        "ret_1",
        "ret_5",
        "momentum_10",
        "ema_12",
        "ema_26",
        "macd",
        "vol_10",
        "rsi_14",
        "atm_distance",
        "calls_open_interest",
        "puts_open_interest",
        "calls_volume",
        "puts_volume",
        "put_call_oi_ratio",
        "put_call_vol_ratio",
        "synthetic",
    ]


def _prepare_regime_features(
    x_values: np.ndarray,
    gmm: GaussianMixtureRegimeModel,
    hmm: MarkovTransitionModel,
) -> np.ndarray:
    stacked: list[np.ndarray] = []
    for row in x_values:
        g_probs = gmm.update(row)
        h_probs = hmm.update(g_probs)
        stacked.append(np.concatenate([row, g_probs, h_probs]))
    return np.vstack(stacked)


def _train_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_regimes: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    x_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["target_return_1d"].to_numpy(dtype=float)

    gmm = GaussianMixtureRegimeModel(n_regimes=n_regimes, feature_dim=x_train.shape[1])
    hmm = MarkovTransitionModel(n_regimes=n_regimes, smoothing=0.7)
    x_train_full = _prepare_regime_features(x_train, gmm, hmm)

    linear_model = LinearRegression()
    linear_model.fit(x_train_full, y_train)

    decision_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
    decision_tree.fit(x_train_full, y_train)

    gboost = GradientBoostingRegressor(random_state=42)
    gboost.fit(x_train_full, y_train)

    xgb_model = None
    used_xgboost = False
    try:
        from xgboost import XGBRegressor  # type: ignore

        xgb_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        xgb_model.fit(x_train_full, y_train)
        used_xgboost = True
    except Exception:
        xgb_model = None

    lstm_config = LSTMPredictorConfig(sequence_length=20, future_steps=1, epochs=10, batch_size=16)
    lstm_scaler = None
    used_lstm = False

    try:
        lstm_model, lstm_scaler = train_lstm_profit_model(
            train_df.rename(columns={"close": "Close"})[["Close"]],
            config=lstm_config,
        )
        lstm_model.save(str(artifact_dir / "lstm_model.keras"))
        joblib.dump(lstm_scaler, artifact_dir / "lstm_scaler.joblib")
        used_lstm = True
    except Exception:
        used_lstm = False

    artifacts = {
        "gmm": gmm,
        "hmm": hmm,
        "linear": linear_model,
        "decision_tree": decision_tree,
        "gboost": gboost,
        "xgboost": xgb_model,
        "feature_cols": feature_cols,
        "n_regimes": n_regimes,
        "used_lstm": used_lstm,
        "used_xgboost": used_xgboost,
        "lstm_config": asdict(lstm_config),
    }

    joblib.dump(artifacts, artifact_dir / "models.joblib")
    return artifacts


def _load_artifacts(artifact_dir: Path) -> dict[str, Any]:
    artifacts = joblib.load(artifact_dir / "models.joblib")
    return artifacts


def _predict_row(
    row_features: np.ndarray,
    gmm: GaussianMixtureRegimeModel,
    hmm: MarkovTransitionModel,
    models: dict[str, Any],
    close_history: pd.DataFrame,
    lstm_model: Any | None = None,
    lstm_scaler: Any | None = None,
) -> float:
    g_probs = gmm.update(row_features)
    h_probs = hmm.update(g_probs)
    x_full = np.concatenate([row_features, g_probs, h_probs]).reshape(1, -1)

    preds = [
        float(models["linear"].predict(x_full)[0]),
        float(models["decision_tree"].predict(x_full)[0]),
        float(models["gboost"].predict(x_full)[0]),
    ]

    if models.get("xgboost") is not None:
        preds.append(float(models["xgboost"].predict(x_full)[0]))

    if models.get("used_lstm") and lstm_model is not None and lstm_scaler is not None:
        try:
            cfg = LSTMPredictorConfig(**models["lstm_config"])
            profit_pred = predict_latest_profit(
                close_history.rename(columns={"close": "Close"})[["Close"]],
                model=lstm_model,
                scaler=lstm_scaler,
                config=cfg,
                feature_columns=["Close", "rsi", "macd", "ema", "bb_high", "bb_low"],
            )
            last_close = float(close_history["close"].iloc[-1])
            preds.append(profit_pred / (abs(last_close) + 1e-9))
        except Exception:
            pass

    return float(np.mean(preds))


def _simulate_last_month(
    engineered_df: pd.DataFrame,
    artifacts: dict[str, Any],
    simulate_days: int,
    symbol: str,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, RunSummary]:
    feature_cols = artifacts["feature_cols"]
    gmm = artifacts["gmm"]
    hmm = artifacts["hmm"]

    sim_df = engineered_df.tail(simulate_days).copy().reset_index(drop=True)

    lstm_model = None
    lstm_scaler = None
    if artifacts.get("used_lstm"):
        try:
            from tensorflow.keras.models import load_model  # type: ignore

            lstm_model = load_model(str(artifact_dir / "lstm_model.keras"))
            lstm_scaler = joblib.load(artifact_dir / "lstm_scaler.joblib")
        except Exception:
            lstm_model = None
            lstm_scaler = None

    cash = 100000.0
    position = 0.0
    buy_trades = 0
    sell_trades = 0

    rows: list[dict[str, Any]] = []
    context_df = engineered_df.copy().reset_index(drop=True)

    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        close_px = float(row["close"])
        row_features = row[feature_cols].to_numpy(dtype=float)

        # Keep a rolling window until current row for LSTM context.
        history_until_now = context_df[context_df["date"] <= row["date"]][["date", "close"]].copy()
        pred = _predict_row(
            row_features,
            gmm,
            hmm,
            artifacts,
            history_until_now,
            lstm_model=lstm_model,
            lstm_scaler=lstm_scaler,
        )

        action = 0
        if pred > 0.001:
            action = 1
        elif pred < -0.001:
            action = -1

        if action == 1 and position == 0:
            position = cash / close_px
            cash = 0.0
            buy_trades += 1
        elif action == -1 and position > 0:
            cash = position * close_px
            position = 0.0
            sell_trades += 1

        portfolio_value = cash + position * close_px
        rows.append(
            {
                "date": row["date"],
                "close": close_px,
                "predicted_return": pred,
                "action": action,
                "cash": cash,
                "position": position,
                "portfolio_value": portfolio_value,
            }
        )

    if len(sim_df) > 0 and position > 0:
        final_close = float(sim_df["close"].iloc[-1])
        cash = position * final_close
        position = 0.0

    final_value = cash
    summary = RunSummary(
        symbol=symbol,
        total_rows=int(len(engineered_df)),
        train_rows=int(len(engineered_df) - len(sim_df)),
        simulate_rows=int(len(sim_df)),
        used_lstm=bool(artifacts.get("used_lstm", False)),
        used_xgboost=bool(artifacts.get("used_xgboost", False)),
        final_portfolio_value=float(final_value),
        final_pnl=float(final_value - 100000.0),
        buy_trades=int(buy_trades),
        sell_trades=int(sell_trades),
    )
    return pd.DataFrame(rows), summary


def run_train_simulate(
    symbol: str,
    total_days: int,
    simulate_days: int,
    output_csv: Path,
    summary_json: Path,
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    snapshots = get_index_option_chain_history(symbol=symbol, days=total_days, interval="1d")
    if len(snapshots) < simulate_days + 30:
        raise ValueError("Not enough history for train+simulate split")

    market_df = _snapshots_to_frame(snapshots)
    engineered = _add_technical_features(market_df)

    split = max(30, len(engineered) - simulate_days)
    train_df = engineered.iloc[:split].copy()

    feature_cols = _feature_columns()
    artifacts = _train_models(
        train_df=train_df,
        feature_cols=feature_cols,
        n_regimes=3,
        artifact_dir=artifact_dir,
    )

    loaded = _load_artifacts(artifact_dir)
    result_df, summary = _simulate_last_month(
        engineered_df=engineered,
        artifacts=loaded,
        simulate_days=simulate_days,
        symbol=symbol,
        artifact_dir=artifact_dir,
    )

    result_df.to_csv(output_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as fp:
        json.dump(asdict(summary), fp, indent=2)

    print(f"Saved simulation output: {output_csv}")
    print(f"Saved summary: {summary_json}")
    print(json.dumps(asdict(summary), indent=2))


def run_load_and_simulate(
    symbol: str,
    total_days: int,
    simulate_days: int,
    output_csv: Path,
    summary_json: Path,
    artifact_dir: Path,
) -> None:
    snapshots = get_index_option_chain_history(symbol=symbol, days=total_days, interval="1d")
    market_df = _snapshots_to_frame(snapshots)
    engineered = _add_technical_features(market_df)

    artifacts = _load_artifacts(artifact_dir)
    result_df, summary = _simulate_last_month(
        engineered_df=engineered,
        artifacts=artifacts,
        simulate_days=simulate_days,
        symbol=symbol,
        artifact_dir=artifact_dir,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as fp:
        json.dump(asdict(summary), fp, indent=2)

    print(f"Loaded artifacts from: {artifact_dir}")
    print(f"Saved simulation output: {output_csv}")
    print(f"Saved summary: {summary_json}")
    print(json.dumps(asdict(summary), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train market-only models (GMM/HMM/LSTM/Linear/DecisionTree/GradientBoosting/XGBoost if available) "
            "on ~7 months of NIFTY50 option-chain-derived market data and simulate last 1 month."
        )
    )
    parser.add_argument("--symbol", default="^NSEI", help="Index symbol (default: ^NSEI)")
    parser.add_argument("--total-days", type=int, default=210, help="Total history days (~7 months)")
    parser.add_argument("--simulate-days", type=int, default=30, help="Last N days simulation window")
    parser.add_argument(
        "--mode",
        choices=["train-simulate", "simulate-only"],
        default="train-simulate",
        help="train models then simulate, or only load existing artifacts and simulate",
    )
    parser.add_argument(
        "--artifact-dir",
        default="results/nifty50_market_only_artifacts",
        help="Directory to save/load trained artifacts",
    )
    parser.add_argument(
        "--output-csv",
        default="results/nifty50_last_month_simulation.csv",
        help="Path to save per-step simulation output CSV",
    )
    parser.add_argument(
        "--summary-json",
        default="results/nifty50_last_month_summary.json",
        help="Path to save run summary JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)

    if args.mode == "train-simulate":
        run_train_simulate(
            symbol=args.symbol,
            total_days=args.total_days,
            simulate_days=args.simulate_days,
            output_csv=output_csv,
            summary_json=summary_json,
            artifact_dir=artifact_dir,
        )
    else:
        run_load_and_simulate(
            symbol=args.symbol,
            total_days=args.total_days,
            simulate_days=args.simulate_days,
            output_csv=output_csv,
            summary_json=summary_json,
            artifact_dir=artifact_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
