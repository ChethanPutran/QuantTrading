"""Command-line entrypoint for the packaged trading system."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from app import TradingSystem
from config.settings import AppSettings, ReplaySettings, RuntimeSettings



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the adaptive probabilistic trading system")
    parser.add_argument("--symbol", default="^NSEI", help="Market symbol to stream or replay")
    parser.add_argument("--csv-path", type=Path, default=None, help="Replay a CSV file instead of synthetic streaming")
    parser.add_argument("--steps", type=int, default=500, help="Number of synthetic ticks to simulate when no CSV is provided")
    parser.add_argument("--delay-per-tick", type=float, default=0.0, help="Delay between replay ticks")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for synthetic replay")
    parser.add_argument("--state-store", type=Path, default=Path("results/state_store"), help="Directory for analytics outputs")
    parser.add_argument("--replay-store", type=Path, default=Path("results/replay_store"), help="Directory for replay logs")
    parser.add_argument("--redis-url", default=None, help="Optional Redis URL for live state publication")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def build_settings(args: argparse.Namespace) -> AppSettings:
    return AppSettings(
        replay=ReplaySettings(
            symbol=args.symbol,
            delay_per_tick=args.delay_per_tick,
            deterministic_seed=args.seed,
        ),
        runtime=RuntimeSettings(
            state_store_path=args.state_store,
            replay_store_path=args.replay_store,
            redis_url=args.redis_url,
            log_level=args.log_level,
        ),
    )


async def _run_async(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    system = TradingSystem(settings=build_settings(args))
    if args.csv_path is not None:
        frame = pd.read_csv(args.csv_path)
        await system.run_replay(frame, symbol=args.symbol)
    else:
        await system.run_synthetic(steps=args.steps)
    outputs = system.flush_storage()
    print(
        json.dumps(
            {
                "report": system.report(),
                "outputs": {name: str(path) for name, path in outputs.items()},
            },
            indent=2,
            default=str,
        )
    )
    return system.report()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()