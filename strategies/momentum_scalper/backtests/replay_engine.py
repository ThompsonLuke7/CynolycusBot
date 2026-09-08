"""Compatibility CLI for the chronological, quote-aware replay engine."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from strategies.momentum_scalper.configs.v1 import load_config
from strategies.momentum_scalper.replay.event_engine import MomentumReplayEngine, ReplayResult
from strategies.momentum_scalper.utils.io import write_parquet


def replay(
    *,
    bars: pd.DataFrame,
    metadata: pd.DataFrame,
    news: pd.DataFrame,
    quotes: pd.DataFrame,
    halts: pd.DataFrame | None = None,
    risk_budget_dollars: float,
) -> ReplayResult:
    return MomentumReplayEngine(
        config=load_config(), risk_budget_dollars=risk_budget_dollars,
    ).run(bars=bars, metadata=metadata, news=news, quotes=quotes, halts=halts)


def _read(path: Path | None) -> pd.DataFrame:
    return pd.read_parquet(path) if path is not None else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal, quote-aware momentum scalper replay")
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--news", type=Path, required=True)
    parser.add_argument("--quotes", type=Path, required=True)
    parser.add_argument("--halts", type=Path)
    parser.add_argument("--risk-budget-dollars", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = replay(
        bars=_read(args.bars), metadata=_read(args.metadata), news=_read(args.news),
        quotes=_read(args.quotes), halts=_read(args.halts),
        risk_budget_dollars=args.risk_budget_dollars,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("decisions", "orders", "fills", "transitions"):
        write_parquet(result.frame(name), args.output_dir / f"{name}.parquet")
    print(f"wrote replay ledgers to {args.output_dir}")


if __name__ == "__main__":
    main()


__all__ = ["replay"]
