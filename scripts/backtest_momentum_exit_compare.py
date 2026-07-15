"""Head-to-head backtest: SIMPLE shared-engine exits vs the MOMENTUM machinery.

Same entries (top-N rank + 1H trigger) and same synthetic-option pricing; only the
EXIT policy differs:
  * momentum   — trailing stop + initial ATR stop + score-decay + time stop
                 (strategies/momentum_expansion/backtest/simulate.py::_check_exit)
  * simple_hold— the live shared 4H engine: TP +20% option -> sell half -> ride the
                 rest to a 25-bar horizon / 3-bar drop-out grace.

Sizing-neutral (unconstrained, $1k notional/trade) so the equity curve reflects
signal+policy quality, not position sizing. Reports on the OOS test window.

Run: PYTHONPATH=. .venv/bin/python scripts/backtest_momentum_exit_compare.py [--start ... --end ...]
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from strategies.momentum_expansion.backtest.simulate import MomentumBacktester
from strategies.momentum_expansion.config.momentum_config import TRAINING_MATRIX
from strategies.momentum_expansion.inference.ranker import ExpansionRanker


def _fmt(m: dict) -> str:
    x = m["metrics"]
    return (f"trades={m['trades']:>5}  win={x['winrate']:.1%}  "
            f"avg_trade={x['avg_trade_pct_return']:+.2%}  med={x['median_trade_pct_return']:+.2%}  "
            f"CAGR={x['cagr']:+.1%}  Sharpe={x['sharpe']:.2f}  maxDD={x['max_drawdown']:.1%}  "
            f"finalEq=${x['final_equity']:,.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-02-06")   # OOS test window (after the 80% split)
    ap.add_argument("--end", default="2026-05-14")
    ap.add_argument("--modes", default="momentum,simple_hold")
    args = ap.parse_args()

    print(f"loading features {TRAINING_MATRIX} ...")
    feats = pd.read_parquet(TRAINING_MATRIX)
    print(f"window {args.start} -> {args.end}")

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    results = {}
    for mode in args.modes.split(","):
        mode = mode.strip()
        print(f"\n=== running exit_mode={mode} ===")
        bt = MomentumBacktester(ranker=ExpansionRanker(), unconstrained=True, exit_mode=mode)
        results[mode] = bt.run(features=feats, start=start, end=end)
        print(f"  {mode:12} {_fmt(results[mode])}")

    print("\n================ SUMMARY ================")
    for mode, r in results.items():
        print(f"{mode:12} {_fmt(r)}")


if __name__ == "__main__":
    main()
