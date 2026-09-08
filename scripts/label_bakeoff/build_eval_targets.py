"""Independent evaluation targets for the label bake-off.

The candidates must be judged on something NONE of them is, or a candidate that
happens to equal the metric wins by definition. Two targets:

  eval_mfe_10b  forward MFE in ATR over 10 x 4H bars (~5 trading days) -- the
                horizon the modules ACTUALLY trade (Stage 6: median hold 4-5 days)
  eval_clean_10b  the same, minus the adverse excursion over the same window --
                a "clean move" target no candidate label equals

Computed from Data/shared/bars/4h, forward-only, and ATR is taken from the bar
BEFORE the decision so the normaliser carries no look-ahead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
BARS = REPO / "Data/shared/bars/4h"
OUT = REPO / "research/execution_quality/data/eval_targets_4h.parquet"
FWD_BARS = 10          # ~5 trading days at 2 x 4H bars/day


def one(ticker: str) -> pd.DataFrame | None:
    path = BARS / f"{ticker}.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    if len(d) < 60:
        return None
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values("timestamp").reset_index(drop=True)
    prev = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(),
                    (d["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean().shift(1)      # prior bar: no look-ahead
    ref = d["close"]
    # Forward window is bars t+1 .. t+FWD_BARS (never includes the decision bar).
    fwd_high = d["high"].shift(-1).rolling(FWD_BARS, min_periods=FWD_BARS).max().shift(-(FWD_BARS - 1))
    fwd_low = d["low"].shift(-1).rolling(FWD_BARS, min_periods=FWD_BARS).min().shift(-(FWD_BARS - 1))
    good = atr.notna() & (atr > 0) & ref.notna() & (ref > 0)
    out = pd.DataFrame({
        "timestamp": d["timestamp"],
        "ticker": ticker,
        "eval_mfe_10b": np.where(good, (fwd_high - ref) / atr, np.nan),
        "eval_mae_10b": np.where(good, (ref - fwd_low) / atr, np.nan),
    })
    out["eval_clean_10b"] = out["eval_mfe_10b"] - out["eval_mae_10b"]
    return out.dropna(subset=["eval_mfe_10b"])


def main() -> None:
    tickers = sorted(p.stem for p in BARS.glob("*.parquet"))
    if len(sys.argv) > 1:
        wanted = set(Path(sys.argv[1]).read_text().split())
        tickers = [t for t in tickers if t in wanted]
    frames, done = [], 0
    for t in tickers:
        f = one(t)
        if f is not None and len(f):
            frames.append(f)
        done += 1
        if done % 250 == 0:
            print(f"  {done}/{len(tickers)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out = out.set_index(["timestamp", "ticker"]).sort_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    print(f"wrote {OUT}  rows={len(out):,}  tickers={out.index.get_level_values(1).nunique()}")


if __name__ == "__main__":
    main()
