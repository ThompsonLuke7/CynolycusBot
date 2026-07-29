"""Compute the TRUE parabolic label: forward maximum favorable excursion (MFE).

Why this exists
---------------
`realized_move_atr` derived from the spine is NOT a measure of how far a stock ran.
It is quantized by each module's exit rule -- verified:
  momentum_expansion: 2,860 trades at exactly +2.0 ATR (tp), 865 at exactly -4.0 (sl)
  multi_ticker_swing_htf: 12,969 at exactly -2.0 (sl), 5,411 at exactly +5.0 (tp)

So a stock that squeezed +12 ATR is recorded as +2.0 ATR, because the module took
profit at 2 ATR. Any "parabolic" analysis built on that column is really measuring
"did the take-profit fire", which is a different question and structurally cannot
see the tail that long options are good at capturing.

This computes, from the underlying bars alone and independent of the module's exit:
  mfe_atr_{H}  : max favorable excursion within H bars of the decision, in ATR units
  mae_atr_{H}  : max adverse excursion within H bars, in ATR units
  parabolic_{H}: mfe_atr_{H} >= threshold

Leakage: these are FORWARD-looking by construction and are LABELS ONLY. They must
never be used as features. The decision timestamp is the anchor; the window is
strictly after it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPINE = REPO / "research/options_experiment/data/signal_spine.parquet"
BARS = REPO / "Data/shared/bars/4h"
OUT = REPO / "research/options_experiment/data/forward_excursion.parquet"

HORIZONS = (10, 20, 40, 60)  # 4H bars: ~1.5, 3, 6, 9 trading weeks
MODULES = ("momentum_expansion", "multi_ticker_swing_htf")


def _load_bars(ticker: str) -> pd.DataFrame | None:
    p = BARS / f"{ticker}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p)
    if "timestamp" not in b.columns:
        b = b.reset_index()
    ts = "timestamp" if "timestamp" in b.columns else b.columns[0]
    b[ts] = pd.to_datetime(b[ts], utc=True)
    b = b.rename(columns={ts: "timestamp"}).sort_values("timestamp")
    need = {"high", "low", "close"}
    if not need.issubset(b.columns):
        return None
    return b[["timestamp", "high", "low", "close"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parabolic-atr", type=float, default=4.0)
    args = ap.parse_args()

    s = pd.read_parquet(SPINE)
    s = s[s.module.isin(MODULES)].copy()
    s["decision_ts"] = pd.to_datetime(s.signal_ts.fillna(s.entry_ts), utc=True)
    s = s[s.atr_at_entry.notna() & (s.atr_at_entry > 0)]

    rows = []
    missing = 0
    for ticker, grp in s.groupby("ticker"):
        bars = _load_bars(ticker)
        if bars is None or bars.empty:
            missing += len(grp)
            continue
        bt = bars.timestamp.values
        hi = bars.high.to_numpy()
        lo = bars.low.to_numpy()
        for r in grp.itertuples(index=False):
            # first bar STRICTLY after the decision -- no same-bar peeking
            i = int(np.searchsorted(bt, np.datetime64(r.decision_ts), side="right"))
            if i >= len(bt):
                continue
            rec = {
                "module": r.module, "ticker": ticker, "decision_ts": r.decision_ts,
                "entry_ts": r.entry_ts, "direction": r.direction,
                "entry_px": r.entry_px_underlying, "atr": r.atr_at_entry,
                "score": r.score, "exit_reason": r.exit_reason,
            }
            for H in HORIZONS:
                j = min(i + H, len(bt))
                if j <= i:
                    rec[f"mfe_atr_{H}"] = np.nan
                    rec[f"mae_atr_{H}"] = np.nan
                    continue
                if r.direction > 0:
                    fav = (hi[i:j].max() - r.entry_px_underlying) / r.atr_at_entry
                    adv = (lo[i:j].min() - r.entry_px_underlying) / r.atr_at_entry
                else:
                    fav = (r.entry_px_underlying - lo[i:j].min()) / r.atr_at_entry
                    adv = (r.entry_px_underlying - hi[i:j].max()) / r.atr_at_entry
                rec[f"mfe_atr_{H}"] = fav
                rec[f"mae_atr_{H}"] = adv
            rows.append(rec)

    out = pd.DataFrame(rows)
    for H in HORIZONS:
        out[f"parabolic_{H}"] = (out[f"mfe_atr_{H}"] >= args.parabolic_atr).astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    print(f"wrote {len(out)} rows -> {OUT}   (skipped {missing} trades: no 4H bars)")
    print(f"\nTRUE forward excursion (>= {args.parabolic_atr} ATR favorable), by horizon:")
    agg = out.groupby("module").agg(**{
        "n": ("ticker", "size"),
        **{f"parab_{H}": (f"parabolic_{H}", "mean") for H in HORIZONS},
        **{f"medMFE_{H}": (f"mfe_atr_{H}", "median") for H in HORIZONS},
    })
    print(agg.round(3).to_string())
    print("\nCompare: exit-rule-capped 'move' vs true MFE (20 bars)")
    for m, g in out.groupby("module"):
        print(f"  {m}: median true MFE_20 = {g.mfe_atr_20.median():.2f} ATR, "
              f"p90 = {g.mfe_atr_20.quantile(.90):.2f}, max = {g.mfe_atr_20.max():.2f}")


if __name__ == "__main__":
    main()
