"""CLEAN option re-run: corrected costs + corrected filter + volatility screen.

Every prior option conclusion in this experiment was produced with at least one
now-known-wrong input. This re-runs the comparison with all three fixed:

 1. COST MODEL -- cents, not percent.
    G1 calibrated a 25.6% round-trip ratio on the live trades (median premium
    $0.94, ~8-cent half-spread) and it was then applied to contracts with a
    median premium of $5.60, implying a 72-cent half-spread (~9x reality).
    Here the spread is modeled in CENTS with a percentage floor, and every
    result is reported across a 5/10/15-cent sensitivity band.

 2. PARABOLIC FILTER -- percentage move, not ATR-normalised.
    The ATR label counted a 3.4% drift on a quiet stock as "parabolic" and the
    filter learned to select LOW-volatility names. Relabelled at >= +25% in 20
    bars, the filter selects volatile names and lifts share returns significantly.

 3. VOLATILITY SCREEN -- new, per user.
    Low-volatility names were never excluded from option routing. A long option
    on a 1%-ATR name cannot pay for its own premium; those names should be
    shares-only by construction, not by discovery.

Baseline is always SHARES on the identical trades.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LR = REPO / "research/options_experiment/data/let_it_run.parquet"
P3 = REPO / "research/options_experiment/data/phase3_counterfactual.parquet"
SHARES = REPO / "research/options_experiment/data/shares_filter_eval.parquet"
OUT = REPO / "research/options_experiment/data/clean_option_rerun.parquet"

COMMISSION = 0.65
CENTS = (0.05, 0.10, 0.15)      # half-spread sensitivity band
MIN_PCT_SPREAD = 0.01           # floor: never assume tighter than 1% of premium


def net_pnl(entry_px, exit_px, cents, n_contracts):
    """Round-trip P&L per position after a CENTS-based half-spread, floored at a
    small percentage so very cheap contracts are not treated as frictionless."""
    half_e = max(cents, MIN_PCT_SPREAD * entry_px)
    half_x = max(cents, MIN_PCT_SPREAD * exit_px)
    gross = (exit_px - entry_px) * 100.0 * n_contracts
    cost = (half_e + half_x) * 100.0 * n_contracts + 2 * COMMISSION * n_contracts
    return gross - cost


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atr-min", type=float, default=0.03,
                    help="minimum ATR as fraction of price for a name to be option-eligible")
    args = ap.parse_args()

    lr = pd.read_parquet(LR)
    lr = lr[lr.policy == "hold_to_expiry"].copy()
    p3 = pd.read_parquet(P3)
    meta = (p3[(p3.strategy == "long_call_atm") & (p3.sizing_mode == "matched_notional")]
            [["trade_id", "entry_ts", "entry_px_underlying", "exit_px_underlying",
              "atr_at_entry", "direction", "module"]].drop_duplicates("trade_id"))
    d = lr.drop(columns=[c for c in ("module", "score") if c in lr.columns]).merge(
        meta, on="trade_id", how="inner")
    d["entry_ts"] = pd.to_datetime(d.entry_ts, utc=True)
    d["atr_pct"] = d.atr_at_entry / d.entry_px_underlying
    d["n_contracts"] = 5000.0 / (d.entry_px_underlying * 100.0)
    # shares baseline on the identical trade
    d["share_ret"] = d.direction * (d.exit_px_underlying - d.entry_px_underlying) / d.entry_px_underlying
    d["share_pnl"] = d.share_ret * 5000.0
    d = d[np.isfinite(d.share_ret) & np.isfinite(d.atr_pct)]

    # attach the corrected %-based parabolic filter score where available
    if SHARES.exists():
        s = pd.read_parquet(SHARES)
        s["entry_ts"] = pd.to_datetime(s.entry_ts, utc=True)
        d = d.merge(s[["ticker", "entry_ts", "p"]].drop_duplicates(["ticker", "entry_ts"]),
                    on=["ticker", "entry_ts"], how="left")
    else:
        d["p"] = np.nan

    print(f"option-priced trades: {len(d)}   with filter score: {d.p.notna().sum()}")
    print(f"ATR% of price: p10 {100*d.atr_pct.quantile(.1):.1f}  median {100*d.atr_pct.median():.1f}  "
          f"p90 {100*d.atr_pct.quantile(.9):.1f}")

    for cents in CENTS:
        d[f"opt_{int(cents*100)}"] = [
            net_pnl(e, x, cents, n) for e, x, n in zip(d.entry_px, d.exit_px, d.n_contracts)]
    d.to_parquet(OUT, index=False)

    def block(sub: pd.DataFrame, label: str) -> None:
        if len(sub) < 30:
            print(f"\n{label}: n={len(sub)} -- too few, skipped")
            return
        cap = 5000.0 * len(sub)
        sh = 100 * sub.share_pnl.sum() / cap
        print(f"\n{label}   n={len(sub)}")
        print(f"  SHARES                     {sh:+7.2f}% on ${cap:,.0f}")
        for cents in CENTS:
            col = f"opt_{int(cents*100)}"
            optcap = (sub.entry_px * 100 * sub.n_contracts).sum()
            roc = 100 * sub[col].sum() / optcap
            print(f"  CALLS @ {int(cents*100):>2}c half-spread   {roc:+7.2f}% on ${optcap:,.0f} "
                  f"({100*optcap/cap:.0f}% of the share capital)")

    print("\n" + "=" * 72)
    print("A) EVERYTHING (what Phase 3 effectively tested)")
    block(d, "  all trades")

    print("\n" + "=" * 72)
    print(f"B) VOLATILITY SCREEN ONLY (ATR% >= {args.atr_min:.0%}) -- the missing filter")
    block(d[d.atr_pct >= args.atr_min], f"  ATR% >= {args.atr_min:.0%}")
    block(d[d.atr_pct < args.atr_min], f"  ATR% <  {args.atr_min:.0%}  (should be shares-only)")

    print("\n" + "=" * 72)
    print("C) VOLATILITY SCREEN + PARABOLIC FILTER (the intended strategy)")
    f = d[(d.atr_pct >= args.atr_min) & d.p.notna()]
    if len(f) >= 30:
        for k in (0.50, 0.30):
            block(f.nlargest(max(int(len(f) * k), 30), "p"), f"  top {k:.0%} by filter, ATR%>={args.atr_min:.0%}")
    else:
        print(f"  only {len(f)} scored+screened trades -- insufficient")


if __name__ == "__main__":
    main()
