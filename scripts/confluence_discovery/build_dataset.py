"""Build the point-in-time confluence-discovery dataset (research only).

Joins, on the meta ranker matrix spine (timestamp, ticker):
  1. meta_ranker_matrix.parquet     — walk-forward OOF mom/htf scores, theme state,
                                      news catalysts, calendars, treasuries, regime,
                                      forward-outcome labels (already PIT-joined by
                                      signals/meta_context/build_meta_ranker_matrix.py).
  2. features_4h.parquet            — causal technical-state columns, exact
                                      (timestamp, ticker) join (features are computed
                                      from bars up to and including the decision bar).
  3. finra_short_volume.parquet     — daily short-sale volume aggregates, converted to
                                      short_ratio / z-score / percentile features and
                                      joined STRICTLY prior-day (merge_asof,
                                      allow_exact_matches=False) so a bar on date D only
                                      sees FINRA data from date <= D-1.

Excluded as features (leakage): trend_persistence (forward label),
earnings_in_fwd_window (defined over the label window). Forward columns are kept
only as outcomes.

Output: research/confluence/confluence_dataset.parquet

Usage:
  .venv/bin/python scripts/confluence_discovery/build_dataset.py [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

META_MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
FEATURES_4H = REPO / "strategies/momentum_expansion/data/processed/features_4h.parquet"
FINRA_SHORT = REPO / "signals/news/data/processed/finra_short_volume.parquet"
OUT_DEFAULT = REPO / "research/confluence/confluence_dataset.parquet"

# Causal technical-state columns pulled from the 4H feature matrix.
TECH_COLS = [
    "near_52w_high", "daily_new_high_252", "breakout_20", "bars_since_breakout_20",
    "is_compressed_5_20", "compression_count_20", "rvol_20", "volume_spike_20",
    "atr_expand_14_60", "range_pos_20", "dist_20bar_high_atr", "low_price_flag",
    "xsec_ret_20_rank", "xsec_rvol_20_rank", "xsec_near_high_rank",
]

FINRA_Z_WINDOW = 20      # trading days for short-ratio z-score
FINRA_PCT_WINDOW = 63    # trading days for short-ratio percentile rank
FINRA_MIN_TOTAL_VOL = 10_000  # ignore near-zero-volume days when forming ratios


def build_finra_features() -> pd.DataFrame:
    """Per (ticker, date) short-flow features from FINRA daily short volume."""
    f = pd.read_parquet(FINRA_SHORT, columns=["date", "ticker", "short_volume", "total_volume"])
    f = f[f["total_volume"] >= FINRA_MIN_TOTAL_VOL].copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values(["ticker", "date"]).reset_index(drop=True)
    f["short_ratio"] = f["short_volume"] / f["total_volume"]

    g = f.groupby("ticker", sort=False)
    roll_mean = g["short_ratio"].transform(lambda s: s.rolling(FINRA_Z_WINDOW, min_periods=10).mean())
    roll_std = g["short_ratio"].transform(lambda s: s.rolling(FINRA_Z_WINDOW, min_periods=10).std())
    f["short_ratio_z20"] = (f["short_ratio"] - roll_mean) / roll_std.replace(0.0, np.nan)
    f["short_ratio_pct63"] = g["short_ratio"].transform(
        lambda s: s.rolling(FINRA_PCT_WINDOW, min_periods=21).rank(pct=True)
    )
    svol_mean = g["short_volume"].transform(lambda s: s.rolling(FINRA_Z_WINDOW, min_periods=10).mean())
    f["short_vol_surge20"] = f["short_volume"] / svol_mean.replace(0.0, np.nan)
    return f[["ticker", "date", "short_ratio", "short_ratio_z20", "short_ratio_pct63", "short_vol_surge20"]]


def asof_prior_day_ticker(spine: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """merge_asof on 'date' within ticker, strictly prior day (no lookahead).

    Same discipline as build_meta_ranker_matrix._asof_prior_day_ticker.
    """
    s = spine.sort_values("date").reset_index()
    r = right.sort_values("date")
    merged = pd.merge_asof(s, r, on="date", by="ticker", direction="backward", allow_exact_matches=False)
    return merged.set_index("index").sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    print(f"loading meta matrix: {META_MATRIX}")
    meta = pd.read_parquet(META_MATRIX).reset_index()
    meta["timestamp"] = pd.to_datetime(meta["timestamp"], utc=True)
    # Drop known-forward columns from the FEATURE side; keep outcomes explicitly.
    meta = meta.drop(columns=["earnings_in_fwd_window"], errors="ignore")
    print(f"  rows={len(meta):,} tickers={meta['ticker'].nunique()} "
          f"range={meta['timestamp'].min()} .. {meta['timestamp'].max()}")

    print(f"joining technical states from: {FEATURES_4H}")
    tech = pd.read_parquet(FEATURES_4H, columns=["timestamp", "ticker"] + TECH_COLS)
    if "timestamp" not in tech.columns:  # stored as (timestamp, ticker) index
        tech = tech.reset_index()
    tech["timestamp"] = pd.to_datetime(tech["timestamp"], utc=True)
    df = meta.merge(tech, on=["timestamp", "ticker"], how="left", validate="1:1")
    hit = df[TECH_COLS[0]].notna().mean()
    print(f"  technical join coverage: {hit:.3f}")

    print(f"building FINRA short-flow features from: {FINRA_SHORT}")
    finra = build_finra_features()
    print(f"  finra rows={len(finra):,} range={finra['date'].min().date()} .. {finra['date'].max().date()}")
    df["date"] = pd.to_datetime(df["date"])
    df = asof_prior_day_ticker(df, finra)
    print(f"  finra join coverage: {df['short_ratio'].notna().mean():.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = df.set_index(["timestamp", "ticker"]).sort_index()
    df.to_parquet(args.out)
    print(f"wrote {args.out}  rows={len(df):,} cols={df.shape[1]}")


if __name__ == "__main__":
    main()
