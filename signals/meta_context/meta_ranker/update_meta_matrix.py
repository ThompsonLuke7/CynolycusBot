"""
Incremental Meta Ranker matrix updater (live).

Instead of rebuilding the full 2022-> matrix, this scores ONLY the new 4H bars with
the *deployed* base models (momentum ExpansionRanker + HTF scorer) -- not OOF -- then
joins the same context feeds and APPENDS to a rolling ~1yr matrix window. This is the
live analogue of build_meta_ranker_matrix.py and is what the 4H loop calls.

Reuses skew-free training code: build_ticker_features_4h (feature construction), the
deployed base scorers, and build_meta_ranker_matrix's join helpers/constants.

  PYTHONPATH=. python signals/meta_context/meta_ranker/update_meta_matrix.py
  PYTHONPATH=. python signals/meta_context/meta_ranker/update_meta_matrix.py --tickers NVDA MU INTC  # quick test

Output: overwrites meta_ranker_matrix.parquet (rolling window) with new bars appended.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from strategies.momentum_expansion.config.momentum_config import CONTEXT_TICKERS, SECTOR_ETFS
from strategies.momentum_expansion.features.feature_matrix_4h import (
    FEATURE_COLUMNS_4H,
    build_ticker_features_4h,
)
from strategies.momentum_expansion.inference.ranker import ExpansionRanker
from strategies.multi_ticker_swing_htf.inference.scorer import HTFSwingScorer
from signals.meta_context import build_meta_ranker_matrix as B

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BARS_4H = REPO / "Data/shared/bars/4h"
BARS_1D = REPO / "Data/shared/bars/1d"
MATRIX = HERE / "meta_ranker_matrix.parquet"
UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"
ROLL_DAYS = 400  # rolling window kept in the matrix (long-lookback features need ~1yr)


def _read_bars(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    b = pd.read_parquet(path)
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    return b.drop_duplicates("timestamp").sort_values("timestamp")


def _load_context() -> dict[str, pd.DataFrame]:
    ctx = {}
    for sym in list(CONTEXT_TICKERS) + list(SECTOR_ETFS):
        b = _read_bars(BARS_4H / f"{sym}.parquet")
        if b is not None:
            ctx[sym] = b.set_index("timestamp")
    return ctx


def _build_live_features(tickers: list[str], ctx_4h: dict, since: pd.Timestamp) -> pd.DataFrame:
    """Per-ticker feature build (skew-free); keep bars strictly after `since`."""
    rows = []
    for i, t in enumerate(tickers, 1):
        df_4h = _read_bars(BARS_4H / f"{t}.parquet")
        if df_4h is None or df_4h.empty:
            continue
        df_1d = _read_bars(BARS_1D / f"{t}.parquet")
        try:
            feats = build_ticker_features_4h(ticker=t, df_4h=df_4h.set_index("timestamp"),
                                             df_1d=None if df_1d is None else df_1d.set_index("timestamp"),
                                             ctx_4h=ctx_4h)
        except Exception:
            continue
        if feats is None or feats.empty:
            continue
        feats = feats.reset_index().rename(columns={"index": "timestamp"})
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
        feats = feats[feats["timestamp"] > since]
        if feats.empty:
            continue
        feats["ticker"] = t
        rows.append(feats)
        if i % 200 == 0:
            print(f"  features {i}/{len(tickers)}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--matrix", default=str(MATRIX))
    ap.add_argument("--roll-days", type=int, default=ROLL_DAYS)
    args = ap.parse_args()

    existing = pd.read_parquet(args.matrix).reset_index()
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    max_ts = existing["timestamp"].max()
    print(f"existing matrix: {len(existing):,} rows, max ts {max_ts}")

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = sorted(pd.read_csv(UNIVERSE)["ticker"].dropna().astype(str).str.upper().unique())
    ctx_4h = _load_context()
    print(f"building live features for {len(tickers)} tickers since {max_ts} ...")
    feats = _build_live_features(tickers, ctx_4h, since=max_ts)
    if feats.empty:
        print("no new bars to add — matrix already current.")
        return
    print(f"new feature rows: {len(feats):,}  bars: {feats['timestamp'].nunique()}  "
          f"range {feats['timestamp'].min()} -> {feats['timestamp'].max()}")

    # ---- base scores from DEPLOYED models (not OOF) ----
    feat_idx = feats.set_index(["timestamp", "ticker"])
    mom = ExpansionRanker().score(feat_idx[[c for c in FEATURE_COLUMNS_4H if c in feat_idx.columns]])
    htf = HTFSwingScorer().score(feat_idx)
    spine = pd.DataFrame({"mom_score": mom.values, "htf_score": htf.values}, index=feat_idx.index).reset_index()
    spine["date"] = B._norm_date(spine["timestamp"])
    # forward-label columns are unknown for live bars
    for c in ["fwd_close_return", "fwd_max_drawdown", "fwd_atr_adj_return", "trend_persistence",
              "fwd_max_return", "fwd_max_alpha", "meta_good", "meta_upside"]:
        spine[c] = np.nan
    # ticker meta + regime come straight from the feature frame (point-in-time)
    meta_present = [c for c in B.META_COLS if c in feats.columns]
    spine = spine.merge(feats[["timestamp", "ticker"] + meta_present], on=["timestamp", "ticker"], how="left")

    # ---- theme context (as-of prior day) ----
    dtf_path = B.DYNAMIC_THEME_FEATURES_HISTORY if B.DYNAMIC_THEME_FEATURES_HISTORY.exists() else B.DYNAMIC_THEME_FEATURES
    if dtf_path.exists():
        dtf = pd.read_parquet(dtf_path)
        dtf["date"] = pd.to_datetime(dtf["date"]).dt.normalize()
        dtf = dtf.rename(columns={"primary_theme": "theme"}).sort_values(["ticker", "date"])
        ctx_cols = [c for c in B.DYNAMIC_THEME_CTX if c in dtf.columns and c != "primary_theme"]
        spine = B._asof_prior_day_ticker(spine, dtf[["ticker", "date", "theme"] + ctx_cols])
    else:
        spine["theme"] = np.nan

    # ---- news catalyst (as-of prior day) ----
    if B.NEWS_CATALYST_SIGNAL.exists():
        ncs = pd.read_parquet(B.NEWS_CATALYST_SIGNAL)
        ncs["date"] = pd.to_datetime(ncs["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
        avail = [c for c in B.NEWS_CATALYST_COLS if c in ncs.columns]
        spine = B._asof_prior_day_ticker(spine, ncs[["ticker", "date"] + avail].sort_values(["ticker", "date"]))

    # ---- calendar / macro / treasury ----
    spine, _ = B._join_calendar_macro_features(spine)

    # ---- forward guidance (as-of prior day) ----
    if B.FORWARD_GUIDANCE_SIGNAL.exists():
        fg = pd.read_parquet(B.FORWARD_GUIDANCE_SIGNAL)
        fg["date"] = pd.to_datetime(fg["date"]).dt.normalize()
        fg_cols = [c for c in ["fg_guidance_score", "fg_guidance_count_90d", "fg_days_since_guidance"] if c in fg.columns]
        spine = B._asof_prior_day_ticker(spine, fg[["ticker", "date"] + fg_cols].sort_values(["ticker", "date"]))

    # ---- cross-sectional context (per bar) ----
    g = spine.groupby("timestamp")
    spine["mom_xs_rank"] = g["mom_score"].rank(pct=True)
    spine["htf_xs_rank"] = g["htf_score"].rank(pct=True)
    spine["signal_agreement"] = spine["mom_xs_rank"] * spine["htf_xs_rank"]
    gt = spine.groupby(["timestamp", "theme"])
    spine["within_theme_mom_rank"] = gt["mom_score"].rank(pct=True)
    spine["_hot"] = (spine["mom_xs_rank"] > 0.8).astype(float)
    spine["theme_crowding_frac"] = gt["_hot"].transform("mean")
    spine = spine.drop(columns=["_hot"])

    # ---- append + roll ----
    combined = pd.concat([existing, spine], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.drop_duplicates(subset=["timestamp", "ticker"], keep="last")
    cutoff = combined["timestamp"].max() - pd.Timedelta(days=args.roll_days)
    combined = combined[combined["timestamp"] >= cutoff].sort_values(["timestamp", "ticker"])
    combined = combined.set_index(["timestamp", "ticker"])
    combined.to_parquet(args.matrix)
    print(f"\nwrote {len(combined):,} rows (rolling {args.roll_days}d) -> {args.matrix}")
    print(f"  new max ts: {combined.index.get_level_values('timestamp').max()}")


if __name__ == "__main__":
    main()
