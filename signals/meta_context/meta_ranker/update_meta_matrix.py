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
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from strategies.momentum_expansion.config.momentum_config import CONTEXT_TICKERS, SECTOR_ETFS
from strategies.momentum_expansion.features.live_feature_panel_4h import (
    assert_manifest_coverage,
    build_live_feature_panel_4h,
)
from strategies.momentum_expansion.inference.ranker import ExpansionRanker
from strategies.multi_ticker_swing_htf.inference.scorer import HTFSwingScorer
from signals.meta_context import build_meta_ranker_matrix as B

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BARS_4H = REPO / "Data/shared/bars/4h"
BARS_1D = REPO / "Data/shared/bars/1d"


def _atomic_to_parquet(df: pd.DataFrame, out_path, **kwargs) -> None:
    """Write via temp file + atomic rename so a concurrent reader (the 4H loops)
    or a second writer (SharedDataRefresher / nightly job overlap) never sees a
    torn/partial parquet file."""
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, **kwargs)
    tmp.replace(out_path)
MATRIX = HERE / "meta_ranker_matrix.parquet"
UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"
ROLL_DAYS = 400  # rolling window kept in the matrix (long-lookback features need ~1yr)
REFERENCE_BAR_TICKERS = ("SPY", "QQQ")


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


def _latest_reference_bar_timestamp() -> pd.Timestamp | None:
    latest: list[pd.Timestamp] = []
    for ticker in REFERENCE_BAR_TICKERS:
        bars = _read_bars(BARS_4H / f"{ticker}.parquet")
        if bars is not None and not bars.empty:
            latest.append(pd.to_datetime(bars["timestamp"], utc=True).max())
    return max(latest) if latest else None


def _load_4h_for_panel(ticker: str) -> pd.DataFrame:
    b = _read_bars(BARS_4H / f"{ticker}.parquet")
    if b is None or b.empty:
        raise FileNotFoundError(ticker)
    return b.set_index("timestamp")


def _load_1d_for_panel(ticker: str) -> pd.DataFrame | None:
    b = _read_bars(BARS_1D / f"{ticker}.parquet")
    return None if b is None else b.set_index("timestamp")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--matrix", default=str(MATRIX))
    ap.add_argument("--roll-days", type=int, default=ROLL_DAYS)
    args = ap.parse_args()

    existing = pd.read_parquet(args.matrix).reset_index()
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    max_ts = existing["timestamp"].max()
    print(f"existing matrix: {len(existing):,} rows, max ts {max_ts}")

    latest_reference_ts = _latest_reference_bar_timestamp()
    if latest_reference_ts is not None and max_ts >= latest_reference_ts:
        print(
            "no new reference-market 4H bar to add "
            f"(matrix={max_ts}, latest reference={latest_reference_ts})."
        )
        return

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = sorted(pd.read_csv(UNIVERSE)["ticker"].dropna().astype(str).str.upper().unique())
    ctx_4h = _load_context()
    logger.info("building live features for %d tickers since %s ...", len(tickers), max_ts)
    result = build_live_feature_panel_4h(
        tickers=tickers, load_4h_bars=_load_4h_for_panel,
        load_1d_bars=_load_1d_for_panel, ctx_4h=ctx_4h, since=max_ts,
    )
    if result.panel.empty:
        print("no new bars to add — matrix already current.")
        return
    feat_idx = result.panel
    feats = feat_idx.reset_index()
    print(f"new feature rows: {len(feats):,}  bars: {feats['timestamp'].nunique()}  "
          f"range {feats['timestamp'].min()} -> {feats['timestamp'].max()}")

    # ---- base scores from DEPLOYED models (not OOF) ----
    # feat_idx already carries every xsec_*/earnings column (build_live_feature_panel_4h
    # bundles both post-processing steps unconditionally -- see its docstring and
    # LIVING_SUMMARY.md 2026-07-19 for the audit that found both were previously
    # missing live, silently, at this exact call site).
    mom_ranker = ExpansionRanker()
    htf_scorer = HTFSwingScorer()
    assert_manifest_coverage(scorer=mom_ranker, panel=feat_idx, label="update_meta_matrix/momentum")
    assert_manifest_coverage(scorer=htf_scorer, panel=feat_idx, label="update_meta_matrix/htf")
    mom = mom_ranker.score(feat_idx)
    htf = htf_scorer.score(feat_idx)
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
    _atomic_to_parquet(combined, args.matrix)
    print(f"\nwrote {len(combined):,} rows (rolling {args.roll_days}d) -> {args.matrix}")
    print(f"  new max ts: {combined.index.get_level_values('timestamp').max()}")


if __name__ == "__main__":
    main()
