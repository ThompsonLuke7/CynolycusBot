"""
Rebuild the merged 192-feature *test-window* matrix for a Colab competition bundle.

The competition models (see data/training_export/colab_competition.py) were trained on a
192-feature frame produced by merging two context parquets onto the base 30m matrix:
  - bar_location_context_features.parquet  ->  joined on [timestamp, ticker]
  - daily_context_features.parquet         ->  joined as-of the prior day on [ticker, date]

This merge happened in Colab and was never materialised locally, so the local backtest
(simulate.py / FEATURE_COLUMNS) cannot reproduce the competition's feature set. This script
rebuilds it for the final time-split (the competition test set) only, which keeps it within
this machine's RAM. The split boundary mirrors colab_competition.time_split: sort by
timestamp, drop null targets, take the last (1 - train_frac - val_frac) fraction of rows.

Usage:
  python -m strategies.multi_ticker_swing.backtest.build_competition_test_matrix \
      --bundle-dir strategies/multi_ticker_swing/models/competition_20260615 \
      --out strategies/multi_ticker_swing/backtest/competition_20260615_test_matrix.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "training_export"


def _norm_ticker(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace("$", "", regex=False)


def compute_test_cutoff(base_path: Path, target: str, train_frac: float, val_frac: float) -> pd.Timestamp:
    t = pq.read_table(base_path, columns=["timestamp", target]).to_pandas()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    t = t.sort_values("timestamp").reset_index(drop=True).dropna(subset=[target])
    t2 = int(len(t) * (train_frac + val_frac))
    return t["timestamp"].iloc[t2]


def build(bundle_dir: Path, out_path: Path) -> None:
    manifest = json.loads((EXPORT_DIR / "feature_manifest.json").read_text())
    features: list[str] = manifest["feature_columns"]
    target: str = manifest.get("target_column", "target")
    weight_col: str = manifest.get("sample_weight_column", "sample_weight")
    train_frac = float(manifest.get("train_frac", 0.70))
    val_frac = float(manifest.get("val_frac", 0.15))

    base_path = EXPORT_DIR / manifest.get("matrix_file", "training_matrix_30m.parquet")
    bar_path = EXPORT_DIR / manifest["bar_context_file"]
    ctx_path = EXPORT_DIR / manifest["context_file"]

    cut = compute_test_cutoff(base_path, target, train_frac, val_frac)
    logger.info("Test cutoff timestamp: %s", cut)

    # --- base matrix, test window only ---
    filt = [("timestamp", ">=", cut.to_pydatetime())]
    base_cols = ["timestamp", "ticker", target]
    if weight_col not in base_cols:
        base_cols.append(weight_col)
    base_feats = [c for c in features if c in pq.read_schema(base_path).names]
    df = pq.read_table(base_path, columns=base_cols + base_feats, filters=filt).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ticker"] = _norm_ticker(df["ticker"])
    logger.info("Base test slice: %s rows, %s tickers", len(df), df["ticker"].nunique())

    # --- bar-location context: exact join on [timestamp, ticker] ---
    bar_feats = [c for c in features if c in pq.read_schema(bar_path).names and c not in df.columns]
    bar = pq.read_table(bar_path, columns=["timestamp", "ticker"] + bar_feats, filters=filt).to_pandas()
    bar["timestamp"] = pd.to_datetime(bar["timestamp"], utc=True)
    bar["ticker"] = _norm_ticker(bar["ticker"])
    df = df.merge(bar, on=["timestamp", "ticker"], how="left")
    del bar
    logger.info("After bar-location merge: %s cols", df.shape[1])

    # --- daily context: as-of prior-day join on [ticker, date] ---
    ctx = pd.read_parquet(ctx_path)
    ctx["date"] = pd.to_datetime(ctx["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    ctx["ticker"] = _norm_ticker(ctx["ticker"])
    ctx_feats = [c for c in features if c in ctx.columns and c not in df.columns]
    df["_context_date"] = df["timestamp"].dt.tz_convert(None).dt.normalize()
    df = df.merge(
        ctx[["ticker", "date"] + ctx_feats].rename(columns={"date": "_context_date"}),
        on=["ticker", "_context_date"], how="left",
    ).drop(columns=["_context_date"])
    del ctx
    logger.info("After daily-context merge: %s cols", df.shape[1])

    # --- normalise features to numeric float32 (mirrors colab normalize_features) ---
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature columns after merge: {missing[:20]}")
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)

    keep = ["timestamp", "ticker", target] + ([weight_col] if weight_col in df.columns else []) + features
    df = df[keep].sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s  (%s rows x %s cols)", out_path, len(df), df.shape[1])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    build(args.bundle_dir, args.out)
