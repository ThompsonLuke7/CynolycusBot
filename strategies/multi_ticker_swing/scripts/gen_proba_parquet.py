"""
Generate p_swing_probs.parquet in sweep_v2 long-format from a saved model.

Output schema: timestamp (col), ticker (col), split (col: train/val/test),
               p_long, p_short, p_neutral  (raw softmax probs)

Usage:
  python multi_ticker_swing/scripts/gen_proba_parquet.py \
      --model models/swing_xgb_model.json \
      --features models/selected_features_1500.txt   # optional, else uses FEATURE_COLUMNS \
      --out    models/p_swing_probs.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from strategies.multi_ticker_swing.config.pipeline_config import (
    FEATURE_COLUMNS, MODELS_DIR, TRAIN_FRAC, VAL_FRAC,
)
from strategies.multi_ticker_swing.data.load_data import load_training_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _time_split(df: pd.DataFrame, ts_col: str):
    df = df.sort_values(ts_col).reset_index(drop=True)
    n  = len(df)
    t1 = int(n * TRAIN_FRAC)
    t2 = int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


def _find_ts_col(df: pd.DataFrame) -> str:
    for name in ("timestamp", "index", "t", "time", "date"):
        if name in df.columns:
            return name
    return df.columns[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True, help="Path to XGBoost .json model")
    p.add_argument("--features", default=None,  help="Newline-delimited feature list .txt")
    p.add_argument("--out",      required=True, help="Output parquet path")
    args = p.parse_args()

    from xgboost import XGBClassifier
    clf = XGBClassifier()
    clf.load_model(args.model)
    logger.info("Loaded model from %s", args.model)

    if args.features:
        feat_path = Path(args.features)
        feature_columns = [l.strip() for l in feat_path.read_text().splitlines() if l.strip()]
        logger.info("Using %d features from %s", len(feature_columns), feat_path)
    else:
        feature_columns = FEATURE_COLUMNS
        logger.info("Using %d features from FEATURE_COLUMNS", len(feature_columns))

    matrix = load_training_matrix()
    ts_col = _find_ts_col(matrix)
    logger.info("Matrix: %d rows, %d cols", *matrix.shape)

    train_df, val_df, test_df = _time_split(matrix, ts_col)

    frames = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        avail = [c for c in feature_columns if c in split_df.columns]
        X = split_df[avail].values.astype(np.float32)
        proba = clf.predict_proba(X)   # (N, 3): [P(short), P(neutral), P(long)]

        out = split_df[["timestamp", "ticker"]].copy() if "ticker" in split_df.columns \
              else split_df[["timestamp"]].copy()
        out["split"]     = split_name
        out["p_long"]    = proba[:, 2]
        out["p_short"]   = proba[:, 0]
        out["p_neutral"] = proba[:, 1]
        frames.append(out)
        logger.info("  %s: %d rows, avg P(long)=%.3f P(short)=%.3f",
                    split_name, len(out), proba[:, 2].mean(), proba[:, 0].mean())

    proba_df = pd.concat(frames, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proba_df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows → %s", len(proba_df), out_path)


if __name__ == "__main__":
    main()
