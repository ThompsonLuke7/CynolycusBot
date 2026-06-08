from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multi_ticker_swing.config.pipeline_config import MODELS_DIR, MODEL_PATH, PROCESSED_30M_DIR
from multi_ticker_swing.data.fetch_data import universe_tickers


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _find_ts_col(df: pd.DataFrame) -> str:
    for name in ("timestamp", "index", "t", "time", "date"):
        if name in df.columns:
            return name
    return df.columns[0]


def _feature_list(path: Path | None) -> list[str]:
    if path is not None:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    default = MODELS_DIR / "selected_features.txt"
    return [line.strip() for line in default.read_text().splitlines() if line.strip()]


def _split_cutoffs(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not path.exists():
        return pd.Timestamp("2024-11-06 18:30:00Z"), pd.Timestamp("2025-08-07 18:30:00Z")
    df = pd.read_parquet(path, columns=["timestamp", "split"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    train = df.loc[df["split"].eq("train"), "timestamp"].max()
    val = df.loc[df["split"].eq("val"), "timestamp"].max()
    if pd.isna(train) or pd.isna(val):
        return pd.Timestamp("2024-11-06 18:30:00Z"), pd.Timestamp("2025-08-07 18:30:00Z")
    return pd.Timestamp(train), pd.Timestamp(val)


def _assign_split(ts: pd.Series, train_max: pd.Timestamp, val_max: pd.Timestamp) -> pd.Series:
    out = pd.Series("test", index=ts.index, dtype="object")
    out.loc[ts <= train_max] = "train"
    out.loc[(ts > train_max) & (ts <= val_max)] = "val"
    return out


def _backup(path: Path, backup_dir: Path) -> Path | None:
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / path.name
    shutil.copy2(path, dest)
    return dest


def _score_ticker(
    *,
    ticker: str,
    feature_path: Path,
    clf,
    feature_columns: list[str],
    train_max: pd.Timestamp,
    val_max: pd.Timestamp,
) -> pd.DataFrame | None:
    df = pd.read_parquet(feature_path).reset_index()
    ts_col = _find_ts_col(df)
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        logger.warning("[%s] missing %d model features; skipped", ticker, len(missing))
        return None

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    X_df = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    ok = ts.notna() & X_df.notna().all(axis=1)
    if not ok.any():
        logger.warning("[%s] no valid rows after NaN filtering", ticker)
        return None

    proba = clf.predict_proba(X_df.loc[ok].to_numpy(dtype=np.float32))
    out = pd.DataFrame(
        {
            "timestamp": ts.loc[ok].to_numpy(),
            "ticker": ticker,
            "split": _assign_split(ts.loc[ok], train_max, val_max).to_numpy(),
            "p_long": proba[:, 2].astype(np.float32),
            "p_short": proba[:, 0].astype(np.float32),
            "p_neutral": proba[:, 1].astype(np.float32),
        }
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Score refreshed multi-ticker swing feature files with the current model.")
    parser.add_argument("--universe", default="Data/shared/universe/shared_universe.csv")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_30M_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--features", type=Path, default=MODELS_DIR / "selected_features.txt")
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "p_swing_probs.parquet")
    parser.add_argument("--split-source", type=Path, default=MODELS_DIR / "p_swing_probs.parquet")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from xgboost import XGBClassifier

    feature_columns = _feature_list(args.features)
    train_max, val_max = _split_cutoffs(args.split_source)
    logger.info("Using split cutoffs train<=%s val<=%s", train_max, val_max)

    clf = XGBClassifier()
    clf.load_model(str(args.model))
    logger.info("Loaded model %s with %d features", args.model, len(feature_columns))

    backup_dir = args.backup_dir
    if backup_dir is None:
        backup_dir = MODELS_DIR / "archive" / pd.Timestamp.utcnow().strftime("pre_rescore_%Y%m%dT%H%M%SZ")
    backup = _backup(args.out, backup_dir)
    if backup is not None:
        logger.info("Backed up existing proba parquet to %s", backup)

    tickers = universe_tickers(args.universe)
    if args.limit > 0:
        tickers = tickers[: args.limit]
    writer: pq.ParquetWriter | None = None
    rows = 0
    scored = 0
    try:
        for i, ticker in enumerate(tickers, 1):
            path = args.processed_dir / f"{ticker}_features.parquet"
            if not path.exists():
                logger.warning("[%s] feature file missing", ticker)
                continue
            out = _score_ticker(
                ticker=ticker,
                feature_path=path,
                clf=clf,
                feature_columns=feature_columns,
                train_max=train_max,
                val_max=val_max,
            )
            if out is None or out.empty:
                continue
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(args.out, table.schema, compression="snappy")
            writer.write_table(table)
            rows += len(out)
            scored += 1
            if scored % 50 == 0:
                logger.info("scored=%d/%d rows=%d latest=%s", scored, len(tickers), rows, ticker)
    finally:
        if writer is not None:
            writer.close()

    if rows == 0:
        logger.error("No probability rows written.")
        return 1
    logger.info("Saved %d probability rows for %d tickers -> %s", rows, scored, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
