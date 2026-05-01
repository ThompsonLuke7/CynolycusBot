from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.ga_xgboost.train import _load_split_indices


DATASET = "10min"
X_FILENAME = "X_10min_tree.parquet"
DATASET_DIR = Path("Data/processed/spy/datasets/10min")
EXECUTION_1M = Path("Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet")
MODEL_ROOT = Path("Data/models/ga_xgboost/10min/single")


def _build_plot_frame() -> None:
    out_path = DATASET_DIR / "plot_frame.parquet"
    if out_path.exists():
        print(f"[prepare] plot frame exists: {out_path}")
        return
    y = pd.read_parquet(DATASET_DIR / "y.parquet")
    one = pd.read_parquet(EXECUTION_1M, columns=["timestamp", "open", "high", "low", "close", "volume"])
    idx = pd.to_datetime(one["timestamp"], utc=True, errors="coerce").dt.tz_convert(y.index.tz)
    one = one.assign(timestamp=idx).dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    bars = (
        one.resample("10min", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .reindex(y.index)
    )
    bars["close"] = bars["close"].ffill()
    bars["open"] = bars["open"].fillna(bars["close"])
    bars["high"] = bars["high"].fillna(bars[["open", "close"]].max(axis=1))
    bars["low"] = bars["low"].fillna(bars[["open", "close"]].min(axis=1))
    bars["volume"] = bars["volume"].fillna(0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out_path)
    print(f"[prepare] wrote plot frame: {out_path} rows={len(bars)}")


def _empty_probs(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "p_short_oof_train": np.nan,
            "p_neutral_oof_train": np.nan,
            "p_long_oof_train": np.nan,
            "p_short_test": np.nan,
            "p_neutral_test": np.nan,
            "p_long_test": np.nan,
        },
        index=index,
    )


def _write_probs(label_dir: str) -> None:
    model_dir = MODEL_ROOT / label_dir
    out_path = model_dir / "p_swing_probs.parquet"
    y = pd.read_parquet(DATASET_DIR / "y.parquet")
    splits = _load_split_indices("SPY", DATASET, X_FILENAME)
    train_val = np.sort(np.concatenate([np.sort(splits["train"]), np.sort(splits["val"])]))
    test = np.sort(splits["test"])
    probs = _empty_probs(y.index)

    full = np.load(model_dir / "p_swing_full.npy")
    if full.shape[0] == len(y):
        probs[["p_short_full", "p_neutral_full", "p_long_full"]] = full

    oof = np.load(model_dir / "p_swing_oof_train.npy")
    test_probs = np.load(model_dir / "p_swing_test.npy")
    if oof.shape[0] == len(train_val) and test_probs.shape[0] == len(test):
        probs.iloc[train_val, probs.columns.get_indexer(["p_short_oof_train", "p_neutral_oof_train", "p_long_oof_train"])] = oof
        probs.iloc[test, probs.columns.get_indexer(["p_short_test", "p_neutral_test", "p_long_test"])] = test_probs
        source = "oos"
    elif full.shape[0] == len(y):
        probs[["p_short_oof_train", "p_neutral_oof_train", "p_long_oof_train"]] = full
        probs[["p_short_test", "p_neutral_test", "p_long_test"]] = full
        source = "full_fit"
    else:
        raise ValueError(f"Cannot map probability arrays for {model_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    probs.to_parquet(out_path)
    print(f"[prepare] wrote {out_path} source={source} rows={len(probs)}")


def main() -> None:
    _build_plot_frame()
    _write_probs("swing_single")
    _write_probs("swing_support_single")


if __name__ == "__main__":
    main()
