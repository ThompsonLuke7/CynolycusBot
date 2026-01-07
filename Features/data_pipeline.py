from pathlib import Path
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from Data.load_data import (
    get_ticker_processed_base_dir,
    get_ticker_processed_split_dir,
    get_ticker_processed_stats_dir,
)
from Data.retrieve_data import normalize_ticker
from Features.feature_engineering import (
    SCALE_FEATURE_COLUMNS,
    apply_scaler_from_stats,
    normalize_continuous_features,
    save_normalization_stats,
)

# Default prefix for processed artifacts
DEFAULT_PREFIX = "spy_daily"


def _resolve_label_suffix(label_mode: str) -> str:
    if label_mode not in {"swing", "leg"}:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    return label_mode


def _infer_prefix(processed_dir: Path, ticker: str) -> str | None:
    slug = normalize_ticker(ticker).lower()
    candidates = sorted(processed_dir.glob(f"X_{slug}_*.parquet"))
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    name = newest.stem
    if name.startswith("X_"):
        return name[2:]
    return None


def _resolve_label_path(
    processed_dir: Path, prefix: str, label_suffix: str, side: str
) -> Path:
    candidate = processed_dir / f"y_{prefix}_{label_suffix}_{side}.parquet"
    if candidate.exists():
        return candidate
    return processed_dir / f"y_{prefix}_{side}.parquet"


def load_processed_frames(
    processed_dir: Path,
    prefix: str = DEFAULT_PREFIX,
    label_mode: str = "leg",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load the cleaned, unnormalized feature matrix and labels produced by feature_engineering.
    """
    processed_dir = Path(processed_dir)
    label_suffix = _resolve_label_suffix(label_mode)
    X = pd.read_parquet(processed_dir / f"X_{prefix}.parquet")
    y_long = pd.read_parquet(
        _resolve_label_path(processed_dir, prefix, label_suffix, "long")
    ).iloc[:, 0]
    y_short = pd.read_parquet(
        _resolve_label_path(processed_dir, prefix, label_suffix, "short")
    ).iloc[:, 0]
    close = pd.read_parquet(processed_dir / f"close_{prefix}.parquet").iloc[:, 0]
    return X, y_long, y_short, close


def chronological_split(
    X: pd.DataFrame,
    y_long: pd.Series,
    y_short: pd.Series,
    close: pd.Series,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, dict]:
    """
    Chronologically split features/labels into train/val/test.
    Fractions are relative; test takes the remainder.
    """
    n = len(X)
    if n == 0:
        raise ValueError("No rows to split")
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0,1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    splits = {
        "train": {
            "X": X.iloc[:train_end],
            "y_long": y_long.iloc[:train_end],
            "y_short": y_short.iloc[:train_end],
            "close": close.iloc[:train_end],
        },
        "val": {
            "X": X.iloc[train_end:val_end],
            "y_long": y_long.iloc[train_end:val_end],
            "y_short": y_short.iloc[train_end:val_end],
            "close": close.iloc[train_end:val_end],
        },
        "test": {
            "X": X.iloc[val_end:],
            "y_long": y_long.iloc[val_end:],
            "y_short": y_short.iloc[val_end:],
            "close": close.iloc[val_end:],
        },
    }
    return splits


def apply_train_scaler_to_splits(
    splits: Dict[str, dict], scale_cols: set[str] = SCALE_FEATURE_COLUMNS
) -> Tuple[Dict[str, dict], dict]:
    """
    Fit scaler on train split only, then apply to val/test splits.
    """
    train_X = splits["train"]["X"]
    train_scaled, stats = normalize_continuous_features(train_X, scale_cols)

    scaled_splits = {}
    for name, part in splits.items():
        X_part = part["X"]
        X_scaled = (
            train_scaled if name == "train" else apply_scaler_from_stats(X_part, stats)
        )
        scaled_splits[name] = {
            "X": X_scaled,
            "y_long": part["y_long"],
            "y_short": part["y_short"],
            "close": part["close"],
        }
    return scaled_splits, stats


def save_split_outputs(
    split_root: Path,
    prefix: str,
    scaled_splits: Dict[str, dict],
    stats: dict,
    stats_dir: Path,
    label_suffix: str = "leg",
) -> None:
    """
    Persist split datasets and scaler stats.
    """
    for split_name, part in scaled_splits.items():
        split_prefix = f"{prefix}_{label_suffix}"
        split_dir = split_root / split_prefix / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        X = part["X"].to_numpy(dtype=np.float32)
        y_long = part["y_long"].to_numpy(dtype=np.int64)
        y_short = part["y_short"].to_numpy(dtype=np.int64)
        close = part["close"].to_numpy(dtype=float)

        np.save(split_dir / f"X_{prefix}_{split_name}.npy", X)
        np.save(
            split_dir / f"y_{prefix}_{label_suffix}_long_{split_name}.npy", y_long
        )
        np.save(
            split_dir / f"y_{prefix}_{label_suffix}_short_{split_name}.npy", y_short
        )
        np.save(split_dir / f"close_{prefix}_{split_name}.npy", close)

        part["X"].astype(np.float32).to_parquet(
            split_dir / f"X_{prefix}_{split_name}.parquet", index=False
        )
        long_name = part["y_long"].name or "y_long"
        short_name = part["y_short"].name or "y_short"
        part["y_long"].to_frame(long_name).to_parquet(
            split_dir / f"y_{prefix}_{label_suffix}_long_{split_name}.parquet",
            index=False,
        )
        part["y_short"].to_frame(short_name).to_parquet(
            split_dir / f"y_{prefix}_{label_suffix}_short_{split_name}.parquet",
            index=False,
        )
        part["close"].to_frame("close").to_parquet(
            split_dir / f"close_{prefix}_{split_name}.parquet", index=False
        )

    save_normalization_stats(
        stats_dir,
        stats,
        filename=f"norm_stats_{prefix}_train.json",
    )


def main(
    processed_dir: Path | None = None,
    prefix: str | None = None,
    ticker: str = "$SPY",
    label_mode: str = "leg",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> None:
    """
    End-to-end: load processed features (unnormalized), split chronologically,
    fit scaler on train, apply to val/test, and save split artifacts.
    """
    clean_ticker = normalize_ticker(ticker)
    label_suffix = _resolve_label_suffix(label_mode)
    if prefix is None:
        inferred = _infer_prefix(
            processed_dir if processed_dir is not None else get_ticker_processed_base_dir(clean_ticker),
            clean_ticker,
        )
        prefix = inferred or f"{clean_ticker.lower()}_daily"
    if processed_dir is None:
        processed_dir = get_ticker_processed_base_dir(clean_ticker)

    X, y_long, y_short, close = load_processed_frames(
        processed_dir, prefix, label_mode=label_mode
    )
    splits = chronological_split(X, y_long, y_short, close, train_frac, val_frac)
    scaled_splits, stats = apply_train_scaler_to_splits(
        splits, SCALE_FEATURE_COLUMNS
    )
    split_root = get_ticker_processed_split_dir(clean_ticker)
    stats_dir = get_ticker_processed_stats_dir(clean_ticker)
    save_split_outputs(
        split_root,
        prefix,
        scaled_splits,
        stats,
        stats_dir,
        label_suffix=label_suffix,
    )
    print(
        f"Saved split datasets under {split_root / f'{prefix}_{label_suffix}'} "
        f"and scaler stats to {stats_dir} "
        f"with train/val/test = {train_frac}/{val_frac}/{1 - train_frac - val_frac}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split processed features into train/val/test chronologically."
    )
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--ticker", type=str, default="$SPY")
    parser.add_argument("--label_mode", type=str, default="leg", choices=["leg", "swing"])
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    main(
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        prefix=args.prefix,
        ticker=args.ticker,
        label_mode=args.label_mode,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
