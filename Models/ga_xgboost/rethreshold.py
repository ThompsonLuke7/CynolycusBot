from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from Data.retrieve_data import normalize_ticker
from Models.ga_xgboost.train import (
    _find_best_fbeta_threshold,
    _load_split_indices,
    _normalize_ga_label_dir,
    _print_binary_metrics,
)


def _label_columns(label_mode: str) -> tuple[str, str]:
    mode = str(label_mode).strip().lower()
    if mode == "swing":
        return "long_swing_label", "short_swing_label"
    if mode == "pivot":
        return "pivot_down", "pivot_up"
    if mode == "tb":
        return "tb_long_label", "tb_short_label"
    if mode in {"tb_cont", "tb_sparse", "continuation_tb"}:
        return "tb_cont_long_label", "tb_cont_short_label"
    if mode in {"entry_edge", "edge", "entry_quality"}:
        return "entry_edge_long_label", "entry_edge_short_label"
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _load_probs(path: Path, column: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if column not in df.columns:
        raise KeyError(f"Missing column {column} in {path}")
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute GA-XGB decision thresholds from saved probabilities."
    )
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--dataset-name", type=str, default="10min")
    parser.add_argument("--x-filename", type=str, default="X_10min_tree.parquet")
    parser.add_argument("--label-mode", type=str, default="swing")
    parser.add_argument("--threshold-beta", type=float, default=1.0)
    parser.add_argument("--model-root", type=str, default="Data/models")
    parser.add_argument("--split-root", type=str, default=None)
    args = parser.parse_args()

    beta = float(args.threshold_beta)
    if beta <= 0:
        raise ValueError("--threshold-beta must be > 0.")
    metric_label = "F1" if np.isclose(beta, 1.0) else f"F{beta:g}"

    clean = normalize_ticker(args.ticker)
    dataset_dir = REPO_ROOT / "Data" / "processed" / clean.lower() / "datasets" / args.dataset_name
    y_path = dataset_dir / "y.parquet"
    if not y_path.exists():
        raise FileNotFoundError(y_path)
    y_df = pd.read_parquet(y_path)
    long_col, short_col = _label_columns(args.label_mode)
    y_long = pd.to_numeric(y_df[long_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    y_short = pd.to_numeric(y_df[short_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)

    split_root = Path(args.split_root) if args.split_root else None
    splits = _load_split_indices(
        args.ticker,
        args.dataset_name,
        args.x_filename,
        split_root=split_root,
    )
    train_val_idx = np.sort(np.concatenate([np.sort(splits["train"]), np.sort(splits["val"])]))
    test_idx = np.sort(splits["test"])

    label_dir = _normalize_ga_label_dir(args.label_mode)
    root = REPO_ROOT / args.model_root / "ga_xgboost" / args.dataset_name
    long_probs_path = root / "long" / label_dir / "p_long_probs.parquet"
    short_probs_path = root / "short" / label_dir / "p_short_probs.parquet"
    p_long_oof = _load_probs(long_probs_path, "p_long_oof_train")[train_val_idx]
    p_short_oof = _load_probs(short_probs_path, "p_short_oof_train")[train_val_idx]
    p_long_test = _load_probs(long_probs_path, "p_long_test")[test_idx]
    p_short_test = _load_probs(short_probs_path, "p_short_test")[test_idx]

    y_long_train = y_long[train_val_idx]
    y_short_train = y_short[train_val_idx]
    y_long_test = y_long[test_idx]
    y_short_test = y_short[test_idx]

    long_thr, long_score = _find_best_fbeta_threshold(
        y_long_train,
        p_long_oof,
        beta=beta,
    )
    short_thr, short_score = _find_best_fbeta_threshold(
        y_short_train,
        p_short_oof,
        beta=beta,
    )
    print(
        f"[GA-XGB] LONG best OOF {metric_label} threshold={long_thr:.4f} "
        f"({metric_label.lower()}={long_score:.4f})"
    )
    _print_binary_metrics(
        y_long_train,
        p_long_oof,
        name="LONG OOF metrics",
        threshold=long_thr,
        threshold_metric_beta=beta,
    )
    _print_binary_metrics(
        y_long_test,
        p_long_test,
        name="LONG test metrics",
        threshold=long_thr,
        threshold_metric_beta=beta,
    )
    print(
        f"[GA-XGB] SHORT best OOF {metric_label} threshold={short_thr:.4f} "
        f"({metric_label.lower()}={short_score:.4f})"
    )
    _print_binary_metrics(
        y_short_train,
        p_short_oof,
        name="SHORT OOF metrics",
        threshold=short_thr,
        threshold_metric_beta=beta,
    )
    _print_binary_metrics(
        y_short_test,
        p_short_test,
        name="SHORT test metrics",
        threshold=short_thr,
        threshold_metric_beta=beta,
    )


if __name__ == "__main__":
    main()
