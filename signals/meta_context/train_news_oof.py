"""Out-of-fold (OOF) prediction generator for the catalyst classifier.

Trains K time-ordered expanding-window folds on the train slice of the
catalyst training matrix. Each fold trains a fresh model on the cumulative
prior chunks, then predicts on its held-out chunk. Concatenated predictions
form a leak-free OOF parquet that downstream consumers (meta-ranker) can
join on ``record_id`` without seeing the model's training memory.

Output:
    meta_context/data/processed/catalyst_oof_{target}.parquet

Columns:
    record_id, ticker, timestamp, fold
    + binary: oof_pred  (P(class=1))
    + multiclass: oof_p_<class> for each class label

Usage:
    python -m meta_context.train_news_oof --target target_expansion_10pct
    python -m meta_context.train_news_oof --target target_trajectory_code

The "final" production model is whatever already lives in
meta_context/models/news_catalyst_*_v3.xgb.json. That model is trained on
ALL train data and is what we use for val/test/live records (where it has
never seen the labels). This script handles only the train-slice records
(where naive scoring would leak).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


DEFAULT_MATRIX = Path("signals/meta_context/data/processed/catalyst_training_matrix.parquet")
OUTPUT_DIR = Path("signals/meta_context/data/processed")

# Trajectory class names — must match build_catalyst_training_matrix.py and
# news/live_scorer.py TRAJECTORY_LABELS so downstream joins line up.
TRAJECTORY_NAMES = ("flat", "bull_steady", "bull_volatile", "v_bounce", "crash_stayed")


def _split_xy(sub: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [
        "record_id", "ticker", "timestamp", "catalyst_family", "catalyst_subtype",
        "source_quality",
        "expansion_label",
        "target_expansion_10pct", "target_expansion_5pct", "target_crash_5pct",
        "target_fwd_10d_reg", "target_trajectory", "target_trajectory_code",
        "max_forward_return", "max_drawdown",
        "forward_5d_return", "forward_10d_return", "split",
    ]
    X = sub.drop(columns=[c for c in drop_cols if c in sub.columns])
    y = sub[target]
    return X, y


def _make_classifier(*, num_class: int | None, device: str = "cuda",
                     n_estimators: int = 600, learning_rate: float = 0.03,
                     max_depth: int = 6) -> xgb.XGBClassifier:
    if num_class and num_class > 2:
        return xgb.XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss",
            num_class=int(num_class),
            tree_method="hist", device=device,
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, verbosity=0,
        )
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", device=device,
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbosity=0,
    )


def train_oof(
    target: str,
    matrix_path: Path = DEFAULT_MATRIX,
    *,
    n_folds: int = 5,
    device: str = "cuda",
    expanding: bool = True,
) -> pd.DataFrame:
    print(f"loading matrix from {matrix_path}")
    df = pd.read_parquet(matrix_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    train = df[df["split"] == "train"].copy().sort_values("timestamp").reset_index(drop=True)
    print(f"  train rows: {len(train):,}  splitting into {n_folds} folds ({'expanding window' if expanding else 'rolling'})")

    is_multiclass = target == "target_trajectory_code"

    # Time-ordered fold assignment — each fold is one contiguous chunk
    fold_size = len(train) // n_folds
    fold_assignments = np.zeros(len(train), dtype=int)
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else len(train)
        fold_assignments[start:end] = k
    train["_fold"] = fold_assignments

    # Pre-compute X/y once
    X_all, y_all = _split_xy(train, target)
    feature_cols = list(X_all.columns)

    oof_rows = []
    if is_multiclass:
        classes = sorted(train[target].unique())
        num_class = len(classes)
        print(f"  multiclass: {num_class} classes = {classes}")
    else:
        num_class = None

    for fold_k in range(1, n_folds):  # fold 0 has no prior data → skip
        if expanding:
            train_mask = train["_fold"] < fold_k
        else:
            # Rolling: only the previous fold
            train_mask = train["_fold"] == (fold_k - 1)
        test_mask = train["_fold"] == fold_k

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_train < 100 or n_test < 1:
            continue
        print(f"  fold {fold_k}: train={n_train:,} predict={n_test:,}")

        X_tr = X_all.loc[train_mask]
        y_tr = y_all.loc[train_mask]
        X_te = X_all.loc[test_mask]

        clf = _make_classifier(num_class=num_class, device=device)
        clf.fit(X_tr, y_tr, verbose=False)

        if is_multiclass:
            proba = clf.predict_proba(X_te)
            for i_local, (i_global, rec) in enumerate(train.loc[test_mask].iterrows()):
                row = {
                    "record_id": rec["record_id"],
                    "ticker": rec["ticker"],
                    "timestamp": rec["timestamp"],
                    "fold": fold_k,
                }
                for c_idx, c_name in enumerate(TRAJECTORY_NAMES):
                    if c_idx < proba.shape[1]:
                        row[f"oof_p_{c_name}"] = float(proba[i_local, c_idx])
                oof_rows.append(row)
        else:
            proba = clf.predict_proba(X_te)[:, 1]
            for i_local, (i_global, rec) in enumerate(train.loc[test_mask].iterrows()):
                oof_rows.append({
                    "record_id": rec["record_id"],
                    "ticker": rec["ticker"],
                    "timestamp": rec["timestamp"],
                    "fold": fold_k,
                    "oof_pred": float(proba[i_local]),
                })

    out = pd.DataFrame(oof_rows)
    print(f"  total OOF predictions: {len(out):,} of {len(train):,} train records")
    print(f"  (records in fold 0 have no OOF prediction — earliest chunk has no prior data to train on)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"catalyst_oof_{target}.parquet"
    out.to_parquet(output_path, index=False)
    print(f"saved -> {output_path}")

    # Sanity check: AUC across all OOF predictions
    if not out.empty:
        if is_multiclass:
            from sklearn.metrics import roc_auc_score
            train_with_oof = train.merge(out, on="record_id", how="inner")
            for c_idx, c_name in enumerate(TRAJECTORY_NAMES):
                col = f"oof_p_{c_name}"
                if col not in train_with_oof.columns:
                    continue
                y_bin = (train_with_oof[target] == c_idx).astype(int)
                if y_bin.nunique() < 2:
                    continue
                auc = roc_auc_score(y_bin, train_with_oof[col])
                print(f"    {c_name:<16}  OOF AUC = {auc:.3f}")
        else:
            from sklearn.metrics import roc_auc_score
            train_with_oof = train.merge(out, on="record_id", how="inner")
            auc = roc_auc_score(train_with_oof[target], train_with_oof["oof_pred"])
            print(f"  OOF AUC on train slice: {auc:.3f}")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train K-fold OOF catalyst predictions.")
    parser.add_argument("--target",
                        choices=["target_expansion_10pct", "target_expansion_5pct",
                                 "target_crash_5pct", "target_trajectory_code"],
                        required=True)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_oof(args.target, matrix_path=Path(args.matrix), n_folds=args.folds, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
