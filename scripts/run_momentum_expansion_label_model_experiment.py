from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_momentum_expansion_label_variants import (
    _rank_by_date,
    _raw_cross_sectional_score,
)


DEFAULT_MATRIX = Path("strategies/momentum_expansion/data/processed/training_matrix_4h.parquet")
DEFAULT_OUT = Path("strategies/momentum_expansion/data/processed/label_model_experiment")
RANDOM_STATE = 42


def _feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {
        "fwd_max_return",
        "fwd_max_alpha",
        "fwd_atr_adj_return",
        "fwd_max_drawdown",
        "fwd_close_return",
        "trend_persistence",
        "expansion_score",
        "expansion_target",
        "momentum_candidate",
        "raw_xsec_expansion_score",
        "target_b_xsec_top10",
    }
    return [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]


def _clean_X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    return X.fillna(med).fillna(0.0)


def _time_split(df: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    ts = pd.Series(pd.to_datetime(df.index.get_level_values("timestamp")).unique()).sort_values()
    split_idx = int(len(ts) * (1.0 - test_frac))
    split_ts = pd.Timestamp(ts.iloc[max(1, min(split_idx, len(ts) - 1))])
    idx_ts = pd.to_datetime(df.index.get_level_values("timestamp"))
    return df.loc[idx_ts < split_ts].copy(), df.loc[idx_ts >= split_ts].copy(), split_ts


def _sample_train(df: pd.DataFrame, target: pd.Series, max_rows: int) -> np.ndarray:
    if len(df) <= max_rows:
        return np.arange(len(df))
    rng = np.random.default_rng(RANDOM_STATE)
    if target.nunique(dropna=True) == 2:
        pos_idx = np.flatnonzero(target.to_numpy() > 0)
        neg_idx = np.flatnonzero(target.to_numpy() <= 0)
        pos_keep = min(len(pos_idx), max_rows // 2)
        neg_keep = max_rows - pos_keep
        keep = np.concatenate(
            [
                rng.choice(pos_idx, size=pos_keep, replace=False),
                rng.choice(neg_idx, size=min(len(neg_idx), neg_keep), replace=False),
            ]
        )
        rng.shuffle(keep)
        return keep
    return np.sort(rng.choice(len(df), size=max_rows, replace=False))


def _selection_summary(test: pd.DataFrame, pred: pd.Series, name: str) -> list[dict[str, object]]:
    work = test[
        ["fwd_max_return", "fwd_close_return", "fwd_max_drawdown"]
    ].copy()
    work["pred"] = pred.reindex(work.index)
    work = work.dropna(subset=["pred"])
    base_gt20 = float((work["fwd_max_return"] >= 0.20).mean())
    rows: list[dict[str, object]] = []
    for label, mask in {
        "top_1pct": work["pred"] >= work["pred"].quantile(0.99),
        "top_5pct": work["pred"] >= work["pred"].quantile(0.95),
        "top_10pct": work["pred"] >= work["pred"].quantile(0.90),
    }.items():
        g = work.loc[mask]
        rate20 = float((g["fwd_max_return"] >= 0.20).mean()) if len(g) else np.nan
        winners = g.loc[g["fwd_close_return"] > 0, "fwd_close_return"]
        losers = g.loc[g["fwd_close_return"] <= 0, "fwd_close_return"]
        rows.append(
            {
                "model": name,
                "selection": label,
                "rows": int(len(g)),
                "avg_fwd_max_return": float(g["fwd_max_return"].mean()) if len(g) else np.nan,
                "median_fwd_max_return": float(g["fwd_max_return"].median()) if len(g) else np.nan,
                "avg_fwd_close_return": float(g["fwd_close_return"].mean()) if len(g) else np.nan,
                "avg_drawdown": float(g["fwd_max_drawdown"].mean()) if len(g) else np.nan,
                "pct_gt_20": rate20,
                "pct_gt_25": float((g["fwd_max_return"] >= 0.25).mean()) if len(g) else np.nan,
                "pct_gt_40": float((g["fwd_max_return"] >= 0.40).mean()) if len(g) else np.nan,
                "gt20_lift_vs_test": float(rate20 / base_gt20) if base_gt20 > 0 else np.nan,
                "avg_close_winner": float(winners.mean()) if len(winners) else np.nan,
                "avg_close_loser": float(losers.mean()) if len(losers) else np.nan,
                "close_win_rate": float((g["fwd_close_return"] > 0).mean()) if len(g) else np.nan,
            }
        )

    topn = (
        work.reset_index()
        .sort_values(["timestamp", "pred"], ascending=[True, False])
        .groupby("timestamp")
        .head(5)
    )
    rate20 = float((topn["fwd_max_return"] >= 0.20).mean()) if len(topn) else np.nan
    rows.append(
        {
            "model": name,
            "selection": "top5_per_4h_bar",
            "rows": int(len(topn)),
            "avg_fwd_max_return": float(topn["fwd_max_return"].mean()) if len(topn) else np.nan,
            "median_fwd_max_return": float(topn["fwd_max_return"].median()) if len(topn) else np.nan,
            "avg_fwd_close_return": float(topn["fwd_close_return"].mean()) if len(topn) else np.nan,
            "avg_drawdown": float(topn["fwd_max_drawdown"].mean()) if len(topn) else np.nan,
            "pct_gt_20": rate20,
            "pct_gt_25": float((topn["fwd_max_return"] >= 0.25).mean()) if len(topn) else np.nan,
            "pct_gt_40": float((topn["fwd_max_return"] >= 0.40).mean()) if len(topn) else np.nan,
            "gt20_lift_vs_test": float(rate20 / base_gt20) if base_gt20 > 0 else np.nan,
            "avg_close_winner": float(topn.loc[topn["fwd_close_return"] > 0, "fwd_close_return"].mean()) if len(topn) else np.nan,
            "avg_close_loser": float(topn.loc[topn["fwd_close_return"] <= 0, "fwd_close_return"].mean()) if len(topn) else np.nan,
            "close_win_rate": float((topn["fwd_close_return"] > 0).mean()) if len(topn) else np.nan,
        }
    )
    return rows


def _binary_metrics(y_true: pd.Series, pred: pd.Series, name: str) -> dict[str, object]:
    return {
        "model": name,
        "target_rate": float(y_true.mean()),
        "auc": float(roc_auc_score(y_true, pred)) if y_true.nunique() == 2 else np.nan,
        "average_precision": float(average_precision_score(y_true, pred)) if y_true.nunique() == 2 else np.nan,
    }


def run(matrix_path: Path, out_dir: Path, max_train_rows: int, test_frac: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols_for_read = None
    df = pd.read_parquet(matrix_path, columns=cols_for_read).dropna(
        subset=[
            "fwd_max_return",
            "fwd_max_alpha",
            "fwd_atr_adj_return",
            "fwd_max_drawdown",
            "fwd_close_return",
            "trend_persistence",
            "expansion_score",
            "expansion_target",
        ]
    )
    df["target_b_xsec_top10"] = (_rank_by_date(df, "expansion_score") >= 0.90).astype(int)
    df["raw_xsec_expansion_score"] = _raw_cross_sectional_score(df)
    df = df.dropna(subset=["raw_xsec_expansion_score"]).copy()

    feature_cols = _feature_columns(df)
    train, test, split_ts = _time_split(df, test_frac)
    X_train_all = _clean_X(train, feature_cols)
    X_test = _clean_X(test, feature_cols)

    configs = [
        ("EXP_A_current_classifier", "classifier", train["expansion_target"].astype(int), test["expansion_target"].astype(int)),
        ("EXP_B_xsec_top10_classifier", "classifier", train["target_b_xsec_top10"].astype(int), test["target_b_xsec_top10"].astype(int)),
        ("EXP_C_xsec_score_regression", "regression", train["raw_xsec_expansion_score"].astype(float), test["raw_xsec_expansion_score"].astype(float)),
    ]

    metric_rows = []
    selection_rows = []
    preds_out = []
    for name, kind, y_train, y_test in configs:
        keep = _sample_train(train, y_train, max_train_rows)
        X_train = X_train_all.iloc[keep]
        y_fit = y_train.iloc[keep]
        if kind == "classifier":
            model = HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                early_stopping=True,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_fit)
            pred = pd.Series(model.predict_proba(X_test)[:, 1], index=test.index, name=name)
            metric_rows.append(_binary_metrics(y_test, pred, name))
        else:
            model = HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                early_stopping=True,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_fit)
            pred = pd.Series(model.predict(X_test), index=test.index, name=name)
            metric_rows.append(
                {
                    "model": name,
                    "target_rate": np.nan,
                    "auc": np.nan,
                    "average_precision": np.nan,
                    "spearman_to_raw_xsec_score": float(pred.corr(y_test, method="spearman")),
                }
            )
        selection_rows.extend(_selection_summary(test, pred, name))
        preds_out.append(pred)

    metrics = pd.DataFrame(metric_rows)
    selection = pd.DataFrame(selection_rows)
    preds = pd.concat(preds_out, axis=1)
    meta = {
        "matrix": str(matrix_path),
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "split_timestamp": str(split_ts),
        "feature_count": int(len(feature_cols)),
        "max_train_rows_per_model": int(max_train_rows),
        "test_frac": float(test_frac),
        "note": "CAGR is intentionally omitted: labels are overlapping 25-bar forward windows, not a trade simulation.",
    }

    metrics.to_csv(out_dir / "model_metrics.csv", index=False)
    selection.to_csv(out_dir / "selection_quality.csv", index=False)
    preds.to_parquet(out_dir / "holdout_predictions.parquet")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("Metadata")
    print(json.dumps(meta, indent=2))
    print()
    print("Model metrics")
    print(metrics.to_string(index=False))
    print()
    print("Selection quality")
    print(selection.to_string(index=False))
    print()
    print(f"Wrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-train-rows", type=int, default=250_000)
    parser.add_argument("--test-frac", type=float, default=0.20)
    args = parser.parse_args()
    run(args.matrix, args.out, args.max_train_rows, args.test_frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
