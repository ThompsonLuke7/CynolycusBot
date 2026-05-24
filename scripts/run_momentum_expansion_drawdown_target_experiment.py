from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_momentum_expansion_label_variants import _raw_cross_sectional_score
from scripts.run_momentum_expansion_label_model_experiment import (
    _clean_X,
    _feature_columns,
    _sample_train,
    _selection_summary,
    _time_split,
)


DEFAULT_MATRIX = Path("momentum_expansion/data/processed/training_matrix_4h.parquet")
DEFAULT_OUT = Path("momentum_expansion/data/processed/drawdown_target_experiment")
RANDOM_STATE = 42


def _soft_dd_penalty(dd: pd.Series, *, free: float, severe: float, floor: float = 0.25) -> pd.Series:
    span = max(severe - free, 1e-9)
    penalty = 1.0 - ((dd - free).clip(lower=0.0) / span)
    return penalty.clip(lower=floor, upper=1.0)


def _target_variants(df: pd.DataFrame) -> dict[str, pd.Series]:
    base = df["raw_xsec_expansion_score"].astype(float)
    dd = df["fwd_max_drawdown"].astype(float)
    return {
        "DD_A_no_extra_penalty": base,
        "DD_B_soft_8_to_25": base * _soft_dd_penalty(dd, free=0.08, severe=0.25, floor=0.25),
        "DD_C_soft_12_to_30": base * _soft_dd_penalty(dd, free=0.12, severe=0.30, floor=0.25),
        "DD_D_hard_over_35": base.where(dd <= 0.35, base * 0.05),
    }


def _drawdown_diagnostics(test: pd.DataFrame, pred: pd.Series, name: str) -> dict[str, object]:
    work = test[["fwd_max_return", "fwd_close_return", "fwd_max_drawdown"]].copy()
    work["pred"] = pred.reindex(work.index)
    g = work.loc[work["pred"] >= work["pred"].quantile(0.95)].copy()
    if g.empty:
        return {"model": name}
    gt20 = g["fwd_max_return"] >= 0.20
    clean_gt20 = gt20 & (g["fwd_max_drawdown"] <= 0.15)
    very_clean_gt20 = gt20 & (g["fwd_max_drawdown"] <= 0.10)
    ugly = g["fwd_max_drawdown"] > 0.25
    return {
        "model": name,
        "top5_rows": int(len(g)),
        "top5_pct_gt20": float(gt20.mean()),
        "top5_pct_clean_gt20_dd_lte_15": float(clean_gt20.mean()),
        "top5_pct_very_clean_gt20_dd_lte_10": float(very_clean_gt20.mean()),
        "top5_pct_ugly_dd_gt_25": float(ugly.mean()),
        "top5_avg_return_per_dd": float((g["fwd_max_return"] / g["fwd_max_drawdown"].clip(lower=0.01)).mean()),
        "top5_median_drawdown": float(g["fwd_max_drawdown"].median()),
        "top5_p90_drawdown": float(g["fwd_max_drawdown"].quantile(0.90)),
    }


def run(matrix_path: Path, out_dir: Path, max_train_rows: int, test_frac: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    required = [
        "fwd_max_return",
        "fwd_max_alpha",
        "fwd_atr_adj_return",
        "fwd_max_drawdown",
        "fwd_close_return",
        "trend_persistence",
        "expansion_score",
        "expansion_target",
    ]
    df = pd.read_parquet(matrix_path).dropna(subset=required)
    df["raw_xsec_expansion_score"] = _raw_cross_sectional_score(df)
    df = df.dropna(subset=["raw_xsec_expansion_score"]).copy()

    feature_cols = _feature_columns(df)
    train, test, split_ts = _time_split(df, test_frac)
    X_train_all = _clean_X(train, feature_cols)
    X_test = _clean_X(test, feature_cols)

    train_targets = _target_variants(train)
    test_targets = _target_variants(test)
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    pred_frames: list[pd.Series] = []

    for name, y_train in train_targets.items():
        keep = _sample_train(train, y_train, max_train_rows)
        model = HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_all.iloc[keep], y_train.iloc[keep])
        pred = pd.Series(model.predict(X_test), index=test.index, name=name)
        y_test = test_targets[name]
        metric_rows.append(
            {
                "model": name,
                "spearman_to_own_target": float(pred.corr(y_test, method="spearman")),
                "spearman_to_raw_xsec_score": float(pred.corr(test["raw_xsec_expansion_score"], method="spearman")),
                "spearman_to_fwd_max_return": float(pred.corr(test["fwd_max_return"], method="spearman")),
                "spearman_to_drawdown": float(pred.corr(test["fwd_max_drawdown"], method="spearman")),
            }
        )
        selection_rows.extend(_selection_summary(test, pred, name))
        diagnostic_rows.append(_drawdown_diagnostics(test, pred, name))
        pred_frames.append(pred)

    metrics = pd.DataFrame(metric_rows)
    selection = pd.DataFrame(selection_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    preds = pd.concat(pred_frames, axis=1)
    meta = {
        "matrix": str(matrix_path),
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "split_timestamp": str(split_ts),
        "feature_count": int(len(feature_cols)),
        "max_train_rows_per_model": int(max_train_rows),
        "test_frac": float(test_frac),
        "targets": {
            "DD_A_no_extra_penalty": "raw_xsec_expansion_score",
            "DD_B_soft_8_to_25": "base * penalty: no penalty <=8% drawdown, floor 0.25 by 25%",
            "DD_C_soft_12_to_30": "base * penalty: no penalty <=12% drawdown, floor 0.25 by 30%",
            "DD_D_hard_over_35": "base unchanged <=35% drawdown, multiplied by 0.05 above 35%",
        },
    }

    metrics.to_csv(out_dir / "model_metrics.csv", index=False)
    selection.to_csv(out_dir / "selection_quality.csv", index=False)
    diagnostics.to_csv(out_dir / "drawdown_diagnostics.csv", index=False)
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
    print("Drawdown diagnostics, top 5% by prediction")
    print(diagnostics.to_string(index=False))
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
