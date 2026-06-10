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

from strategies.momentum_expansion.config.momentum_config import FEATURES_COMBINED, LABELS_COMBINED
from strategies.momentum_expansion.features.feature_matrix_4h import FEATURE_COLUMNS_4H
from strategies.momentum_expansion.inference.candidate_filter import momentum_candidate_mask
from scripts.analyze_momentum_expansion_label_variants import _rank_by_date, _raw_cross_sectional_score


DEFAULT_OUT = Path("strategies/momentum_expansion/data/processed/candidate_filter_experiment")
RANDOM_STATE = 42

LABEL_COLUMNS = [
    "fwd_max_return",
    "fwd_max_alpha",
    "fwd_atr_adj_return",
    "fwd_max_drawdown",
    "fwd_close_return",
    "trend_persistence",
    "expansion_score",
    "expansion_target",
]


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _base_liquid(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    mask &= _num(df, "low_price_flag", 1.0).fillna(1.0) <= 0.0
    mask &= _num(df, "dollar_vol_pctile_252", 0.0).fillna(0.0) >= 0.20
    return mask


def _current(df: pd.DataFrame) -> pd.Series:
    return momentum_candidate_mask(df)


def _loose_current(df: pd.DataFrame) -> pd.Series:
    mask = _base_liquid(df)
    near_high = (_num(df, "dist_to_52w_high_atr") <= 16.0) | (_num(df, "xsec_near_high_rank") >= 0.45)
    rel_strength = (_num(df, "rs_spy_20") >= -0.02) | (_num(df, "xsec_ret_20_rank") >= 0.45)
    mask &= near_high.fillna(False)
    mask &= rel_strength.fillna(False)
    mask &= _num(df, "range_pos_20").fillna(0.0) >= 0.35
    return mask


def _strict_current(df: pd.DataFrame) -> pd.Series:
    mask = _base_liquid(df)
    mask &= _num(df, "dist_to_52w_high_atr").fillna(999.0) <= 8.0
    mask &= _num(df, "rs_spy_20").fillna(-999.0) >= 0.0
    mask &= _num(df, "xsec_ret_20_rank").fillna(0.0) >= 0.65
    mask &= _num(df, "range_pos_20").fillna(0.0) >= 0.55
    return mask


def _current_above_200dma(df: pd.DataFrame) -> pd.Series:
    return _current(df) & (_num(df, "daily_dist_200dma_atr").fillna(-999.0) >= 0.0)


def _above_200dma_rs(df: pd.DataFrame) -> pd.Series:
    mask = _base_liquid(df)
    mask &= _num(df, "daily_dist_200dma_atr").fillna(-999.0) >= 0.0
    mask &= ((_num(df, "rs_spy_20") >= 0.0) | (_num(df, "xsec_ret_20_rank") >= 0.60)).fillna(False)
    mask &= _num(df, "range_pos_20").fillna(0.0) >= 0.35
    return mask


def _near_high_rs(df: pd.DataFrame) -> pd.Series:
    mask = _base_liquid(df)
    mask &= _num(df, "dist_to_52w_high_atr").fillna(999.0) <= 8.0
    mask &= ((_num(df, "rs_spy_20") >= 0.0) | (_num(df, "xsec_ret_20_rank") >= 0.60)).fillna(False)
    return mask


def _base_reclaim(df: pd.DataFrame) -> pd.Series:
    mask = _base_liquid(df)
    mask &= _num(df, "drawdown_from_60h").between(0.05, 0.35, inclusive="both").fillna(False)
    mask &= _num(df, "range_contraction_20_60").fillna(999.0) <= 1.05
    mask &= _num(df, "close_tightness_10").fillna(999.0) <= 0.70
    mask &= _num(df, "base_position_60").fillna(0.0) >= 0.45
    mask &= _num(df, "range_pos_20").fillna(0.0) >= 0.50
    mask &= _num(df, "ret_5").fillna(-999.0) >= 0.0
    mask &= _num(df, "ret_10").fillna(-999.0) >= -0.02
    mask &= ((_num(df, "rs_spy_20") >= -0.02) | (_num(df, "xsec_ret_20_rank") >= 0.45)).fillna(False)
    return mask


FILTERS = {
    "ALL_VALID_ROWS": lambda df: pd.Series(True, index=df.index),
    "CURRENT_SHARED_FILTER": _current,
    "LOOSE_CURRENT": _loose_current,
    "STRICT_CURRENT": _strict_current,
    "CURRENT_PLUS_ABOVE_200DMA": _current_above_200dma,
    "ABOVE_200DMA_RS": _above_200dma_rs,
    "NEAR_HIGH_RS": _near_high_rs,
    "BASE_RECLAIM": _base_reclaim,
    "CURRENT_OR_BASE_RECLAIM": lambda df: _current(df) | _base_reclaim(df),
}


def _load_matrix(features_path: Path, labels_path: Path) -> pd.DataFrame:
    feats = pd.read_parquet(features_path, columns=FEATURE_COLUMNS_4H)
    labs = pd.read_parquet(labels_path, columns=LABEL_COLUMNS)
    df = feats.join(labs, how="inner")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=LABEL_COLUMNS)
    feature_cols = [c for c in FEATURE_COLUMNS_4H if c in df.columns]
    df = df.dropna(subset=feature_cols)
    df["raw_xsec_expansion_score"] = _raw_cross_sectional_score(df)
    df = df.dropna(subset=["raw_xsec_expansion_score"]).copy()
    df["raw_xsec_top10"] = (_rank_by_date(df, "raw_xsec_expansion_score") >= 0.90).astype(float)
    return df


def _time_split(df: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    ts = pd.Series(pd.to_datetime(df.index.get_level_values("timestamp")).unique()).sort_values()
    split_idx = int(len(ts) * (1.0 - test_frac))
    split_ts = pd.Timestamp(ts.iloc[max(1, min(split_idx, len(ts) - 1))])
    idx_ts = pd.to_datetime(df.index.get_level_values("timestamp"))
    return df.loc[idx_ts < split_ts].copy(), df.loc[idx_ts >= split_ts].copy(), split_ts


def _sample_index(n: int, max_rows: int) -> np.ndarray:
    if n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(RANDOM_STATE)
    return np.sort(rng.choice(n, size=max_rows, replace=False))


def _clean_train_test(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    med = x_train.median(numeric_only=True)
    x_test = test[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    return x_train.fillna(med).fillna(0.0), x_test.fillna(med).fillna(0.0)


def _summarize_rows(df: pd.DataFrame, mask: pd.Series, name: str) -> dict[str, object]:
    mask = mask.reindex(df.index).fillna(False)
    g = df.loc[mask]
    base_gt20 = float((df["fwd_max_return"] >= 0.20).mean())
    selected_gt20 = float((g["fwd_max_return"] >= 0.20).mean()) if len(g) else np.nan
    clean20 = (g["fwd_max_return"] >= 0.20) & (g["fwd_max_drawdown"] <= 0.15) if len(g) else pd.Series(dtype=bool)
    ts_level = "timestamp" if "timestamp" in df.index.names else 0
    return {
        "gate": name,
        "rows": int(len(g)),
        "row_share": float(len(g) / max(len(df), 1)),
        "tickers": int(g.index.get_level_values("ticker").nunique()) if len(g) else 0,
        "avg_tickers_per_bar": float(g.groupby(level=ts_level).size().mean()) if len(g) else 0.0,
        "avg_raw_xsec_score": float(g["raw_xsec_expansion_score"].mean()) if len(g) else np.nan,
        "raw_xsec_top10_rate": float(g["raw_xsec_top10"].mean()) if len(g) else np.nan,
        "expansion_target_rate": float(g["expansion_target"].mean()) if len(g) else np.nan,
        "avg_fwd_max_return": float(g["fwd_max_return"].mean()) if len(g) else np.nan,
        "median_fwd_max_return": float(g["fwd_max_return"].median()) if len(g) else np.nan,
        "avg_fwd_alpha": float(g["fwd_max_alpha"].mean()) if len(g) else np.nan,
        "avg_fwd_close_return": float(g["fwd_close_return"].mean()) if len(g) else np.nan,
        "avg_drawdown": float(g["fwd_max_drawdown"].mean()) if len(g) else np.nan,
        "median_drawdown": float(g["fwd_max_drawdown"].median()) if len(g) else np.nan,
        "pct_gt_20": selected_gt20,
        "pct_clean_gt20_dd_lte_15": float(clean20.mean()) if len(g) else np.nan,
        "pct_gt_40": float((g["fwd_max_return"] >= 0.40).mean()) if len(g) else np.nan,
        "gt20_lift_vs_all": float(selected_gt20 / base_gt20) if base_gt20 > 0 else np.nan,
        "close_win_rate": float((g["fwd_close_return"] > 0).mean()) if len(g) else np.nan,
    }


def _selection_summary(test: pd.DataFrame, pred: pd.Series, eval_mask: pd.Series, model_name: str, eval_gate: str) -> list[dict[str, object]]:
    work = test[
        [
            "fwd_max_return",
            "fwd_max_alpha",
            "fwd_close_return",
            "fwd_max_drawdown",
            "raw_xsec_expansion_score",
            "raw_xsec_top10",
        ]
    ].copy()
    work["pred"] = pred.reindex(work.index)
    work = work.loc[eval_mask.reindex(work.index).fillna(False)].dropna(subset=["pred"])
    if work.empty:
        return []
    base_gt20 = float((work["fwd_max_return"] >= 0.20).mean())
    rows: list[dict[str, object]] = []
    selections = {
        "top_5pct": work["pred"] >= work["pred"].quantile(0.95),
        "top_10pct": work["pred"] >= work["pred"].quantile(0.90),
    }
    topn = (
        work.reset_index()
        .sort_values(["timestamp", "pred"], ascending=[True, False])
        .groupby("timestamp")
        .head(5)
        .set_index(["timestamp", "ticker"])
    )
    selections["top5_per_4h_bar"] = work.index.isin(topn.index)
    for selection_name, sel in selections.items():
        g = work.loc[sel]
        gt20 = g["fwd_max_return"] >= 0.20
        clean20 = gt20 & (g["fwd_max_drawdown"] <= 0.15)
        rows.append(
            {
                "model": model_name,
                "eval_gate": eval_gate,
                "selection": selection_name,
                "rows": int(len(g)),
                "avg_pred": float(g["pred"].mean()) if len(g) else np.nan,
                "avg_raw_xsec_score": float(g["raw_xsec_expansion_score"].mean()) if len(g) else np.nan,
                "raw_xsec_top10_rate": float(g["raw_xsec_top10"].mean()) if len(g) else np.nan,
                "avg_fwd_max_return": float(g["fwd_max_return"].mean()) if len(g) else np.nan,
                "median_fwd_max_return": float(g["fwd_max_return"].median()) if len(g) else np.nan,
                "avg_fwd_alpha": float(g["fwd_max_alpha"].mean()) if len(g) else np.nan,
                "avg_fwd_close_return": float(g["fwd_close_return"].mean()) if len(g) else np.nan,
                "avg_drawdown": float(g["fwd_max_drawdown"].mean()) if len(g) else np.nan,
                "median_drawdown": float(g["fwd_max_drawdown"].median()) if len(g) else np.nan,
                "pct_gt_20": float(gt20.mean()) if len(g) else np.nan,
                "pct_clean_gt20_dd_lte_15": float(clean20.mean()) if len(g) else np.nan,
                "pct_gt_40": float((g["fwd_max_return"] >= 0.40).mean()) if len(g) else np.nan,
                "gt20_lift_vs_eval_gate": float(gt20.mean() / base_gt20) if len(g) and base_gt20 > 0 else np.nan,
                "close_win_rate": float((g["fwd_close_return"] > 0).mean()) if len(g) else np.nan,
            }
        )
    return rows


def _train_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_masks: dict[str, pd.Series],
    eval_masks: dict[str, pd.Series],
    feature_cols: list[str],
    max_train_rows: int,
    min_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    x_test_cache: pd.DataFrame | None = None

    for model_name, train_mask in train_masks.items():
        fit_df = train.loc[train_mask.reindex(train.index).fillna(False)]
        if len(fit_df) < min_train_rows:
            metric_rows.append({"model": model_name, "train_rows": int(len(fit_df)), "skipped": True})
            continue
        keep = _sample_index(len(fit_df), max_train_rows)
        x_fit_all, x_test = _clean_train_test(fit_df, test, feature_cols)
        x_test_cache = x_test
        y_fit = fit_df["raw_xsec_expansion_score"].astype(float).iloc[keep]
        model = HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
        model.fit(x_fit_all.iloc[keep], y_fit)
        pred = pd.Series(model.predict(x_test_cache), index=test.index, name=model_name)
        metric_rows.append(
            {
                "model": model_name,
                "train_rows": int(len(fit_df)),
                "fit_rows": int(len(keep)),
                "skipped": False,
                "test_spearman_all_rows": float(pred.corr(test["raw_xsec_expansion_score"], method="spearman")),
                "test_spearman_fwd_max_return": float(pred.corr(test["fwd_max_return"], method="spearman")),
                "test_spearman_drawdown": float(pred.corr(test["fwd_max_drawdown"], method="spearman")),
            }
        )
        for eval_gate, eval_mask in eval_masks.items():
            selection_rows.extend(_selection_summary(test, pred, eval_mask, model_name, eval_gate))

    return pd.DataFrame(metric_rows), pd.DataFrame(selection_rows)


def run(
    features_path: Path,
    labels_path: Path,
    out_dir: Path,
    max_train_rows: int,
    min_train_rows: int,
    test_frac: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_matrix(features_path, labels_path)
    masks = {name: fn(df).astype(bool) for name, fn in FILTERS.items()}
    masks["BASE_RECLAIM_OUTSIDE_CURRENT"] = masks["BASE_RECLAIM"] & ~masks["CURRENT_SHARED_FILTER"]

    train, test, split_ts = _time_split(df, test_frac)
    train_masks = {name: mask.reindex(train.index).fillna(False) for name, mask in masks.items()}
    test_masks = {name: mask.reindex(test.index).fillna(False) for name, mask in masks.items()}

    gate_quality = pd.DataFrame(
        [_summarize_rows(test, test_masks[name], name) for name in test_masks]
    ).sort_values(["pct_clean_gt20_dd_lte_15", "pct_gt_20"], ascending=False)

    model_scope_names = [
        "ALL_VALID_ROWS",
        "CURRENT_SHARED_FILTER",
        "LOOSE_CURRENT",
        "STRICT_CURRENT",
        "CURRENT_PLUS_ABOVE_200DMA",
        "ABOVE_200DMA_RS",
        "BASE_RECLAIM",
        "CURRENT_OR_BASE_RECLAIM",
    ]
    eval_scope_names = [
        "ALL_VALID_ROWS",
        "CURRENT_SHARED_FILTER",
        "CURRENT_OR_BASE_RECLAIM",
        "BASE_RECLAIM",
        "BASE_RECLAIM_OUTSIDE_CURRENT",
    ]
    feature_cols = [c for c in FEATURE_COLUMNS_4H if c in df.columns]
    model_metrics, model_selection = _train_and_score(
        train=train,
        test=test,
        train_masks={name: train_masks[name] for name in model_scope_names},
        eval_masks={name: test_masks[name] for name in eval_scope_names},
        feature_cols=feature_cols,
        max_train_rows=max_train_rows,
        min_train_rows=min_train_rows,
    )

    metadata = {
        "features_path": str(features_path),
        "labels_path": str(labels_path),
        "rows_total_unfiltered": int(len(df)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "split_timestamp": str(split_ts),
        "feature_count": int(len(feature_cols)),
        "max_train_rows_per_model": int(max_train_rows),
        "min_train_rows": int(min_train_rows),
        "test_frac": float(test_frac),
        "target": "raw_xsec_expansion_score from fwd_max_alpha, fwd_atr_adj_return, trend_persistence, and fwd_max_drawdown ranks",
        "note": "Metrics use overlapping 25-bar forward labels; they measure selection quality, not account-level CAGR.",
    }

    gate_quality.to_csv(out_dir / "gate_quality_holdout.csv", index=False)
    model_metrics.to_csv(out_dir / "model_training_scope_metrics.csv", index=False)
    model_selection.to_csv(out_dir / "model_selection_quality.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Metadata")
    print(json.dumps(metadata, indent=2))
    print()
    print("Gate quality, holdout")
    print(gate_quality.to_string(index=False))
    print()
    print("Model training-scope metrics")
    print(model_metrics.to_string(index=False))
    print()
    print("Top5 per 4H bar on CURRENT_SHARED_FILTER")
    cur = model_selection[
        (model_selection["eval_gate"] == "CURRENT_SHARED_FILTER")
        & (model_selection["selection"] == "top5_per_4h_bar")
    ].sort_values(["pct_clean_gt20_dd_lte_15", "pct_gt_20"], ascending=False)
    print(cur.to_string(index=False))
    print()
    print("Top5 per 4H bar on CURRENT_OR_BASE_RECLAIM")
    cur_base = model_selection[
        (model_selection["eval_gate"] == "CURRENT_OR_BASE_RECLAIM")
        & (model_selection["selection"] == "top5_per_4h_bar")
    ].sort_values(["pct_clean_gt20_dd_lte_15", "pct_gt_20"], ascending=False)
    print(cur_base.to_string(index=False))
    print()
    print(f"Wrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES_COMBINED)
    parser.add_argument("--labels", type=Path, default=LABELS_COMBINED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-train-rows", type=int, default=150_000)
    parser.add_argument("--min-train-rows", type=int, default=25_000)
    parser.add_argument("--test-frac", type=float, default=0.20)
    args = parser.parse_args()
    run(
        features_path=args.features,
        labels_path=args.labels,
        out_dir=args.out,
        max_train_rows=args.max_train_rows,
        min_train_rows=args.min_train_rows,
        test_frac=args.test_frac,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
