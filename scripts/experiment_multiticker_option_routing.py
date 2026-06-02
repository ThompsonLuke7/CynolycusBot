from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROBS_PATH = Path("multi_ticker_swing/models/p_swing_probs_600.parquet")
FEATURES_PATH = Path("multi_ticker_swing/data/processed/features_30m.parquet")
UNIVERSE_PATH = Path("multi_ticker_swing/config/trading_universe.json")
OUT_DIR = Path("Data/analysis/multi_ticker_swing_live/experiments/option_routing")

HORIZONS = [1, 2, 4, 8, 13, 26]

FEATURE_COLS = [
    "atr_pct_14",
    "atr_pct_z_64",
    "realized_vol_4",
    "realized_vol_16",
    "range_expansion_4",
    "range_expansion_16",
    "range_regime_8_32",
    "true_range_pct",
    "close_location_in_bar",
    "volume_z_20",
    "volume_rel_20",
    "dollar_volume_pctile",
    "relative_volume_open_window",
    "range_pos_20",
    "dist_20bar_high",
    "dist_20bar_low",
    "breakout_pressure_score",
    "swing_setup_score_long",
    "swing_setup_score_short",
    "bars_from_open",
    "bars_to_close",
    "gap_pct",
    "gap_to_atr",
    "spy_ret_4",
    "spy_ret_16",
    "rel_str_spy_4",
    "rel_str_spy_16",
    "daily_atr_pct",
    "daily_rsi_14",
    "daily_trend_state",
    "daily_range_pos_20",
    "daily_vol_rel_20",
    "volatility_pctile_rolling",
    "dollar_vol_pctile_rolling",
    "trendiness_score_rolling",
]


def _load_universe(path: Path, max_tier: int) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for ticker, cfg in raw.items():
        tier = int(cfg["tier"])
        if tier > max_tier:
            continue
        rows.append(
            {
                "ticker": ticker,
                "tier": tier,
                "entry_threshold": float(cfg["entry_threshold"]),
                "avg_win_pct": float(cfg["avg_win_pct"]),
                "avg_loss_pct": float(cfg["avg_loss_pct"]),
                "profit_factor": float(cfg["profit_factor"]),
                "sharpe": float(cfg["sharpe"]),
            }
        )
    return pd.DataFrame(rows)


def _forward_frame(
    features: pd.DataFrame,
    tickers: set[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    cols = ["close", "high", "low", *[c for c in FEATURE_COLS if c in features.columns]]
    idx_ts = features.index.get_level_values("timestamp")
    idx_ticker = features.index.get_level_values("ticker")
    end_with_buffer = end_ts + pd.Timedelta(days=14)
    mask = idx_ticker.isin(tickers) & (idx_ts >= start_ts) & (idx_ts <= end_with_buffer)
    df = features.loc[mask, cols].copy()
    df = df.reset_index().sort_values(["ticker", "timestamp"])
    df["bar_i"] = df.groupby("ticker").cumcount()

    parts: list[pd.DataFrame] = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.copy()
        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        rev_high = high.shift(-1).iloc[::-1]
        rev_low = low.shift(-1).iloc[::-1]
        for h in HORIZONS:
            g[f"close_fwd_{h}"] = close.shift(-h)
            g[f"future_high_{h}"] = rev_high.rolling(h, min_periods=1).max().iloc[::-1].to_numpy()
            g[f"future_low_{h}"] = rev_low.rolling(h, min_periods=1).min().iloc[::-1].to_numpy()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _build_signal_dataset(max_tier: int, non_overlap_bars: int, splits: set[str]) -> pd.DataFrame:
    universe = _load_universe(UNIVERSE_PATH, max_tier=max_tier)
    probs = pd.read_parquet(PROBS_PATH)
    probs = probs[probs["ticker"].isin(set(universe["ticker"]))].copy()
    probs = probs[probs["split"].isin(splits)].copy()
    probs = probs.merge(universe, on="ticker", how="inner")

    p_sum = (probs["p_long"] + probs["p_short"]).clip(lower=1e-9)
    probs["p_long_dir"] = probs["p_long"] / p_sum
    probs["p_short_dir"] = probs["p_short"] / p_sum
    probs["direction"] = 0
    probs.loc[probs["p_long_dir"] >= probs["entry_threshold"], "direction"] = 1
    probs.loc[
        (probs["direction"] == 0) & (probs["p_short_dir"] >= probs["entry_threshold"]),
        "direction",
    ] = -1
    probs["p_dir"] = np.where(probs["direction"] == 1, probs["p_long_dir"], probs["p_short_dir"])
    probs["ev_score"] = probs["p_dir"] * probs["avg_win_pct"] + (1.0 - probs["p_dir"]) * probs["avg_loss_pct"]
    probs = probs[(probs["direction"] != 0) & (probs["ev_score"] > 0)].copy()
    start_ts = pd.to_datetime(probs["timestamp"], utc=True).min() - pd.Timedelta(days=7)
    end_ts = pd.to_datetime(probs["timestamp"], utc=True).max()

    needed = ["close", "high", "low", *FEATURE_COLS]
    available_cols = set(pq.ParquetFile(FEATURES_PATH).schema_arrow.names)
    feature_cols = [c for c in needed if c in available_cols]
    features = pd.read_parquet(FEATURES_PATH, columns=feature_cols)
    fwd = _forward_frame(features, set(universe["ticker"]), start_ts=start_ts, end_ts=end_ts)
    merged = probs.merge(fwd, on=["timestamp", "ticker"], how="inner")

    for h in HORIZONS:
        merged[f"signed_close_ret_{h}"] = (
            merged["direction"] * (merged[f"close_fwd_{h}"] - merged["close"]) / merged["close"] * 100.0
        )
        merged[f"mfe_{h}"] = np.where(
            merged["direction"] == 1,
            (merged[f"future_high_{h}"] - merged["close"]) / merged["close"] * 100.0,
            (merged["close"] - merged[f"future_low_{h}"]) / merged["close"] * 100.0,
        )
        merged[f"mae_{h}"] = np.where(
            merged["direction"] == 1,
            (merged["close"] - merged[f"future_low_{h}"]) / merged["close"] * 100.0,
            (merged[f"future_high_{h}"] - merged["close"]) / merged["close"] * 100.0,
        )

    merged["dir_close_location"] = np.where(
        merged["direction"] == 1,
        merged.get("close_location_in_bar", np.nan),
        1.0 - merged.get("close_location_in_bar", np.nan),
    )
    merged["dir_setup_score"] = np.where(
        merged["direction"] == 1,
        merged.get("swing_setup_score_long", np.nan),
        merged.get("swing_setup_score_short", np.nan),
    )
    merged["direction_name"] = np.where(merged["direction"] == 1, "long", "short")

    if non_overlap_bars > 0:
        kept = []
        for _, g in merged.sort_values(["ticker", "timestamp"]).groupby("ticker", sort=False):
            last_i = -10**9
            for idx, row in g.iterrows():
                bar_i = int(row["bar_i"])
                if bar_i - last_i >= non_overlap_bars:
                    kept.append(idx)
                    last_i = bar_i
        merged = merged.loc[kept].copy()
    return merged


def _summarize_group(df: pd.DataFrame, name: str, horizon: int, target_move: float) -> dict[str, Any]:
    if df.empty:
        return {
            "strategy": name,
            "signals": 0,
            "gamma_hit_rate": math.nan,
            "directional_win_rate": math.nan,
            "avg_signed_close_ret": math.nan,
            "avg_mfe": math.nan,
            "avg_mae": math.nan,
            "median_mfe": math.nan,
            "large_close_win_rate": math.nan,
        }
    gamma_hit = (df[f"mfe_{horizon}"] >= target_move) & (df[f"signed_close_ret_{horizon}"] > 0)
    return {
        "strategy": name,
        "signals": int(len(df)),
        "gamma_hit_rate": float(gamma_hit.mean()),
        "directional_win_rate": float((df[f"signed_close_ret_{horizon}"] > 0).mean()),
        "avg_signed_close_ret": float(df[f"signed_close_ret_{horizon}"].mean()),
        "avg_mfe": float(df[f"mfe_{horizon}"].mean()),
        "avg_mae": float(df[f"mae_{horizon}"].mean()),
        "median_mfe": float(df[f"mfe_{horizon}"].median()),
        "large_close_win_rate": float((df[f"signed_close_ret_{horizon}"] >= target_move).mean()),
    }


def _gate_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    true = pd.Series(True, index=df.index)
    return {
        "all_signals_stock_baseline": true,
        "confidence_80": df["p_dir"] >= 0.80,
        "confidence_85": df["p_dir"] >= 0.85,
        "high_atr_1pct": df["atr_pct_14"] >= 0.010,
        "high_atr_1p5pct": df["atr_pct_14"] >= 0.015,
        "volume_expansion": (df["volume_rel_20"] >= 1.5) & (df["range_expansion_4"] >= 1.2),
        "breakout_gamma": (
            (df["p_dir"] >= 0.75)
            & (df["atr_pct_14"] >= 0.010)
            & (df["volume_rel_20"] >= 1.2)
            & (df["dir_close_location"] >= 0.65)
            & (df["bars_to_close"] >= 4)
        ),
        "liquid_breakout_gamma": (
            (df["p_dir"] >= 0.75)
            & (df["atr_pct_14"] >= 0.010)
            & (df["volume_rel_20"] >= 1.2)
            & (df["dir_close_location"] >= 0.65)
            & (df["dollar_volume_pctile"] >= 0.60)
            & (df["bars_to_close"] >= 4)
        ),
        "late_day_veto": df["bars_to_close"] >= 4,
        "stock_route_small_move": (df["atr_pct_14"] < 0.010) | (df["bars_to_close"] < 4),
    }


def _grid_search(df: pd.DataFrame, split: str, horizon: int, target_move: float, min_signals: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sample = df[df["split"] == split].copy()
    for p_min in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        for atr_min in [0.0, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020]:
            for vol_min in [0.0, 1.0, 1.25, 1.5, 2.0]:
                for range_min in [0.0, 1.0, 1.25, 1.5]:
                    for close_loc_min in [0.0, 0.55, 0.65, 0.75]:
                        for bars_min in [0, 2, 4, 6]:
                            mask = (
                                (sample["p_dir"] >= p_min)
                                & (sample["atr_pct_14"] >= atr_min)
                                & (sample["volume_rel_20"].fillna(0) >= vol_min)
                                & (sample["range_expansion_4"].fillna(0) >= range_min)
                                & (sample["dir_close_location"].fillna(0.5) >= close_loc_min)
                                & (sample["bars_to_close"].fillna(0) >= bars_min)
                            )
                            sub = sample[mask]
                            if len(sub) < min_signals:
                                continue
                            summary = _summarize_group(
                                sub,
                                f"p{p_min}_atr{atr_min}_vol{vol_min}_rng{range_min}_loc{close_loc_min}_bars{bars_min}",
                                horizon,
                                target_move,
                            )
                            summary.update(
                                {
                                    "p_min": p_min,
                                    "atr_min": atr_min,
                                    "vol_min": vol_min,
                                    "range_min": range_min,
                                    "close_loc_min": close_loc_min,
                                    "bars_min": bars_min,
                                }
                            )
                            rows.append(summary)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["score"] = (
        out["gamma_hit_rate"] * 2.0
        + out["avg_signed_close_ret"] / max(target_move, 1e-9)
        + out["avg_mfe"] / max(target_move, 1e-9) * 0.25
        - out["avg_mae"].clip(lower=0) / max(target_move, 1e-9) * 0.25
    )
    return out.sort_values(["score", "gamma_hit_rate", "signals"], ascending=[False, False, False])


def _fit_move_classifier(df: pd.DataFrame, horizon: int, target_move: float) -> pd.DataFrame:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
    except Exception as exc:
        return pd.DataFrame([{"model": "sklearn_unavailable", "error": str(exc)}])

    cols = [
        "p_dir",
        "ev_score",
        "tier",
        "profit_factor",
        "sharpe",
        "dir_close_location",
        "dir_setup_score",
        *[c for c in FEATURE_COLS if c in df.columns],
    ]
    cols = [c for c in cols if c in df.columns]
    train = df[df["split"].isin(["train", "val"])].copy()
    test = df[df["split"] == "test"].copy()
    train_y = ((train[f"mfe_{horizon}"] >= target_move) & (train[f"signed_close_ret_{horizon}"] > 0)).astype(int)
    test_y = ((test[f"mfe_{horizon}"] >= target_move) & (test[f"signed_close_ret_{horizon}"] > 0)).astype(int)
    if train_y.nunique() < 2 or test_y.nunique() < 2:
        return pd.DataFrame([{"model": "hist_gradient_boosting", "error": "not enough positive/negative labels"}])

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=7),
    )
    model.fit(train[cols], train_y)
    pred = model.predict_proba(test[cols])[:, 1]
    rows = [
        {
            "model": "hist_gradient_boosting",
            "test_signals": int(len(test)),
            "test_positive_rate": float(test_y.mean()),
            "roc_auc": float(roc_auc_score(test_y, pred)),
            "avg_precision": float(average_precision_score(test_y, pred)),
        }
    ]
    scored = test.copy()
    scored["pred_gamma_prob"] = pred
    scored["gamma_label"] = test_y.to_numpy()
    for q in [0.50, 0.70, 0.80, 0.90]:
        cutoff = float(np.quantile(pred, q))
        sub = scored[scored["pred_gamma_prob"] >= cutoff]
        rows.append(
            {
                "model": f"top_{int((1-q)*100)}pct_predicted_gamma",
                "test_signals": int(len(sub)),
                "test_positive_rate": float(sub["gamma_label"].mean()) if len(sub) else math.nan,
                "roc_auc": math.nan,
                "avg_precision": math.nan,
                "avg_mfe": float(sub[f"mfe_{horizon}"].mean()) if len(sub) else math.nan,
                "avg_signed_close_ret": float(sub[f"signed_close_ret_{horizon}"].mean()) if len(sub) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def run(max_tier: int, non_overlap_bars: int, horizon: int, target_move: float, min_signals: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"h{horizon}_target{str(target_move).replace('.', 'p')}"
    df = _build_signal_dataset(max_tier=max_tier, non_overlap_bars=non_overlap_bars, splits={"val", "test"})
    df.to_csv(OUT_DIR / f"{tag}_signal_move_dataset.csv", index=False)

    rows: list[dict[str, Any]] = []
    masks = _gate_masks(df)
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        for name, mask in masks.items():
            row = _summarize_group(split_df[mask.loc[split_df.index]], name, horizon, target_move)
            row["split"] = split
            rows.append(row)
        for direction_name, sub in split_df.groupby("direction_name"):
            row = _summarize_group(sub, f"all_{direction_name}", horizon, target_move)
            row["split"] = split
            rows.append(row)
    gate_summary = pd.DataFrame(rows)
    gate_summary.to_csv(OUT_DIR / f"{tag}_routing_gate_summary.csv", index=False)

    grid_val = _grid_search(df, "val", horizon, target_move, min_signals)
    grid_val.to_csv(OUT_DIR / f"{tag}_routing_gate_grid_val.csv", index=False)
    if not grid_val.empty:
        test_rows = []
        test_df = df[df["split"] == "test"].copy()
        for _, g in grid_val.head(25).iterrows():
            mask = (
                (test_df["p_dir"] >= g["p_min"])
                & (test_df["atr_pct_14"] >= g["atr_min"])
                & (test_df["volume_rel_20"].fillna(0) >= g["vol_min"])
                & (test_df["range_expansion_4"].fillna(0) >= g["range_min"])
                & (test_df["dir_close_location"].fillna(0.5) >= g["close_loc_min"])
                & (test_df["bars_to_close"].fillna(0) >= g["bars_min"])
            )
            row = _summarize_group(test_df[mask], str(g["strategy"]), horizon, target_move)
            for col in ["p_min", "atr_min", "vol_min", "range_min", "close_loc_min", "bars_min"]:
                row[col] = g[col]
            test_rows.append(row)
        pd.DataFrame(test_rows).to_csv(OUT_DIR / f"{tag}_routing_gate_grid_top25_test.csv", index=False)

    clf = _fit_move_classifier(df, horizon, target_move)
    clf.to_csv(OUT_DIR / f"{tag}_gamma_move_classifier_summary.csv", index=False)

    by_bucket = []
    test = df[df["split"] == "test"].copy()
    test["atr_bucket"] = pd.qcut(test["atr_pct_14"], 5, duplicates="drop")
    test["p_bucket"] = pd.qcut(test["p_dir"], 5, duplicates="drop")
    for col in ["atr_bucket", "p_bucket", "direction_name", "tier"]:
        for key, sub in test.groupby(col, observed=True):
            row = _summarize_group(sub, f"{col}={key}", horizon, target_move)
            row["bucket"] = col
            by_bucket.append(row)
    pd.DataFrame(by_bucket).to_csv(OUT_DIR / f"{tag}_test_bucket_diagnostics.csv", index=False)

    print(f"signals={len(df):,} output={OUT_DIR}")
    print(gate_summary[gate_summary["split"] == "test"].sort_values("gamma_hit_rate", ascending=False).head(12).to_string(index=False))
    if not grid_val.empty:
        print("\nTop validation gates retested on test:")
        print(pd.read_csv(OUT_DIR / f"{tag}_routing_gate_grid_top25_test.csv").head(10).to_string(index=False))
    print("\nClassifier:")
    print(clf.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tier", type=int, default=2)
    parser.add_argument("--non-overlap-bars", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8, choices=HORIZONS)
    parser.add_argument("--target-move", type=float, default=1.5)
    parser.add_argument("--min-signals", type=int, default=75)
    args = parser.parse_args()
    run(
        max_tier=args.max_tier,
        non_overlap_bars=args.non_overlap_bars,
        horizon=args.horizon,
        target_move=args.target_move,
        min_signals=args.min_signals,
    )


if __name__ == "__main__":
    main()
