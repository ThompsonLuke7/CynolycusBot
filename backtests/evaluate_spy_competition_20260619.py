from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.shared_plotting import (
    apply_time_ticks,
    compute_time_ticks,
    make_price_probability_figure,
    plot_candles_from_frame,
    save_figure,
)
from scripts.build_spy_daytrader_context_export import (
    _add_dealer_features,
    _load_options_daily,
    _load_prior_bid_ask_10m,
    _local_bar_time,
    _local_date,
)
from signals.location_features import add_liquidity_zone_features
from strategies.spy_intraday.Models.competition_ranker import EmpiricalScoreCalibrator


OUT = ROOT / "backtests/20260619_spy_competition"
RAW_FRAME = Path("/tmp/spy_competition_eval_20260619/raw_feature_label_frame.parquet")
META_FRAME = Path("/tmp/spy_competition_eval_20260619/meta_feature_frame.parquet")
TRAIN_MATRIX = ROOT / "Data/processed/spy/training_export/spy_daytrader_context_matrix.parquet"
MANIFEST = ROOT / "Data/processed/spy/training_export/spy_daytrader_context_manifest.json"
CANDIDATES = ROOT / "Data/models/ga_xgboost/10min/competition_20260619"
ACTIVE = ROOT / "Data/models/ga_xgboost/10min/single/swing_support_single"
LEGACY = ROOT / "Data/models/ga_xgboost/10min"
OPTIONS = ROOT / "drive-download-20260613T045727Z-3-001/spy_options_daily_dataset_3y_fixed_greeks.jsonl"
BID_ASK = ROOT / "drive-download-20260613T045727Z-3-001/spy_intraday_data.parquet"
OI = ROOT / "drive-download-20260613T045727Z-3-001/spy_open_interest_5_yr.parquet"
DEALER = ROOT / "Data/dealer_positioning/reports"
START = pd.Timestamp("2026-04-02", tz="UTC")
TOP_K = 5
HORIZONS = (1, 3, 6, 12, 20)


def _numeric_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return (
        frame.reindex(columns=features)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(np.float32)
    )


def _build_context(raw: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    base = raw.copy()
    base.index = pd.to_datetime(base.index, utc=True)
    base.index.name = "timestamp"
    matrix = base.reset_index()
    matrix["bar_date"] = _local_date(matrix["timestamp"])
    matrix["bar_time_local"] = _local_bar_time(matrix["timestamp"])
    for src, dst in (
        ("open", "raw_open"),
        ("high", "raw_high"),
        ("low", "raw_low"),
        ("close", "raw_close"),
        ("volume", "raw_volume"),
    ):
        matrix[dst] = pd.to_numeric(matrix[src], errors="coerce")

    price = base[["open", "high", "low", "close", "volume"]]
    liquidity, _, helper_cols = add_liquidity_zone_features(
        price,
        lookback=78,
        swing_window=18,
        zone_width_pct=0.0015,
        volume_window=39,
    )
    liquidity = liquidity.reset_index()
    matrix = matrix.merge(
        liquidity[["timestamp", *manifest["liquidity_feature_columns"], *helper_cols]],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    options_daily, _, _ = _load_options_daily(OPTIONS)
    if not options_daily.empty:
        matrix = matrix.merge(options_daily, on="bar_date", how="left")
    bid_ask, _ = _load_prior_bid_ask_10m(BID_ASK)
    if not bid_ask.empty:
        matrix = matrix.merge(bid_ask, on="bar_time_local", how="left")

    plot = matrix[["timestamp", "raw_open", "raw_high", "raw_low", "raw_close", "raw_volume"]]
    matrix, _ = _add_dealer_features(
        matrix,
        oi_path=OI,
        dealer_report_dir=DEALER,
        plot=plot,
    )
    return matrix.set_index("timestamp").sort_index()


def _load_candidate(side: str):
    label = f"{side}_swing_label"
    model = joblib.load(CANDIDATES / label / "best_model.joblib")
    model.set_params(device="cpu")
    return model


def _load_legacy(side: str) -> tuple[xgb.Booster, list[str]]:
    path = LEGACY / side / "swing"
    model = xgb.Booster()
    model.load_model(path / "xgb_model.json")
    model.set_param({"device": "cpu"})
    features = [line.strip() for line in (path / "selected_features.txt").read_text().splitlines() if line.strip()]
    return model, features


def _score_active(raw: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    features = [line.strip() for line in (ACTIVE / "selected_features.txt").read_text().splitlines() if line.strip()]
    combined = raw.copy()
    combined.index = pd.to_datetime(combined.index).tz_convert("America/New_York")
    meta = meta.copy()
    meta.index = pd.to_datetime(meta.index).tz_convert("America/New_York")
    for col in meta.columns:
        if col not in combined.columns:
            combined[col] = meta[col].reindex(combined.index)
    model = xgb.Booster()
    model.load_model(ACTIVE / "xgb_model.json")
    model.set_param({"device": "cpu"})
    preds = model.predict(xgb.DMatrix(_numeric_matrix(combined, features), missing=np.nan))
    classes = json.loads((ACTIVE / "meta.json").read_text())["classes"]
    out = pd.DataFrame(preds, index=combined.index, columns=classes)
    out.index = out.index.tz_convert("UTC")
    return out.rename(columns={"long": "active_long", "short": "active_short", "neutral": "active_neutral"})


def _fit_candidate_calibrators(features: list[str]) -> dict[str, EmpiricalScoreCalibrator]:
    train = pd.read_parquet(TRAIN_MATRIX)
    train["timestamp"] = pd.to_datetime(train["timestamp"], utc=True)
    train = train.sort_values("timestamp").reset_index(drop=True)
    val = train.iloc[int(len(train) * 0.70) : int(len(train) * 0.85)]
    calibrators = {}
    for side in ("long", "short"):
        model = _load_candidate(side)
        scores = model.predict(_numeric_matrix(val, features))
        calibrator = EmpiricalScoreCalibrator.fit(scores)
        joblib.dump(calibrator, CANDIDATES / f"{side}_swing_label" / "score_percentile_calibrator.joblib")
        calibrators[side] = calibrator
    return calibrators


def _score_models(context: pd.DataFrame, raw: pd.DataFrame, meta: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    scores = pd.DataFrame(index=context.index)
    calibrators = _fit_candidate_calibrators(features)
    for side in ("long", "short"):
        candidate = _load_candidate(side)
        raw_score = candidate.predict(_numeric_matrix(context, features))
        scores[f"candidate_{side}_score"] = raw_score
        scores[f"candidate_{side}"] = calibrators[side].transform(raw_score)

        legacy, legacy_features = _load_legacy(side)
        scores[f"legacy_{side}"] = legacy.predict(
            xgb.DMatrix(_numeric_matrix(raw, legacy_features), missing=np.nan)
        )
    return scores.join(_score_active(raw, meta), how="left")


def _add_future_returns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    local_session = pd.Series(
        out.index.tz_convert("America/New_York").normalize(),
        index=out.index,
    )
    # The competition trained on the 10min_shift1 export: binary swing events
    # are moved one bar earlier within each session. The raw feature rebuild
    # emits unshifted labels, so recreate the exact evaluation targets here.
    for side in ("long", "short"):
        source = pd.to_numeric(out[f"{side}_swing_label"], errors="coerce")
        out[f"eval_{side}_swing_label"] = source.groupby(local_session).shift(-1).fillna(0).astype(int)
    close = pd.to_numeric(out["close"], errors="coerce")
    for horizon in HORIZONS:
        out[f"fwd_ret_{horizon}"] = close.shift(-horizon) / close - 1.0
        future_high = pd.concat([out["high"].shift(-i) for i in range(1, horizon + 1)], axis=1).max(axis=1)
        future_low = pd.concat([out["low"].shift(-i) for i in range(1, horizon + 1)], axis=1).min(axis=1)
        out[f"fwd_mfe_long_{horizon}"] = future_high / close - 1.0
        out[f"fwd_mae_long_{horizon}"] = future_low / close - 1.0
    return out


def _max_drawdown(series: pd.Series) -> float:
    equity = series.fillna(0.0).cumsum()
    return float((equity - equity.cummax()).min()) if len(equity) else 0.0


def _metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_pairs = {
        "candidate": ("candidate_long_score", "candidate_short_score"),
        "active": ("active_long", "active_short"),
        "legacy": ("legacy_long", "legacy_short"),
    }
    pred = pred.copy()
    pred["date"] = pred.index.tz_convert("America/New_York").date
    for model_name, (long_col, short_col) in model_pairs.items():
        for side, score_col, label_col, sign in (
            ("long", long_col, "eval_long_swing_label", 1.0),
            ("short", short_col, "eval_short_swing_label", -1.0),
        ):
            valid = pred.dropna(subset=[score_col])
            label = pd.to_numeric(valid[label_col], errors="coerce").fillna(0).astype(int)
            try:
                auc = float(roc_auc_score(label, valid[score_col]))
            except ValueError:
                auc = float("nan")
            for horizon in HORIZONS:
                selected = (
                    valid.groupby("date", group_keys=False)
                    .apply(lambda day: day.nlargest(min(TOP_K, len(day)), score_col), include_groups=False)
                )
                directional = sign * pd.to_numeric(selected[f"fwd_ret_{horizon}"], errors="coerce")
                daily = directional.groupby(selected.index.tz_convert("America/New_York").date).mean()
                corr = spearmanr(
                    pd.to_numeric(valid[score_col], errors="coerce"),
                    sign * pd.to_numeric(valid[f"fwd_ret_{horizon}"], errors="coerce"),
                    nan_policy="omit",
                ).statistic
                rows.append(
                    {
                        "model": model_name,
                        "side": side,
                        "horizon_bars": horizon,
                        "n_rows": len(valid),
                        "auc_swing_label": auc,
                        "score_forward_return_spearman": float(corr),
                        "top5_mean_directional_return": float(directional.mean()),
                        "top5_win_rate": float((directional > 0).mean()),
                        "top5_sum_daily_mean_return": float(daily.sum()),
                        "top5_max_drawdown": _max_drawdown(daily),
                        "top5_label_precision": float(pd.to_numeric(selected[label_col], errors="coerce").mean()),
                    }
                )
    return pd.DataFrame(rows)


def _coverage(context: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    recent = context.loc[context.index >= START]
    groups = {
        "base": [c for c in manifest["feature_columns"] if c not in set(
            manifest["option_feature_columns"]
            + manifest["bid_ask_feature_columns"]
            + manifest["liquidity_feature_columns"]
            + manifest["dealer_feature_columns"]
        )],
        "options": manifest["option_feature_columns"],
        "bid_ask": manifest["bid_ask_feature_columns"],
        "liquidity": manifest["liquidity_feature_columns"],
        "dealer": manifest["dealer_feature_columns"],
    }
    rows = []
    for name, cols in groups.items():
        cols = [c for c in cols if c in recent.columns]
        rows.append(
            {
                "group": name,
                "features": len(cols),
                "non_null_cell_rate": float(recent[cols].notna().to_numpy().mean()) if cols else 0.0,
                "rows_with_any": float(recent[cols].notna().any(axis=1).mean()) if cols else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _plots(pred: pd.DataFrame, metrics: pd.DataFrame) -> None:
    p = pred.loc[pred.index >= pred.index.max() - pd.Timedelta(days=20)].copy()
    p.index = p.index.tz_convert("America/New_York")
    fig, ax_price, ax_prob = make_price_probability_figure(figsize=(16, 9))
    candle = plot_candles_from_frame(ax_price, p, compressed=True)
    for side, marker, color in (("long", "^", "#16a34a"), ("short", "v", "#dc2626")):
        score_col = f"candidate_{side}_score"
        picks = p.groupby(p.index.date, group_keys=False).apply(
            lambda day: day.nlargest(min(TOP_K, len(day)), score_col),
            include_groups=False,
        )
        y = picks["low"] if side == "long" else picks["high"]
        ax_price.scatter(p.index.get_indexer(picks.index), y, marker=marker, color=color, s=22, label=f"candidate {side}")
    ax_price.set_title("SPY candidate ranker top-5 signals per side — latest 20 days")
    ax_price.set_ylabel("SPY")
    ax_price.legend(loc="upper left")
    ax_prob.plot(candle.x, p["candidate_long"], label="candidate long percentile", color="#16a34a")
    ax_prob.plot(candle.x, p["candidate_short"], label="candidate short percentile", color="#dc2626")
    ax_prob.axhline(0.872, color="#64748b", ls="--", lw=0.8)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("Validation percentile")
    ax_prob.legend(loc="upper left")
    tick_positions, tick_labels = compute_time_ticks(p.index, candle.x, fmt="%m/%d")
    apply_time_ticks(ax_prob, tick_positions, tick_labels)
    save_figure(fig, OUT / "candidate_signals_latest_20_days.png", close=True)

    focus = metrics[metrics["horizon_bars"].eq(6)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, side in zip(axes, ("long", "short")):
        s = focus[focus["side"].eq(side)]
        ax.bar(s["model"], 100 * s["top5_mean_directional_return"])
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(f"{side.title()} top-5 mean 60-minute directional return")
        ax.set_ylabel("Return (%)")
    save_figure(fig, OUT / "underlying_60m_comparison.png", close=True)


def main() -> None:
    if not RAW_FRAME.exists() or not META_FRAME.exists():
        raise FileNotFoundError("Build the temporary raw/meta feature frames before running this evaluation.")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    features = manifest["feature_columns"]
    raw = pd.read_parquet(RAW_FRAME)
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw.index.name = "timestamp"
    meta = pd.read_parquet(META_FRAME)
    meta.index.name = "timestamp"
    context = _build_context(raw, manifest)
    scores = _score_models(context, raw, meta, features)
    pred = _add_future_returns(raw.join(scores))
    pred = pred.loc[pred.index >= START].copy()

    keep = [
        "open", "high", "low", "close", "volume", "atr",
        "long_swing_label", "short_swing_label",
        "eval_long_swing_label", "eval_short_swing_label", "atr_realized_return",
        *[c for c in pred.columns if c.startswith(("candidate_", "active_", "legacy_", "fwd_"))],
    ]
    pred[keep].to_parquet(OUT / "prediction_frame.parquet")
    context.loc[context.index >= START, features].to_parquet(OUT / "recent_context_matrix.parquet")
    metrics = _metrics(pred)
    metrics.to_csv(OUT / "underlying_metrics.csv", index=False)
    coverage = _coverage(context, manifest)
    coverage.to_csv(OUT / "feature_coverage.csv", index=False)

    for model in ("candidate", "active"):
        signal = pred[["open", "high", "low", "close", "atr"]].copy()
        signal["p_enter_long"] = pred[f"{model}_long"]
        signal["p_enter_short"] = pred[f"{model}_short"]
        signal.to_parquet(OUT / f"{model}_signal_frame.parquet")

    _plots(pred, metrics)
    summary = {
        "start": str(pred.index.min()),
        "end": str(pred.index.max()),
        "rows": len(pred),
        "days": int(pd.Series(pred.index.tz_convert("America/New_York").date).nunique()),
        "top_k": TOP_K,
        "candidate_policy_threshold": 0.872,
        "active_policy_threshold": 0.5,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(metrics[metrics["horizon_bars"].isin([3, 6, 12])].to_string(index=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
