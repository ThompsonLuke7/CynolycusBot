"""
Bundle the 30m multi-ticker swing training matrix into a Colab archive.

Usage:
    python -m multi_ticker_swing.models.export_for_colab
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from signals.location_features import add_daily_high_low_gap_features, add_liquidity_zone_features
from signals.meta_context.build_meta_ranker_matrix import (
    CALENDAR_MACRO_COLS,
    NEWS_CATALYST_COLS,
    _join_calendar_macro_features,
)
from strategies.multi_ticker_swing.config.pipeline_config import (
    FEATURE_COLUMNS,
    NEUTRAL_WEIGHT_FACTOR,
    OOF_N_FOLDS,
    PROCESSED_30M_DIR,
    TRAINING_MATRIX,
    TRAIN_FRAC,
    VAL_FRAC,
    XGBOOST_CONFIG,
)

logger = logging.getLogger(__name__)

EXPORT_DIR = TRAINING_MATRIX.parent.parent / "training_export"
REPO_ROOT = Path(__file__).resolve().parents[3]
NEWS_CATALYST_SIGNAL = REPO_ROOT / "signals/meta_context/data/processed/news_catalyst_signal.parquet"
CBOE_OPTIONS_SUMMARY = REPO_ROOT / "signals/news/data/processed/cboe_options_summary.parquet"
DYNAMIC_THEME_FEATURES_HISTORY = REPO_ROOT / "themes/dynamic_theme/outputs/ticker_theme_features_history.parquet"
DYNAMIC_THEME_FEATURES = REPO_ROOT / "themes/dynamic_theme/outputs/ticker_theme_features.parquet"

THEME_CONTEXT_COLS = [
    "primary_theme_rank",
    "theme_heat_score",
    "theme_breadth",
    "theme_acceleration",
    "theme_strength",
    "membership_score",
    "related_theme_heat",
    "related_theme_rank",
    "theme_age_days",
    "theme_newness_score",
]

CBOE_CONTEXT_SOURCE_COLS = [
    "current_price",
    "stock_volume",
    "iv30",
    "iv30_change_percent",
    "call_volume",
    "put_volume",
    "call_open_interest",
    "put_open_interest",
    "call_premium",
    "put_premium",
    "put_call_volume_ratio",
    "unusual_strike_count",
    "unusual_total_volume",
]

SWING_PRUNED_BASE_FEATURES = {
    "dist_20bar_high",
    "dist_20bar_low",
    "overnight_ret",
    "dollar_vol_pctile_rolling",
    "bars_to_close",
    "log_ret_1",
}

SWING_LOCATION_FEATURE_COLS = [
    "distance_to_52w_high",
    "distance_to_52w_low",
    "distance_to_recent_20d_high",
    "distance_to_recent_20d_low",
    "distance_to_recent_60d_high",
    "distance_to_recent_60d_low",
    "distance_to_gap_above",
    "distance_to_gap_below",
    "gap_fill_rate_20d",
    "gap_fill_rate_60d",
    "breakout_proximity_daily",
    "support_proximity_daily",
    "resistance_proximity_daily",
    "distance_to_nearest_support_zone",
    "distance_to_nearest_resistance_zone",
    "inside_support_zone",
    "inside_resistance_zone",
    "support_zone_strength",
    "resistance_zone_strength",
    "failed_breakout_count",
    "failed_breakdown_count",
    "support_proximity",
    "resistance_proximity",
    "spy_distance_to_support",
    "spy_distance_to_resistance",
    "spy_inside_support_zone",
    "spy_inside_resistance_zone",
    "spy_regime_score",
]


def _norm_date(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert(None).dt.normalize()


def _asof_prior_day_ticker(spine: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left = spine.sort_values("date").reset_index()
    right = right.sort_values("date")
    merged = pd.merge_asof(
        left,
        right,
        on="date",
        by="ticker",
        direction="backward",
        allow_exact_matches=False,
    )
    return merged.set_index("index").sort_index()


def _daily_context_keys(df: pd.DataFrame) -> pd.DataFrame:
    keys = df[["ticker", "timestamp"]].copy()
    keys["ticker"] = keys["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    keys["date"] = _norm_date(keys["timestamp"])
    return keys[["ticker", "date"]].drop_duplicates().reset_index(drop=True)


def _join_news_context(keys: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str | None]:
    cols = [c for c in NEWS_CATALYST_COLS if c != "news_catalyst_score_std"]
    if not NEWS_CATALYST_SIGNAL.exists():
        return keys.copy(), [], None
    news = pd.read_parquet(NEWS_CATALYST_SIGNAL)
    if news.empty or not {"timestamp", "ticker"}.issubset(news.columns):
        return keys.copy(), [], str(NEWS_CATALYST_SIGNAL)

    keep = ["ticker", "timestamp"] + [c for c in cols if c in news.columns]
    news = news[keep].copy()
    news["ticker"] = news["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    news["date"] = _norm_date(news["timestamp"])
    feature_cols = [c for c in cols if c in news.columns]
    news = (
        news[["ticker", "date"] + feature_cols]
        .groupby(["ticker", "date"], as_index=False)
        .last()
        .sort_values(["ticker", "date"])
    )
    out = _asof_prior_day_ticker(keys, news)
    return out, feature_cols, str(NEWS_CATALYST_SIGNAL)


def _join_cboe_context(keys: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str | None]:
    if not CBOE_OPTIONS_SUMMARY.exists():
        return keys.copy(), [], None
    cboe = pd.read_parquet(CBOE_OPTIONS_SUMMARY)
    if cboe.empty or not {"ticker", "snapshot_date"}.issubset(cboe.columns):
        return keys.copy(), [], str(CBOE_OPTIONS_SUMMARY)

    keep = ["ticker", "snapshot_date"] + [c for c in CBOE_CONTEXT_SOURCE_COLS if c in cboe.columns]
    cboe = cboe[keep].copy()
    cboe["ticker"] = cboe["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    cboe["date"] = pd.to_datetime(cboe["snapshot_date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    feature_cols = []
    rename = {}
    for col in CBOE_CONTEXT_SOURCE_COLS:
        if col in cboe.columns:
            out_col = f"cboe_{col}"
            rename[col] = out_col
            feature_cols.append(out_col)
    cboe = (
        cboe[["ticker", "date"] + list(rename)]
        .rename(columns=rename)
        .groupby(["ticker", "date"], as_index=False)
        .last()
        .sort_values(["ticker", "date"])
    )
    out = _asof_prior_day_ticker(keys, cboe)
    return out, feature_cols, str(CBOE_OPTIONS_SUMMARY)


def _join_theme_context(keys: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str | None]:
    path = DYNAMIC_THEME_FEATURES_HISTORY if DYNAMIC_THEME_FEATURES_HISTORY.exists() else DYNAMIC_THEME_FEATURES
    if not path.exists():
        return keys.copy(), [], None
    theme = pd.read_parquet(path)
    if theme.empty or not {"ticker", "date"}.issubset(theme.columns):
        return keys.copy(), [], str(path)

    feature_cols = [c for c in THEME_CONTEXT_COLS if c in theme.columns]
    theme = theme[["ticker", "date"] + feature_cols].copy()
    theme["ticker"] = theme["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    theme["date"] = pd.to_datetime(theme["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    theme = (
        theme.dropna(subset=["ticker", "date"])
        .groupby(["ticker", "date"], as_index=False)
        .last()
        .sort_values(["ticker", "date"])
    )
    out = _asof_prior_day_ticker(keys, theme)
    return out, feature_cols, str(path)


def _add_context_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str | None]]:
    if not {"timestamp", "ticker"}.issubset(df.columns):
        raise ValueError("Context enrichment requires timestamp and ticker columns.")

    keys = _daily_context_keys(df)
    calendar, calendar_sources = _join_calendar_macro_features(keys.copy())
    context = calendar[["ticker", "date"] + [c for c in CALENDAR_MACRO_COLS if c in calendar.columns]].copy()
    context_cols = [c for c in CALENDAR_MACRO_COLS if c in context.columns]

    joins = [
        ("news", _join_news_context),
        ("cboe_options", _join_cboe_context),
        ("dynamic_theme", _join_theme_context),
    ]
    sources: dict[str, str | None] = dict(calendar_sources)
    for name, joiner in joins:
        joined, cols, source = joiner(keys)
        sources[name] = source
        if not cols:
            continue
        context = context.merge(joined[["ticker", "date"] + cols], on=["ticker", "date"], how="left")
        context_cols.extend(cols)

    context_cols = [c for c in dict.fromkeys(context_cols) if c in context.columns]
    out = df.copy()
    out["_context_date"] = _norm_date(out["timestamp"])
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    out = out.merge(
        context.rename(columns={"date": "_context_date"}),
        on=["ticker", "_context_date"],
        how="left",
    )
    out = out.drop(columns=["_context_date"])
    for col in context_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out, context_cols, sources


def _build_daily_context_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str | None]]:
    if not {"timestamp", "ticker"}.issubset(df.columns):
        raise ValueError("Context enrichment requires timestamp and ticker columns.")

    keys = _daily_context_keys(df)
    calendar, calendar_sources = _join_calendar_macro_features(keys.copy())
    context = calendar[["ticker", "date"] + [c for c in CALENDAR_MACRO_COLS if c in calendar.columns]].copy()
    context_cols = [c for c in CALENDAR_MACRO_COLS if c in context.columns]

    joins = [
        ("news", _join_news_context),
        ("cboe_options", _join_cboe_context),
        ("dynamic_theme", _join_theme_context),
    ]
    sources: dict[str, str | None] = dict(calendar_sources)
    for name, joiner in joins:
        joined, cols, source = joiner(keys)
        sources[name] = source
        if not cols:
            continue
        context = context.merge(joined[["ticker", "date"] + cols], on=["ticker", "date"], how="left")
        context_cols.extend(cols)

    context_cols = [c for c in dict.fromkeys(context_cols) if c in context.columns]
    for col in context_cols:
        context[col] = pd.to_numeric(context[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    context, context_cols = _prune_context_columns(context, context_cols)
    return context, context_cols, sources


def _prune_context_columns(context: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    keep = []
    for col in cols:
        if col not in context.columns:
            continue
        series = pd.to_numeric(context[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        nonnull = series.dropna()
        if nonnull.empty:
            continue
        if col.startswith("cboe_") and float(series.notna().mean()) < 0.01:
            continue
        if nonnull.nunique(dropna=True) <= 1 and col.startswith("cboe_"):
            continue
        context[col] = series
        keep.append(col)
    return context[["ticker", "date"] + keep], keep


def _processed_frame_for_ticker(ticker: str) -> pd.DataFrame:
    path = PROCESSED_30M_DIR / f"{ticker.upper()}_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr_14",
        "ret_16",
        "market_regime_proxy",
    ]
    available = pd.read_parquet(path, columns=None)
    keep = [c for c in cols if c in available.columns]
    frame = available[keep].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def _build_ticker_location_frame(ticker: str, timestamps: pd.Series) -> pd.DataFrame:
    frame = _processed_frame_for_ticker(ticker)
    if frame.empty:
        return pd.DataFrame()
    frame, daily_cols = add_daily_high_low_gap_features(frame)
    frame, zone_cols, _helpers = add_liquidity_zone_features(
        frame,
        atr_col="atr_14" if "atr_14" in frame.columns else None,
        lookback=78,
        swing_window=20,
        zone_width_pct=0.002,
        volume_window=40,
    )
    cols = [c for c in SWING_LOCATION_FEATURE_COLS if c in set(daily_cols + zone_cols)]
    wanted = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True).dropna().unique()).sort_values()
    frame = frame.reindex(wanted)
    out = frame[cols].reset_index().rename(columns={frame.index.name or "index": "timestamp"})
    if "timestamp" not in out.columns:
        out = out.rename(columns={out.columns[0]: "timestamp"})
    out["ticker"] = ticker.upper()
    return out[["timestamp", "ticker"] + cols]


def _build_spy_location_context(index: pd.DatetimeIndex) -> pd.DataFrame:
    spy = _processed_frame_for_ticker("SPY")
    if spy.empty:
        return pd.DataFrame({"timestamp": index})
    spy, _daily_cols = add_daily_high_low_gap_features(spy, prefix="spy_")
    spy, _zone_cols, _helpers = add_liquidity_zone_features(
        spy,
        prefix="spy_",
        atr_col="atr_14" if "atr_14" in spy.columns else None,
        lookback=78,
        swing_window=20,
        zone_width_pct=0.002,
        volume_window=40,
    )
    if "market_regime_proxy" in spy.columns:
        spy["spy_regime_score"] = pd.to_numeric(spy["market_regime_proxy"], errors="coerce")
    elif "ret_16" in spy.columns:
        ret = pd.to_numeric(spy["ret_16"], errors="coerce")
        spy["spy_regime_score"] = np.where(ret > 0.01, 1.0, np.where(ret < -0.01, -1.0, 0.0))
    spy["spy_distance_to_support"] = spy.get("spy_distance_to_nearest_support_zone")
    spy["spy_distance_to_resistance"] = spy.get("spy_distance_to_nearest_resistance_zone")
    keep = [
        "spy_distance_to_support",
        "spy_distance_to_resistance",
        "spy_inside_support_zone",
        "spy_inside_resistance_zone",
        "spy_regime_score",
    ]
    out = spy.reindex(index)[[c for c in keep if c in spy.columns]].reset_index()
    out = out.rename(columns={out.columns[0]: "timestamp"})
    return out


def _build_bar_location_context_features(key_df: pd.DataFrame, out_path: Path) -> tuple[list[str], dict[str, str | None]]:
    key_df = key_df[["timestamp", "ticker"]].copy()
    key_df["timestamp"] = pd.to_datetime(key_df["timestamp"], utc=True)
    key_df["ticker"] = key_df["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    all_index = pd.DatetimeIndex(key_df["timestamp"].dropna().unique()).sort_values()
    spy_context = _build_spy_location_context(all_index)
    writer: pq.ParquetWriter | None = None
    feature_cols: list[str] = list(SWING_LOCATION_FEATURE_COLS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    active = {col: False for col in feature_cols}
    for ticker, group in key_df.groupby("ticker", sort=True):
        ticker_frame = _build_ticker_location_frame(str(ticker), group["timestamp"])
        if ticker_frame.empty:
            continue
        ticker_frame = ticker_frame.merge(spy_context, on="timestamp", how="left")
        for col in feature_cols:
            if col not in ticker_frame.columns:
                ticker_frame[col] = np.nan
            ticker_frame[col] = pd.to_numeric(ticker_frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            active[col] = active[col] or bool(ticker_frame[col].notna().any())
        table = pa.Table.from_pandas(ticker_frame[["timestamp", "ticker"] + feature_cols], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
    if writer is not None:
        writer.close()
    active_cols = [col for col, has_values in active.items() if has_values]
    return active_cols, {"bar_location_context": str(out_path)}


def export_training_bundle(
    *,
    matrix_path: Path = TRAINING_MATRIX,
    out_dir: Path = EXPORT_DIR,
    enrich_context: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Training matrix not found at {matrix_path}. "
            "Run `python -m multi_ticker_swing.main --stage all ...` first."
        )

    context_feature_columns: list[str] = []
    context_sources: dict[str, str | None] = {}
    context_file: str | None = None
    context_path: Path | None = None
    bar_context_file: str | None = None
    bar_context_path: Path | None = None
    bar_location_feature_columns: list[str] = []
    pruned_feature_columns = sorted(SWING_PRUNED_BASE_FEATURES)
    key_df = pd.read_parquet(matrix_path, columns=["timestamp", "ticker"])
    key_df["timestamp"] = pd.to_datetime(key_df["timestamp"], utc=True)
    if enrich_context:
        logger.info("building daily earnings/news/options/theme context for swing export ...")
        context_df, context_feature_columns, context_sources = _build_daily_context_features(key_df)
        context_file = "daily_context_features.parquet"
        context_path = out_dir / context_file
        context_df.to_parquet(context_path, index=False)
        logger.info("building bar-level high/low/gap/SPY location context for swing export ...")
        bar_context_file = "bar_location_context_features.parquet"
        bar_context_path = out_dir / bar_context_file
        bar_location_feature_columns, bar_sources = _build_bar_location_context_features(key_df, bar_context_path)
        context_sources.update(bar_sources)

    feature_columns = [c for c in FEATURE_COLUMNS if c not in SWING_PRUNED_BASE_FEATURES]
    feature_columns.extend([c for c in context_feature_columns if c not in feature_columns])
    feature_columns.extend([c for c in bar_location_feature_columns if c not in feature_columns])
    matrix_file = "training_matrix_30m.parquet"
    target_path = out_dir / matrix_file
    shutil.copy2(matrix_path, target_path)

    manifest = {
        "matrix_file": matrix_file,
        "context_file": context_file,
        "bar_context_file": bar_context_file,
        "feature_columns": feature_columns,
        "base_feature_count": len([c for c in FEATURE_COLUMNS if c not in SWING_PRUNED_BASE_FEATURES]),
        "pruned_feature_columns": pruned_feature_columns,
        "context_feature_columns": context_feature_columns,
        "bar_location_feature_columns": bar_location_feature_columns,
        "context_sources": context_sources,
        "context_join_policy": (
            "Earnings distances use historical event dates. News, CBOE options, treasury rates, "
            "and dynamic theme context are joined as-of the prior day to avoid same-day leakage."
            if enrich_context
            else None
        ),
        "target_column": "target",
        "sample_weight_column": "sample_weight",
        "timestamp_column": "timestamp",
        "ticker_column": "ticker",
        "n_rows": int(len(key_df)),
        "n_tickers": int(key_df["ticker"].nunique()),
        "date_min": str(key_df["timestamp"].min()),
        "date_max": str(key_df["timestamp"].max()),
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "neutral_weight_factor": NEUTRAL_WEIGHT_FACTOR,
        "oof_n_folds": OOF_N_FOLDS,
        "xgboost_config": XGBOOST_CONFIG,
        "output_artifacts": [
            "swing_model_competition_bundle.tgz",
            "best_model.joblib",
            "best_model_native.txt",
            "model_family_summary.csv",
            "seed_results.csv",
            "feature_stability.csv",
            "top_pick_overlap.csv",
            "eval_metrics.json",
            "competition_meta.json",
            "swing_xgb_model.json",
            "selected_features.txt",
            "p_swing_probs.parquet",
            "meta.json",
        ],
    }
    manifest_path = out_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, default=str, indent=2))

    trainer_src = Path(__file__).parent / "colab" / "swing_train_colab.py"
    trainer_dst = out_dir / trainer_src.name
    shutil.copy2(trainer_src, trainer_dst)
    competition_src = REPO_ROOT / "strategies" / "model_training" / "colab_competition.py"
    competition_dst = out_dir / "colab_competition.py"
    shutil.copy2(competition_src, competition_dst)

    bundle_path = out_dir / ("swing_context_colab_bundle.tgz" if enrich_context else "swing_colab_bundle.tgz")
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(target_path, arcname=target_path.name)
        if context_path is not None:
            tar.add(context_path, arcname=context_path.name)
        if bar_context_path is not None:
            tar.add(bar_context_path, arcname=bar_context_path.name)
        tar.add(manifest_path, arcname=manifest_path.name)
        tar.add(trainer_dst, arcname=trainer_dst.name)
        tar.add(competition_dst, arcname=competition_dst.name)

    logger.info("bundle: %s", bundle_path)
    logger.info("  rows: %d  tickers: %s  features: %d", len(key_df), manifest["n_tickers"], len(feature_columns))
    return bundle_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=TRAINING_MATRIX)
    parser.add_argument("--out", type=Path, default=EXPORT_DIR)
    parser.add_argument("--enrich-context", action="store_true", help="Join earnings/news/options/theme context features.")
    args = parser.parse_args()
    export_training_bundle(matrix_path=args.matrix, out_dir=args.out, enrich_context=args.enrich_context)
