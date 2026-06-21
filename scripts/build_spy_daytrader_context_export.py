from __future__ import annotations

import argparse
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signals.location_features import add_liquidity_zone_features

DEFAULT_DATASET_DIR = REPO_ROOT / "Data/processed/spy/datasets/10min_shift1"
DEFAULT_OPTIONS_PATH = REPO_ROOT / "drive-download-20260613T045727Z-3-001/spy_options_daily_dataset_3y_fixed_greeks.jsonl"
DEFAULT_INTRADAY_PATH = REPO_ROOT / "drive-download-20260613T045727Z-3-001/spy_intraday_data.parquet"
DEFAULT_OI_PATH = REPO_ROOT / "drive-download-20260613T045727Z-3-001/spy_open_interest_5_yr.parquet"
DEFAULT_DEALER_REPORT_DIR = REPO_ROOT / "Data/dealer_positioning/reports"
DEFAULT_OUT_DIR = REPO_ROOT / "Data/processed/spy/training_export"
EASTERN = "America/New_York"


def _frame_with_timestamp(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        out = df.copy()
    else:
        out = df.reset_index()
        first = out.columns[0]
        if first != "timestamp":
            out = out.rename(columns={first: "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def _local_bar_time(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert(EASTERN).dt.tz_localize(None).dt.floor("10min")


def _local_date(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert(EASTERN).dt.tz_localize(None).dt.normalize()


def _load_options_daily(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not path.exists():
        return pd.DataFrame(columns=["bar_date"]), [], []
    opt = pd.read_json(path, lines=True)
    if opt.empty or "date" not in opt.columns:
        return pd.DataFrame(columns=["bar_date"]), [], []

    opt["bar_date"] = pd.to_datetime(opt["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    target_cols = [c for c in opt.columns if c.startswith("target_")]
    raw_feature_cols = [
        c
        for c in opt.columns
        if c not in {"date", "bar_date"} and c not in target_cols and pd.api.types.is_numeric_dtype(opt[c])
    ]
    rename = {c: f"spyopt_{c}" for c in raw_feature_cols}
    target_rename = {c: f"spyopt_{c}" for c in target_cols}
    keep = ["bar_date"] + raw_feature_cols + target_cols
    out = opt[keep].rename(columns={**rename, **target_rename}).drop_duplicates("bar_date", keep="last")

    if {"spyopt_target_open", "spyopt_target_close"}.issubset(out.columns):
        move_pct = (out["spyopt_target_close"] - out["spyopt_target_open"]).abs() / out["spyopt_target_open"].replace(0, np.nan)
        out["spyopt_target_big_move_075pct"] = (move_pct >= 0.0075).astype(float)
        out["spyopt_target_big_move_100pct"] = (move_pct >= 0.0100).astype(float)
        target_rename["target_big_move_075pct"] = "spyopt_target_big_move_075pct"
        target_rename["target_big_move_100pct"] = "spyopt_target_big_move_100pct"

    feature_cols = list(rename.values())
    label_cols = [c for c in list(target_rename.values()) if c in out.columns]
    return out, feature_cols, label_cols


def _load_prior_bid_ask_10m(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        return pd.DataFrame(columns=["bar_time_local"]), []

    cols = ["timestamp", "close", "volume", "bid_volume", "ask_volume"]
    intraday = pd.read_parquet(path, columns=cols)
    if intraday.empty:
        return pd.DataFrame(columns=["bar_time_local"]), []

    ts = pd.to_datetime(intraday["timestamp"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(EASTERN, nonexistent="shift_forward", ambiguous="NaT")
    else:
        ts = ts.dt.tz_convert(EASTERN)
    intraday["bar_time_local"] = ts.dt.tz_localize(None).dt.floor("10min")
    intraday = intraday.dropna(subset=["bar_time_local"])
    agg = (
        intraday.groupby("bar_time_local", as_index=False)
        .agg(
            ba_volume_10m=("volume", "sum"),
            ba_bid_volume_10m=("bid_volume", "sum"),
            ba_ask_volume_10m=("ask_volume", "sum"),
            ba_close_10m=("close", "last"),
        )
        .sort_values("bar_time_local")
    )
    total = (agg["ba_bid_volume_10m"] + agg["ba_ask_volume_10m"]).replace(0, np.nan)
    agg["ba_trade_imbalance_10m"] = (agg["ba_ask_volume_10m"] - agg["ba_bid_volume_10m"]) / total
    agg["bar_date"] = agg["bar_time_local"].dt.normalize()

    base_cols = [
        "ba_volume_10m",
        "ba_bid_volume_10m",
        "ba_ask_volume_10m",
        "ba_trade_imbalance_10m",
        "ba_close_10m",
    ]
    for col in base_cols:
        agg[f"{col}_prev"] = agg.groupby("bar_date")[col].shift(1)
    shifted_imb = agg.groupby("bar_date")["ba_trade_imbalance_10m"].shift(1)
    agg["ba_trade_imbalance_30m_prev"] = (
        shifted_imb.groupby(agg["bar_date"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    shifted_vol = agg.groupby("bar_date")["ba_volume_10m"].shift(1)
    vol_mean = shifted_vol.groupby(agg["bar_date"]).rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    vol_std = shifted_vol.groupby(agg["bar_date"]).rolling(20, min_periods=5).std().reset_index(level=0, drop=True)
    agg["ba_volume_z20_prev"] = (shifted_vol - vol_mean) / vol_std
    agg["ba_close_return_10m_prev"] = agg.groupby("bar_date")["ba_close_10m"].pct_change().groupby(agg["bar_date"]).shift(1)

    feature_cols = [
        "ba_volume_10m_prev",
        "ba_bid_volume_10m_prev",
        "ba_ask_volume_10m_prev",
        "ba_trade_imbalance_10m_prev",
        "ba_trade_imbalance_30m_prev",
        "ba_volume_z20_prev",
        "ba_close_return_10m_prev",
    ]
    return agg[["bar_time_local"] + feature_cols], feature_cols


def _load_plot_frame(dataset_dir: Path) -> pd.DataFrame:
    plot_path = dataset_dir / "plot_frame.parquet"
    plot = _frame_with_timestamp(plot_path)
    rename = {
        "open": "raw_open",
        "high": "raw_high",
        "low": "raw_low",
        "close": "raw_close",
        "volume": "raw_volume",
    }
    return plot.rename(columns=rename)[["timestamp", *rename.values()]]


def _add_spy_liquidity_features(dataset_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    plot = _load_plot_frame(dataset_dir)
    frame = plot.set_index("timestamp").rename(
        columns={
            "raw_open": "open",
            "raw_high": "high",
            "raw_low": "low",
            "raw_close": "close",
            "raw_volume": "volume",
        }
    )
    enriched, feature_cols, helper_cols = add_liquidity_zone_features(
        frame,
        lookback=78,
        swing_window=18,
        zone_width_pct=0.0015,
        volume_window=39,
    )
    cols = ["timestamp"] + feature_cols + helper_cols
    out = enriched.reset_index().rename(columns={enriched.index.name or "index": "timestamp"})
    return out[cols], feature_cols, helper_cols


def _load_oi_wall_features(path: Path, plot: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        return pd.DataFrame(columns=["bar_date"]), []
    oi = pd.read_parquet(path)
    if oi.empty or not {"timestamp", "strike", "right", "open_interest"}.issubset(oi.columns):
        return pd.DataFrame(columns=["bar_date"]), []

    spot_by_date = plot.copy()
    spot_by_date["bar_date"] = _local_date(spot_by_date["timestamp"])
    spot_by_date = (
        spot_by_date.sort_values("timestamp")
        .groupby("bar_date", as_index=False)
        .agg(spot_open=("raw_open", "first"), spot_close=("raw_close", "last"))
    )
    spot_map = spot_by_date.set_index("bar_date")["spot_open"]

    oi = oi.copy()
    oi["bar_date"] = pd.to_datetime(oi["timestamp"], errors="coerce").dt.tz_localize(None).dt.normalize()
    oi["open_interest"] = pd.to_numeric(oi["open_interest"], errors="coerce").fillna(0.0)
    oi["strike"] = pd.to_numeric(oi["strike"], errors="coerce")
    if "days_to_expiration" in oi.columns:
        dte = pd.to_numeric(oi["days_to_expiration"], errors="coerce")
        oi = oi[(dte >= 0) & (dte <= 7)].copy()
    oi = oi[oi["bar_date"].isin(spot_map.index)].dropna(subset=["bar_date", "strike"])
    if oi.empty:
        return pd.DataFrame(columns=["bar_date"]), []

    rows = []
    for dt, group in oi.groupby("bar_date", sort=True):
        spot = float(spot_map.get(dt, np.nan))
        if not np.isfinite(spot):
            continue
        calls = group[group["right"].astype(str).str.upper().str.startswith("C")]
        puts = group[group["right"].astype(str).str.upper().str.startswith("P")]
        calls_above = calls[calls["strike"] >= spot]
        puts_below = puts[puts["strike"] <= spot]
        call_wall = _max_oi_strike(calls_above if not calls_above.empty else calls)
        put_wall = _max_oi_strike(puts_below if not puts_below.empty else puts)
        rows.append(
            {
                "bar_date": dt,
                "oi_call_wall": call_wall[0],
                "oi_call_wall_oi": call_wall[1],
                "oi_put_wall": put_wall[0],
                "oi_put_wall_oi": put_wall[1],
            }
        )
    walls = pd.DataFrame(rows)
    feature_cols = [
        "distance_to_call_wall",
        "distance_to_put_wall",
        "inside_call_wall_zone",
        "inside_put_wall_zone",
        "between_major_walls",
        "dealer_wall_width_pct",
        "dealer_wall_oi_ratio",
    ]
    return walls, feature_cols


def _max_oi_strike(group: pd.DataFrame) -> tuple[float, float]:
    if group.empty:
        return np.nan, np.nan
    idx = group["open_interest"].astype(float).idxmax()
    return float(group.loc[idx, "strike"]), float(group.loc[idx, "open_interest"])


def _load_dealer_level_history(report_dir: Path) -> pd.DataFrame:
    files = sorted(report_dir.glob("SPY_dealer_levels_*.csv")) if report_dir.exists() else []
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "timestamp" not in df.columns:
            continue
        keep = [
            c
            for c in [
                "timestamp",
                "spot",
                "call_wall",
                "put_wall",
                "nearest_magnet",
                "next_magnet_above",
                "next_magnet_below",
                "gamma_flip",
                "total_gex",
                "air_gap_above_score",
                "air_gap_below_score",
            ]
            if c in df.columns
        ]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame(columns=["timestamp"])
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _add_dealer_features(
    matrix: pd.DataFrame,
    *,
    oi_path: Path,
    dealer_report_dir: Path,
    plot: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    out = matrix.copy()
    walls, wall_feature_cols = _load_oi_wall_features(oi_path, plot)
    if not walls.empty:
        out = out.merge(walls, on="bar_date", how="left")
        close = pd.to_numeric(out["raw_close"], errors="coerce").replace(0, np.nan)
        out["distance_to_call_wall"] = (out["oi_call_wall"] - close) / close
        out["distance_to_put_wall"] = (close - out["oi_put_wall"]) / close
        out["inside_call_wall_zone"] = (out["distance_to_call_wall"].abs() <= 0.0015).astype(float)
        out["inside_put_wall_zone"] = (out["distance_to_put_wall"].abs() <= 0.0015).astype(float)
        out["between_major_walls"] = ((out["oi_put_wall"] <= close) & (close <= out["oi_call_wall"])).astype(float)
        out["dealer_wall_width_pct"] = (out["oi_call_wall"] - out["oi_put_wall"]).abs() / close
        out["dealer_wall_oi_ratio"] = out["oi_call_wall_oi"] / out["oi_put_wall_oi"].replace(0, np.nan)

    dealer = _load_dealer_level_history(dealer_report_dir)
    dealer_feature_cols = []
    if not dealer.empty:
        right = dealer.rename(
            columns={
                "call_wall": "live_call_wall",
                "put_wall": "live_put_wall",
                "nearest_magnet": "live_nearest_magnet",
                "gamma_flip": "live_gamma_flip",
            }
        )
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            right.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta(minutes=15),
        ).sort_index()
        close = pd.to_numeric(out["raw_close"], errors="coerce").replace(0, np.nan)
        for level_col, out_col in [
            ("live_gamma_flip", "distance_to_gamma_flip"),
            ("live_nearest_magnet", "distance_to_nearest_magnet"),
        ]:
            if level_col in out.columns:
                out[out_col] = (out[level_col] - close) / close
                dealer_feature_cols.append(out_col)
        if "total_gex" in out.columns:
            out["dealer_position_score"] = np.sign(out["total_gex"]) * np.log1p(out["total_gex"].abs())
            dealer_feature_cols.append("dealer_position_score")

    confluence_cols = _add_liquidity_dealer_confluence(out)
    return out, [c for c in wall_feature_cols + dealer_feature_cols + confluence_cols if c in out.columns]


def _add_liquidity_dealer_confluence(df: pd.DataFrame) -> list[str]:
    close = pd.to_numeric(df["raw_close"], errors="coerce").replace(0, np.nan)
    cols = []
    if {"nearest_support_zone", "oi_put_wall"}.issubset(df.columns):
        dist = (df["nearest_support_zone"] - df["oi_put_wall"]).abs() / close
        df["support_and_putwall_confluence"] = np.exp(-dist / 0.002)
        cols.append("support_and_putwall_confluence")
    if {"nearest_resistance_zone", "oi_call_wall"}.issubset(df.columns):
        dist = (df["nearest_resistance_zone"] - df["oi_call_wall"]).abs() / close
        df["resistance_and_callwall_confluence"] = np.exp(-dist / 0.002)
        cols.append("resistance_and_callwall_confluence")
    cluster_distances = []
    if {"nearest_support_zone", "oi_put_wall"}.issubset(df.columns):
        support_cluster = (df["nearest_support_zone"] + df["oi_put_wall"]) / 2.0
        cluster_distances.append(((close - support_cluster).abs() / close).rename("_support_cluster_dist"))
    if {"nearest_resistance_zone", "oi_call_wall"}.issubset(df.columns):
        resistance_cluster = (df["nearest_resistance_zone"] + df["oi_call_wall"]) / 2.0
        cluster_distances.append(((close - resistance_cluster).abs() / close).rename("_resistance_cluster_dist"))
    if cluster_distances:
        cluster_df = pd.concat(cluster_distances, axis=1)
        df["distance_to_nearest_liquidity_cluster"] = cluster_df.min(axis=1)
        df["liquidity_cluster_strength"] = np.exp(-df["distance_to_nearest_liquidity_cluster"] / 0.002)
        if "support_and_putwall_confluence" in df.columns or "resistance_and_callwall_confluence" in df.columns:
            df["liquidity_cluster_strength"] = pd.concat(
                [
                    df.get("support_and_putwall_confluence", pd.Series(np.nan, index=df.index)),
                    df.get("resistance_and_callwall_confluence", pd.Series(np.nan, index=df.index)),
                    df["liquidity_cluster_strength"],
                ],
                axis=1,
            ).max(axis=1)
        cols.extend(["distance_to_nearest_liquidity_cluster", "liquidity_cluster_strength"])
    return cols


def _prune_feature_columns(matrix: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    pruned: list[str] = []
    fingerprints: dict[int, str] = {}
    for col in feature_cols:
        if col not in matrix.columns:
            pruned.append(col)
            continue
        series = pd.to_numeric(matrix[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        nonnull = series.dropna()
        if nonnull.empty or nonnull.nunique(dropna=True) <= 1:
            pruned.append(col)
            continue
        fp = int(pd.util.hash_pandas_object(series, index=False).sum())
        prior = fingerprints.get(fp)
        if prior is not None and series.equals(pd.to_numeric(matrix[prior], errors="coerce").replace([np.inf, -np.inf], np.nan)):
            pruned.append(col)
            continue
        fingerprints[fp] = col
        kept.append(col)
    return kept, pruned


def build_export(
    *,
    dataset_dir: Path,
    options_path: Path,
    intraday_path: Path,
    oi_path: Path,
    dealer_report_dir: Path,
    out_dir: Path,
) -> Path:
    x_path = dataset_dir / "X_10min_shift1_tree.parquet"
    y_path = dataset_dir / "y.parquet"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing SPY X/y parquet files under {dataset_dir}")

    x = _frame_with_timestamp(x_path)
    y = _frame_with_timestamp(y_path)
    plot = _load_plot_frame(dataset_dir)
    label_cols = [c for c in y.columns if c != "timestamp"]
    matrix = x.merge(y, on="timestamp", how="left", validate="one_to_one")
    matrix = matrix.merge(plot, on="timestamp", how="left", validate="one_to_one")
    matrix["bar_date"] = _local_date(matrix["timestamp"])
    matrix["bar_time_local"] = _local_bar_time(matrix["timestamp"])

    liquidity, liquidity_feature_cols, liquidity_helper_cols = _add_spy_liquidity_features(dataset_dir)
    matrix = matrix.merge(liquidity, on="timestamp", how="left", validate="one_to_one")

    options_daily, option_feature_cols, option_label_cols = _load_options_daily(options_path)
    if not options_daily.empty:
        matrix = matrix.merge(options_daily, on="bar_date", how="left")
        label_cols.extend(option_label_cols)

    bid_ask, bid_ask_feature_cols = _load_prior_bid_ask_10m(intraday_path)
    if not bid_ask.empty:
        matrix = matrix.merge(bid_ask, on="bar_time_local", how="left")

    matrix, dealer_feature_cols = _add_dealer_features(
        matrix,
        oi_path=oi_path,
        dealer_report_dir=dealer_report_dir,
        plot=plot,
    )

    helper_cols = {"bar_date", "bar_time_local"}
    existing_labels = [c for c in dict.fromkeys(label_cols) if c in matrix.columns]
    feature_cols = [
        c
        for c in x.columns
        if c != "timestamp" and c not in helper_cols and pd.api.types.is_numeric_dtype(matrix[c])
    ]
    new_feature_groups = option_feature_cols + bid_ask_feature_cols + liquidity_feature_cols + dealer_feature_cols
    feature_cols.extend([c for c in new_feature_groups if c in matrix.columns and c not in feature_cols])

    for col in feature_cols + existing_labels:
        if col in matrix.columns:
            matrix[col] = pd.to_numeric(matrix[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    feature_cols, pruned_feature_cols = _prune_feature_columns(matrix, feature_cols)

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "spy_daytrader_context_matrix.parquet"
    manifest_path = out_dir / "spy_daytrader_context_manifest.json"
    matrix.to_parquet(matrix_path, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_file": matrix_path.name,
        "base_dataset_dir": str(dataset_dir),
        "options_source": str(options_path) if options_path.exists() else None,
        "intraday_bid_ask_source": str(intraday_path) if intraday_path.exists() else None,
        "open_interest_wall_source": str(oi_path) if oi_path.exists() else None,
        "dealer_report_source_dir": str(dealer_report_dir) if dealer_report_dir.exists() else None,
        "timestamp_column": "timestamp",
        "feature_columns": feature_cols,
        "pruned_feature_columns": pruned_feature_cols,
        "label_columns": existing_labels,
        "option_feature_columns": [c for c in option_feature_cols if c in matrix.columns],
        "bid_ask_feature_columns": [c for c in bid_ask_feature_cols if c in matrix.columns],
        "liquidity_feature_columns": [c for c in liquidity_feature_cols if c in feature_cols],
        "dealer_feature_columns": [c for c in dealer_feature_cols if c in feature_cols],
        "liquidity_helper_columns": liquidity_helper_cols,
        "n_rows": int(len(matrix)),
        "date_min": str(matrix["timestamp"].min()),
        "date_max": str(matrix["timestamp"].max()),
        "join_policy": {
            "options_daily": "Same trading day SPY options/Greeks file joined by local market date; target_* columns are labels only.",
            "bid_ask_intraday": "9-second bid/ask volume aggregated to 10-minute bars, then shifted one bar within each day.",
            "liquidity_zones": "Causal bar-close support/resistance zones from rolling swing highs/lows and high-volume wick rejections.",
            "oi_walls": "Historical call/put wall approximations from 0-7 DTE open-interest strikes; live gamma/magnet features are used when overlapping dealer reports exist.",
        },
        "output_artifacts": [
            "spy_daytrader_<target>_model_competition_bundle.tgz",
            "best_model.joblib",
            "best_model_native.txt",
            "model_family_summary.csv",
            "seed_results.csv",
            "feature_stability.csv",
            "top_pick_overlap.csv",
            "competition_meta.json",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    bundle_path = out_dir / "spy_daytrader_context_colab_bundle.tgz"
    trainer_path = REPO_ROOT / "strategies/spy_intraday/Models/colab/spy_daytrader_train_colab.py"
    competition_util_path = REPO_ROOT / "strategies/model_training/colab_competition.py"
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(matrix_path, arcname=matrix_path.name)
        tar.add(manifest_path, arcname=manifest_path.name)
        if trainer_path.exists():
            tar.add(trainer_path, arcname=trainer_path.name)
        if competition_util_path.exists():
            tar.add(competition_util_path, arcname="colab_competition.py")
    return bundle_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build enriched SPY daytrader Colab export matrix.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--options", type=Path, default=DEFAULT_OPTIONS_PATH)
    parser.add_argument("--intraday", type=Path, default=DEFAULT_INTRADAY_PATH)
    parser.add_argument("--oi", type=Path, default=DEFAULT_OI_PATH)
    parser.add_argument("--dealer-reports", type=Path, default=DEFAULT_DEALER_REPORT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    bundle = build_export(
        dataset_dir=args.dataset_dir.resolve(),
        options_path=args.options.resolve(),
        intraday_path=args.intraday.resolve(),
        oi_path=args.oi.resolve(),
        dealer_report_dir=args.dealer_reports.resolve(),
        out_dir=args.out.resolve(),
    )
    print(f"Wrote {bundle}")


if __name__ == "__main__":
    main()
