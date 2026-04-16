from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from API.Alpaca_API.inference.live_inference import (  # noqa: E402
    _LiveMulticlassXGBArtifact,
    LiveIndependentMetaXGBAgent,
    build_tree_feature_frame_from_1m,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare live-built swing setup features/probs against saved swing probability artifacts."
    )
    parser.add_argument("--data-path", default="Data/raw/spy/1m_train.parquet")
    parser.add_argument("--model-dir", default="Data/models/ga_xgboost/10min/single/swing_support_single")
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-04-02")
    parser.add_argument("--warmup-start", default=None)
    parser.add_argument("--tz", default="America/New_York")
    parser.add_argument(
        "--actual-live-agent",
        action="store_true",
        help="Use LiveIndependentMetaXGBAgent._build_independent_base_frame instead of tree-only features.",
    )
    parser.add_argument("--meta-model-root", default="Data/models/meta_xgboost/10min")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _load_1m(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = next((col for col in ("timestamp", "date", "datetime") if col in df.columns), None)
        if ts_col is None:
            raise ValueError(f"{path} has no DatetimeIndex or timestamp column")
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).set_index(ts_col)
    return df.sort_index()


def _localize(ts: str, tz: str) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize(tz)
    return out


def _pick_saved_col(frame: pd.DataFrame, side: str) -> str:
    for col in (f"p_{side}_full", f"p_{side}_test", f"p_{side}_oof_train"):
        if col in frame.columns:
            return col
    raise KeyError(f"No saved probability column for side={side}")


def main() -> None:
    args = _parse_args()
    model_dir = REPO_ROOT / args.model_dir
    data_path = REPO_ROOT / args.data_path
    start = _localize(args.start, args.tz)
    end = _localize(args.end, args.tz)
    warmup_start = _localize(args.warmup_start, args.tz) if args.warmup_start else None

    df_1m = _load_1m(data_path)
    if warmup_start is not None:
        df_1m = df_1m.loc[df_1m.index >= warmup_start]
    df_1m = df_1m.loc[df_1m.index < end]
    print(f"1m rows={len(df_1m):,} range={df_1m.index.min()}..{df_1m.index.max()} path={data_path}")

    artifact = _LiveMulticlassXGBArtifact(model_dir)
    if args.actual_live_agent:
        agent = LiveIndependentMetaXGBAgent(
            model_root=REPO_ROOT / args.meta_model_root,
            ga_model_root=None,
            ga_feature_list_path=None,
            include_pivot_probs=False,
            include_tb_probs=False,
            include_vix_features=True,
            tz=args.tz,
            assume_tz="UTC",
            min_15m_bars=20,
            fill_missing_prob=0.0,
            resample_label="left",
            resample_closed="left",
            label_timeframe_rule="10min",
            entry_prob_source="swing_support_single",
            swing_setup_single_model_dir=model_dir,
            swing_setup_probs_frame=None,
        )
        live_features = agent._build_independent_base_frame(df_1m=df_1m).sort_index()
    else:
        live_features = build_tree_feature_frame_from_1m(
            df_1m,
            label_timeframe="10min",
            resample_label="left",
            resample_closed="left",
            tz=args.tz,
            include_vix_features=True,
        ).sort_index()
    live_features = live_features.loc[(live_features.index >= start) & (live_features.index < end)]
    print(
        f"live feature rows={len(live_features):,} cols={len(live_features.columns):,} "
        f"range={live_features.index.min()}..{live_features.index.max()}"
    )

    missing_cols = [col for col in artifact.feature_cols if col not in live_features.columns]
    aligned = live_features.reindex(columns=artifact.feature_cols)
    nan_rate = aligned.isna().mean().sort_values(ascending=False)
    all_nan_cols = nan_rate[nan_rate >= 1.0].index.tolist()
    high_nan_cols = nan_rate[(nan_rate >= 0.25) & (nan_rate < 1.0)].head(25)
    print(f"selected_features={len(artifact.feature_cols):,} missing_cols={len(missing_cols):,} all_nan_cols={len(all_nan_cols):,}")
    if missing_cols:
        print("missing preview:", ", ".join(missing_cols[:30]))
    if all_nan_cols:
        print("all-NaN preview:", ", ".join(all_nan_cols[:30]))
    if not high_nan_cols.empty:
        print("high NaN selected feature rates:")
        print(high_nan_cols.to_string())

    if args.actual_live_agent and {"p_swing_setup_short", "p_swing_setup_neutral", "p_swing_setup_long"}.issubset(
        live_features.columns
    ):
        live_probs = pd.DataFrame(
            {
                "short": pd.to_numeric(live_features["p_swing_setup_short"], errors="coerce"),
                "neutral": pd.to_numeric(live_features["p_swing_setup_neutral"], errors="coerce"),
                "long": pd.to_numeric(live_features["p_swing_setup_long"], errors="coerce"),
            },
            index=live_features.index,
        )
    else:
        live_probs = artifact.predict_frame(live_features)
    probs_path = model_dir / "p_swing_probs.parquet"
    saved = pd.read_parquet(probs_path).sort_index()
    if saved.index.tz is None:
        saved.index = saved.index.tz_localize(args.tz)
    else:
        saved.index = saved.index.tz_convert(args.tz)
    saved = saved.loc[(saved.index >= start) & (saved.index < end)]

    compare = pd.DataFrame(index=live_probs.index)
    compare["live_short"] = pd.to_numeric(live_probs.get("short"), errors="coerce")
    compare["live_neutral"] = pd.to_numeric(live_probs.get("neutral"), errors="coerce")
    compare["live_long"] = pd.to_numeric(live_probs.get("long"), errors="coerce")
    compare["saved_short"] = pd.to_numeric(saved[_pick_saved_col(saved, "short")].reindex(compare.index), errors="coerce")
    compare["saved_neutral"] = pd.to_numeric(saved[_pick_saved_col(saved, "neutral")].reindex(compare.index), errors="coerce")
    compare["saved_long"] = pd.to_numeric(saved[_pick_saved_col(saved, "long")].reindex(compare.index), errors="coerce")
    compare = compare.dropna(subset=["live_short", "saved_short", "live_long", "saved_long"])
    compare["short_delta"] = compare["live_short"] - compare["saved_short"]
    compare["long_delta"] = compare["live_long"] - compare["saved_long"]
    compare["neutral_delta"] = compare["live_neutral"] - compare["saved_neutral"]
    compare["abs_short_delta"] = compare["short_delta"].abs()
    compare["abs_long_delta"] = compare["long_delta"].abs()

    print(f"comparison rows={len(compare):,} saved_path={probs_path}")
    if compare.empty:
        return
    summary = compare[
        [
            "live_short",
            "saved_short",
            "short_delta",
            "live_neutral",
            "saved_neutral",
            "neutral_delta",
            "live_long",
            "saved_long",
            "long_delta",
        ]
    ].agg(["mean", "median", "min", "max"])
    print(summary.to_string())
    print("\nlargest short mismatches:")
    print(
        compare.sort_values("abs_short_delta", ascending=False)
        .head(20)[
            [
                "live_short",
                "saved_short",
                "short_delta",
                "live_neutral",
                "saved_neutral",
                "live_long",
                "saved_long",
                "long_delta",
            ]
        ]
        .to_string()
    )

    if args.out:
        out_path = REPO_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        compare.to_csv(out_path, index_label="timestamp")
        print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
