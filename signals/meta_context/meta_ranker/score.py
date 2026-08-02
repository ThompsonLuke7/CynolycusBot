"""
Meta Ranker live scorer.

Loads the shipped native XGBoost boosters (CPU inference) for the two trained
labels and produces per-4H-bar ranked picks. Confluence ranking (the average of
each model's within-bar percentile rank) is the recommended live signal: it
beats either model standalone on both raw and risk-adjusted forward return.

  - upside   (relevance=meta_upside, target=fwd_max_return): bigger raw winners.
  - quality  (relevance=meta_good,   target=trade_quality):  cleaner, low-drawdown winners.
  - combo    = mean(rank_pct(upside), rank_pct(quality)) per timestamp  [DEFAULT]

Both boosters were trained on GPU but the native .json booster predicts on CPU.

Run:
  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/score.py --top-k 20
  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/score.py --model upside --latest-only
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
DEFAULT_MATRIX = HERE / "meta_ranker_matrix.parquet"
LABELS = ("upside", "quality")
ID_COLS = ["timestamp", "ticker", "theme"]
BoosterLoader = Callable[[str], tuple[Any, list[str]]]


def _normalize_timestamp_series(values: pd.Series, *, field_name: str) -> pd.Series:
    normalized: list[pd.Timestamp] = []
    for value in values.tolist():
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} contains an invalid timestamp") from exc
        if pd.isna(timestamp) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(f"{field_name} must contain only timezone-aware timestamps")
        normalized.append(timestamp.tz_convert("UTC"))
    return pd.Series(normalized, index=values.index, name=values.name)


def _load_booster(label: str) -> tuple[xgb.Booster, list[str]]:
    mdir = MODELS_DIR / label
    feats = json.loads((mdir / "meta.json").read_text())["feature_names"]
    bst = xgb.Booster()
    bst.load_model(str(mdir / "meta_ranker_xgb.json"))
    return bst, feats


def score_frame(
    df: pd.DataFrame,
    labels=LABELS,
    *,
    booster_loader: BoosterLoader | None = None,
) -> pd.DataFrame:
    """Add per-label score columns (s_<label>) to a meta-ranker matrix frame."""
    out = df.copy()
    if "timestamp" not in out.columns:
        raise KeyError("matrix requires a timestamp column")
    out["timestamp"] = _normalize_timestamp_series(out["timestamp"], field_name="feature-matrix timestamps")
    load_booster = booster_loader or _load_booster
    for label in labels:
        bst, feats = load_booster(label)
        missing = [f for f in feats if f not in out.columns]
        if missing:
            raise KeyError(f"{label}: matrix missing {len(missing)} features e.g. {missing[:5]}")
        dm = xgb.DMatrix(out[feats].astype("float32").values, feature_names=feats)
        out[f"s_{label}"] = bst.predict(dm)
    if set(labels) == set(LABELS):
        ranks: dict[str, pd.Series] = {}
        for label in LABELS:
            score_col = f"s_{label}"
            finite = np.isfinite(out[score_col].to_numpy(dtype=float))
            valid = out.loc[finite, ["timestamp", score_col]]
            ranks[label] = valid.groupby("timestamp")[score_col].rank(pct=True)
        ru = pd.Series(np.nan, index=out.index, dtype=float)
        rq = pd.Series(np.nan, index=out.index, dtype=float)
        ru.loc[ranks["upside"].index] = ranks["upside"]
        rq.loc[ranks["quality"].index] = ranks["quality"]
        out["s_combo"] = (ru + rq) / 2.0
    return out


def rank_picks(df: pd.DataFrame, score_col: str, top_k: int) -> pd.DataFrame:
    keep = [c for c in ID_COLS if c in df.columns] + [c for c in df.columns if c.startswith("s_")]
    picks = (
        df.sort_values(score_col, ascending=False)
        .groupby("timestamp", group_keys=False)
        .head(top_k)
        .sort_values(["timestamp", score_col], ascending=[True, False])
    )
    picks["rank"] = picks.groupby("timestamp")[score_col].rank(ascending=False, method="first").astype(int)
    return picks[keep + ["rank"]]


def main():
    ap = argparse.ArgumentParser(description="Score the Meta Ranker matrix and emit ranked picks.")
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--model", choices=["combo", "upside", "quality"], default="combo")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--latest-only", action="store_true", help="Only score the most recent bar in the matrix.")
    ap.add_argument("--out", default=None, help="Optional parquet/csv output path for ranked picks.")
    args = ap.parse_args()

    df = pd.read_parquet(args.matrix).reset_index()
    df["timestamp"] = _normalize_timestamp_series(df["timestamp"], field_name="feature-matrix timestamps")
    if args.latest_only:
        # Skip degenerate edge bars (a handful of rows) — take the latest fully-populated bar.
        counts = df.groupby("timestamp").size()
        full_bars = counts[counts >= max(50, int(0.25 * counts.max()))].index
        latest_full = full_bars.max()
        df = df[df["timestamp"] == latest_full].copy()

    scored = score_frame(df)
    score_col = f"s_{args.model}"
    picks = rank_picks(scored, score_col, args.top_k)

    latest_ts = scored["timestamp"].max()
    print(f"matrix bars={scored['timestamp'].nunique()}  latest={latest_ts}  model={args.model}  top_k={args.top_k}")
    show = picks[picks["timestamp"] == latest_ts]
    cols = [c for c in ["rank", "ticker", "theme", "s_combo", "s_upside", "s_quality"] if c in show.columns]
    print(f"\n--- top {args.top_k} for {latest_ts} ---")
    print(show[cols].to_string(index=False))

    if args.out:
        if args.out.endswith(".csv"):
            picks.to_csv(args.out, index=False)
        else:
            picks.to_parquet(args.out, index=False)
        print(f"\nwrote {len(picks):,} ranked picks -> {args.out}")


if __name__ == "__main__":
    main()
