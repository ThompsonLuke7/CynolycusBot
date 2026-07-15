"""Build a nightly dealer-positioning swing-potential ranking.

This is a heuristic research artifact, not an execution signal.  It converts
the captured dealer snapshot features into cross-sectional ranks that can later
be validated against forward returns and, if useful, fed into the meta-ranker.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SNAPSHOT_ROOT = REPO / "Data" / "dealer_positioning" / "historical_snapshots"
RANKING_ROOT = REPO / "Data" / "dealer_positioning" / "rankings"
RANKING_HISTORY = RANKING_ROOT / "dealer_swing_rankings_history.parquet"
SCOPE_WEIGHTS = {
    "daily_week": 0.45,
    "through_month": 0.35,
    "two_months": 0.20,
}
ONE_DAY_CHANGE_COLUMNS = [
    "gamma_flip_change_1d",
    "call_wall_change",
    "put_wall_change",
    "magnet_change",
    "vega_wall_change",
    "ceiling_change",
    "floor_change",
]
VELOCITY_CHANGE_COLUMNS = [
    "gamma_flip_velocity_3d",
    "magnet_velocity_3d",
    "callwall_velocity_3d",
]
DIRECTIONAL_CHANGE_WEIGHTS = {
    "gamma_flip_change_1d": 0.15,
    "call_wall_change": 0.15,
    "put_wall_change": 0.10,
    "magnet_change": 0.20,
    "ceiling_change": 0.10,
    "floor_change": 0.10,
    "gamma_flip_velocity_3d": 0.05,
    "magnet_velocity_3d": 0.10,
    "callwall_velocity_3d": 0.05,
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankingResult:
    snapshot_path: Path
    output_path: Path
    latest_path: Path
    history_path: Path
    rows: int
    symbols: int


def latest_snapshot_path(snapshot_root: Path = SNAPSHOT_ROOT) -> Path:
    paths = sorted(Path(snapshot_root).glob("*/dealer_level_summary.parquet"))
    if not paths:
        raise FileNotFoundError(f"no dealer_level_summary.parquet found under {snapshot_root}")
    return paths[-1]


def build_rankings(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["scope"] = work["scope"].astype(str)
    work = work[work["scope"].isin(SCOPE_WEIGHTS)].copy()
    if work.empty:
        return pd.DataFrame()

    for col in [
        "vacuum_score",
        "pinning_score",
        "gamma_density",
        "dealer_imbalance",
        "pct_to_call_wall",
        "pct_to_put_wall",
        "pct_to_magnet",
        "distance_floor_ceiling",
        "gex_concentration_index",
        *ONE_DAY_CHANGE_COLUMNS,
        *VELOCITY_CHANGE_COLUMNS,
    ]:
        if col not in work.columns:
            work[col] = math.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["abs_dealer_imbalance"] = work["dealer_imbalance"].abs()
    work["abs_pct_to_magnet"] = work["pct_to_magnet"].abs()
    work["wall_room"] = work[["pct_to_call_wall", "pct_to_put_wall"]].abs().min(axis=1)
    work["dealer_direction_bias"] = _direction_bias(work)

    parts: list[pd.DataFrame] = []
    for scope, group in work.groupby("scope", sort=False):
        g = group.copy()
        g["vacuum_component"] = _pct_rank(g["vacuum_score"], higher_is_better=True)
        g["pinning_room_component"] = _pct_rank(g["pinning_score"], higher_is_better=True)
        g["sparse_gamma_component"] = _pct_rank(g["gamma_density"], higher_is_better=False)
        g["imbalance_component"] = _pct_rank(g["abs_dealer_imbalance"], higher_is_better=True)
        g["wall_room_component"] = _pct_rank(g["wall_room"], higher_is_better=True)
        g["magnet_room_component"] = _pct_rank(g["abs_pct_to_magnet"], higher_is_better=True)
        g["structure_width_component"] = _pct_rank(g["distance_floor_ceiling"], higher_is_better=True)
        g["cluster_component"] = _pct_rank(g["gex_concentration_index"], higher_is_better=False)
        g["wall_change_component"] = _change_rank(g[["call_wall_change", "put_wall_change"]].abs().mean(axis=1))
        g["magnet_change_component"] = _change_rank(g["magnet_change"].abs())
        g["gamma_flip_change_component"] = _change_rank(g["gamma_flip_change_1d"].abs())
        g["velocity_change_component"] = _change_rank(g[VELOCITY_CHANGE_COLUMNS].abs().mean(axis=1))
        g["floor_ceiling_change_component"] = _change_rank(g[["ceiling_change", "floor_change"]].abs().mean(axis=1))
        g["vega_change_component"] = _change_rank(g["vega_wall_change"].abs())
        directional_parts = []
        for change_col, weight in DIRECTIONAL_CHANGE_WEIGHTS.items():
            directional_parts.append(weight * _signed_change_rank(g[change_col]))
        g["change_direction_bias"] = sum(directional_parts).clip(-1.0, 1.0)
        g["scope_swing_potential_score"] = (
            0.22 * g["vacuum_component"]
            + 0.18 * g["pinning_room_component"]
            + 0.16 * g["sparse_gamma_component"]
            + 0.15 * g["imbalance_component"]
            + 0.11 * g["wall_room_component"]
            + 0.08 * g["magnet_room_component"]
            + 0.06 * g["structure_width_component"]
            + 0.04 * g["cluster_component"]
        )
        g["scope_change_intensity_score"] = (
            0.25 * g["wall_change_component"]
            + 0.20 * g["magnet_change_component"]
            + 0.20 * g["gamma_flip_change_component"]
            + 0.15 * g["velocity_change_component"]
            + 0.10 * g["floor_ceiling_change_component"]
            + 0.10 * g["vega_change_component"]
        )
        parts.append(g)
    scored = pd.concat(parts, ignore_index=True)
    scored["scope_weight"] = scored["scope"].map(SCOPE_WEIGHTS).astype(float)
    scored["weighted_score"] = scored["scope_swing_potential_score"] * scored["scope_weight"]
    scored["weighted_direction_bias"] = scored["dealer_direction_bias"] * scored["scope_weight"]
    scored["weighted_change_score"] = scored["scope_change_intensity_score"] * scored["scope_weight"]
    scored["weighted_change_direction_bias"] = scored["change_direction_bias"] * scored["scope_weight"]

    grouped = scored.groupby("symbol", as_index=False).agg(
        snapshot_date=("snapshot_date", "max"),
        captured_at=("captured_at", "max"),
        scope_count=("scope", "nunique"),
        scope_weight_sum=("scope_weight", "sum"),
        dealer_swing_potential_score=("weighted_score", "sum"),
        dealer_direction_bias=("weighted_direction_bias", "sum"),
        dealer_change_intensity_score=("weighted_change_score", "sum"),
        dealer_change_direction_bias=("weighted_change_direction_bias", "sum"),
        max_scope_score=("scope_swing_potential_score", "max"),
        max_change_scope_score=("scope_change_intensity_score", "max"),
        avg_vacuum_component=("vacuum_component", "mean"),
        avg_sparse_gamma_component=("sparse_gamma_component", "mean"),
        avg_pinning_room_component=("pinning_room_component", "mean"),
        avg_imbalance_component=("imbalance_component", "mean"),
        avg_wall_room_component=("wall_room_component", "mean"),
        avg_magnet_room_component=("magnet_room_component", "mean"),
        avg_wall_change_component=("wall_change_component", "mean"),
        avg_magnet_change_component=("magnet_change_component", "mean"),
        avg_gamma_flip_change_component=("gamma_flip_change_component", "mean"),
        avg_velocity_change_component=("velocity_change_component", "mean"),
        avg_floor_ceiling_change_component=("floor_ceiling_change_component", "mean"),
        avg_vega_change_component=("vega_change_component", "mean"),
        avg_gamma_density=("gamma_density", "mean"),
        avg_dealer_imbalance=("dealer_imbalance", "mean"),
        min_pct_to_magnet=("abs_pct_to_magnet", "min"),
        avg_gamma_flip_change_1d=("gamma_flip_change_1d", "mean"),
        avg_call_wall_change=("call_wall_change", "mean"),
        avg_put_wall_change=("put_wall_change", "mean"),
        avg_magnet_change=("magnet_change", "mean"),
        avg_vega_wall_change=("vega_wall_change", "mean"),
        avg_ceiling_change=("ceiling_change", "mean"),
        avg_floor_change=("floor_change", "mean"),
        avg_gamma_flip_velocity_3d=("gamma_flip_velocity_3d", "mean"),
        avg_magnet_velocity_3d=("magnet_velocity_3d", "mean"),
        avg_callwall_velocity_3d=("callwall_velocity_3d", "mean"),
    )
    grouped["dealer_swing_potential_score"] = grouped["dealer_swing_potential_score"] / grouped["scope_weight_sum"].replace(0, math.nan)
    grouped["dealer_direction_bias"] = grouped["dealer_direction_bias"] / grouped["scope_weight_sum"].replace(0, math.nan)
    grouped["dealer_change_intensity_score"] = grouped["dealer_change_intensity_score"] / grouped["scope_weight_sum"].replace(0, math.nan)
    grouped["dealer_change_direction_bias"] = grouped["dealer_change_direction_bias"] / grouped["scope_weight_sum"].replace(0, math.nan)
    grouped["dealer_swing_potential_score"] = (grouped["dealer_swing_potential_score"] * 100.0).clip(0.0, 100.0)
    grouped["dealer_change_intensity_score"] = (grouped["dealer_change_intensity_score"] * 100.0).clip(0.0, 100.0)
    grouped["dealer_direction"] = grouped["dealer_direction_bias"].map(_direction_label)
    grouped["dealer_change_direction"] = grouped["dealer_change_direction_bias"].map(_direction_label)
    grouped["dealer_swing_rank"] = grouped["dealer_swing_potential_score"].rank(method="first", ascending=False).astype(int)
    grouped["dealer_change_intensity_rank"] = grouped["dealer_change_intensity_score"].rank(method="first", ascending=False).astype(int)
    grouped["dealer_change_bullish_rank"] = grouped["dealer_change_direction_bias"].rank(method="first", ascending=False).astype(int)
    grouped["dealer_change_bearish_rank"] = grouped["dealer_change_direction_bias"].rank(method="first", ascending=True).astype(int)
    grouped = grouped.sort_values(["dealer_swing_rank", "symbol"]).reset_index(drop=True)

    scope_payload = scored[
        [
            "symbol",
            "scope",
            "scope_swing_potential_score",
            "dealer_direction_bias",
            "vacuum_component",
            "sparse_gamma_component",
            "pinning_room_component",
            "imbalance_component",
            "wall_room_component",
            "magnet_room_component",
            "scope_change_intensity_score",
            "change_direction_bias",
            "wall_change_component",
            "magnet_change_component",
            "gamma_flip_change_component",
            "velocity_change_component",
            "floor_ceiling_change_component",
            "vega_change_component",
        ]
    ].copy()
    scope_payload["scope_swing_potential_score"] = scope_payload["scope_swing_potential_score"] * 100.0
    scope_payload["scope_change_intensity_score"] = scope_payload["scope_change_intensity_score"] * 100.0
    scope_json = {
        symbol: json.dumps(group.drop(columns=["symbol"]).to_dict(orient="records"), sort_keys=True)
        for symbol, group in scope_payload.groupby("symbol")
    }
    grouped["scope_scores_json"] = grouped["symbol"].map(scope_json)
    return grouped


def run(
    *,
    snapshot_path: Path | None = None,
    snapshot_root: Path = SNAPSHOT_ROOT,
    output_root: Path = RANKING_ROOT,
) -> RankingResult:
    path = Path(snapshot_path) if snapshot_path is not None else latest_snapshot_path(snapshot_root)
    frame = pd.read_parquet(path)
    rankings = build_rankings(frame)
    if rankings.empty:
        raise RuntimeError(f"no rankings built from {path}")

    snapshot_date = str(rankings["snapshot_date"].dropna().iloc[0]) if "snapshot_date" in rankings else path.parent.name
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"dealer_swing_rankings_{snapshot_date.replace('-', '')}.parquet"
    latest_path = output_root / "dealer_swing_rankings_latest.parquet"
    rankings.to_parquet(output_path, index=False)
    rankings.to_parquet(latest_path, index=False)
    history_path = rebuild_rankings_history(output_root)
    return RankingResult(
        snapshot_path=path,
        output_path=output_path,
        latest_path=latest_path,
        history_path=history_path,
        rows=len(rankings),
        symbols=int(rankings["symbol"].nunique()),
    )


def rebuild_rankings_history(output_root: Path = RANKING_ROOT) -> Path:
    output_root = Path(output_root)
    paths = sorted(output_root.glob("dealer_swing_rankings_*.parquet"))
    dated_paths = [
        path
        for path in paths
        if path.name not in {"dealer_swing_rankings_latest.parquet", "dealer_swing_rankings_history.parquet"}
    ]
    frames: list[pd.DataFrame] = []
    for path in dated_paths:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping unreadable ranking file %s: %s", path, exc)
            continue
        if frame.empty:
            continue
        frame = frame.copy()
        frame["ranking_source_path"] = str(path)
        frames.append(frame)
    history_path = output_root / "dealer_swing_rankings_history.parquet"
    if not frames:
        pd.DataFrame().to_parquet(history_path, index=False)
        return history_path
    history = pd.concat(frames, ignore_index=True, sort=False)
    if {"snapshot_date", "symbol"}.issubset(history.columns):
        history = history.drop_duplicates(["snapshot_date", "symbol"], keep="last")
        history = history.sort_values(["snapshot_date", "dealer_swing_rank", "symbol"]).reset_index(drop=True)
    history.to_parquet(history_path, index=False)
    return history_path


def _direction_bias(frame: pd.DataFrame) -> pd.Series:
    magnet_pull = frame["pct_to_magnet"].clip(-0.15, 0.15) / 0.15
    call_room = frame["pct_to_call_wall"].abs()
    put_room = frame["pct_to_put_wall"].abs()
    wall_room_bias = ((call_room - put_room) / (call_room + put_room).replace(0, math.nan)).clip(-1.0, 1.0)
    imbalance = frame["dealer_imbalance"].clip(-1.0, 1.0)
    out = 0.50 * magnet_pull.fillna(0.0) + 0.30 * wall_room_bias.fillna(0.0) + 0.20 * imbalance.fillna(0.0)
    return out.clip(-1.0, 1.0)


def _direction_label(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if val >= 0.20:
        return "bullish"
    if val <= -0.20:
        return "bearish"
    return "neutral"


def _pct_rank(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    ranked = clean.rank(pct=True, ascending=higher_is_better, na_option="keep")
    if not higher_is_better:
        ranked = (-clean).rank(pct=True, ascending=True, na_option="keep")
    return ranked.fillna(0.50).clip(0.0, 1.0)


def _change_rank(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    non_zero = clean.dropna().abs()
    if non_zero.empty or float(non_zero.max()) == 0.0:
        return pd.Series(0.0, index=series.index)
    return clean.rank(pct=True, ascending=True, na_option="keep").fillna(0.0).clip(0.0, 1.0)


def _signed_change_rank(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    non_zero = clean.dropna()
    if non_zero.empty or float(non_zero.abs().max()) == 0.0 or int(non_zero.nunique()) <= 1:
        return pd.Series(0.0, index=series.index)
    return (2.0 * clean.rank(pct=True, ascending=True, na_option="keep") - 1.0).fillna(0.0).clip(-1.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dealer-positioning swing-potential rankings.")
    parser.add_argument("--snapshot-path", default=None)
    parser.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT))
    parser.add_argument("--output-root", default=str(RANKING_ROOT))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    result = run(
        snapshot_path=Path(args.snapshot_path) if args.snapshot_path else None,
        snapshot_root=Path(args.snapshot_root),
        output_root=Path(args.output_root),
    )
    print(
        "dealer ranking complete: "
        f"symbols={result.symbols} rows={result.rows} output={result.output_path} latest={result.latest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
