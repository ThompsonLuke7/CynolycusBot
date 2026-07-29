"""Compute snapshot-over-snapshot *change* features for dealer positioning levels.

Workstream C of docs/superpowers/plans/2026-07-26-market-regime-and-sector-context.md.

`strategies/dealer_positioning/levels.py` already computes and
`scripts/capture_historical_snapshots.py` already persists *static* per-date
dealer-positioning levels (GEX ladder, walls, magnets, gamma flip, DTE-bucket
levels) to
``Data/dealer_positioning/historical_snapshots/<YYYYMMDD>/dealer_level_summary.parquet``
and ``dealer_strike_ladder.parquet``. This module does not recompute those
levels -- it reads the persisted history and derives how they *change* across
snapshot dates: wall drift, gamma-flip velocity, level persistence, and
open-interest/volume/IV-skew deltas.

IMPORTANT -- naming and interpretation
---------------------------------------
Every feature here is an *inferred change in dealer positioning* derived from
end-of-day-ish open-interest, volume, and Greeks snapshots. Nothing in this
module observes an executed dealer trade, a signed order, or any actual buy
or sell. Open interest can rise or fall for reasons unrelated to dealer
hedging (customer rolls, assignment, new listings). Treat every "change"
feature as a positioning-state delta, not a trade signal.

Time-correctness contract
--------------------------
Every output row carries both ``snapshot_date`` (the calendar session the row
describes) and ``available_at`` (the snapshot's own ``captured_at`` --
generally ~15:45 ET the same session, but see the backfill caveat below).
A consumer must join with ``available_at <= decision_timestamp`` and must
never use a row before its own ``available_at``.

Because dealer_level_summary snapshots are captured intraday (~15:45 ET,
before the 16:00 ET session close), the *current* session's own daily OHLC
bar is not final yet at capture time. ``distance_to_call_wall_atr`` and
``distance_to_put_wall_atr`` therefore use the 14-day ATR computed through
the last daily bar strictly BEFORE ``snapshot_date`` -- using the same-day
bar would be a look-ahead leak.

Gap handling (no silent forward-fill)
--------------------------------------
Some calendar dates under ``historical_snapshots/`` have no snapshot at all
(directory absent) or an empty capture (directory present, zero rows -- every
symbol/scope errored that day; both cases occur in the real history: no
directory for 2026-07-03, and empty captures on 2026-07-07 and 2026-07-22).
All "change" and "velocity" features are computed against the actual prior
*available* row for that (symbol, scope) pair -- found by sorting each
group's real snapshot dates and taking the previous one, never by shifting a
fixed number of calendar days and never by forward-filling a missing date's
values. Every change feature has a companion ``*_gap_days`` (calendar days
between the two compared snapshots) and the row carries
``prior_gap_stale`` / ``third_gap_stale`` flags
(``gap_days > STALE_GAP_DAYS``, where ``STALE_GAP_DAYS`` = 4 tolerates a
normal 3-day weekend but flags holidays, multi-day outages, or a missing
capture). A row with no prior snapshot at all has NaN changes, not zero.

DTE dimension caveat
---------------------
The persisted strike ladder has no per-contract DTE column -- only ``scope``
(``daily_week`` / ``through_month`` / ``two_months``), a coarse expiry-bucket
dimension. ``oi_change_by_dte`` is therefore computed *per scope*, i.e. it is
an oi-change-by-expiry-bucket feature, not a true per-DTE feature. Scopes are
never mixed within a single output row.

Sample size (plan defect D5)
------------------------------
As of this build there are only 14 usable snapshot dates (16 directories
minus 2 empty captures). This is far too small to validate any of these
features. This module makes NO performance, predictive, or edge claim. It
ships the computation so history can accumulate; do not treat any example
number quoted in a report derived from this module as evidence of a tradable
signal.

Outputs
-------
``build(...)`` returns two dataframes:

* summary -- one row per (symbol, scope, snapshot_date): wall_change_1d/3d,
  gex_concentration_change, gamma_flip_velocity, distance_to_call_wall_atr,
  distance_to_put_wall_atr, level_stability_days, oi_change_by_dte,
  volume_to_prior_oi, iv_skew_change, near_level_option_volume_share, plus
  freshness/availability columns.
* by_strike -- one row per (symbol, scope, snapshot_date, strike):
  call_oi_change, put_oi_change, oi_change_by_strike (= call + put),
  volume_to_prior_oi (per-strike), plus availability columns. NaN where the
  strike did not exist in the prior available snapshot (new strike, or no
  prior snapshot at all) -- never treated as a zero-OI prior.

CLI: ``python -m strategies.dealer_positioning.scripts.build_level_dynamics``
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from strategies.momentum_expansion.data.load_bars import load_1d  # noqa: E402

SNAPSHOT_ROOT = REPO / "Data" / "dealer_positioning" / "historical_snapshots"
OUTPUT_ROOT = REPO / "Data" / "dealer_positioning" / "level_dynamics"
SCOPES = ("daily_week", "through_month", "two_months")

# A snapshot-to-snapshot gap longer than this many calendar days is flagged
# stale. 4 tolerates a normal Fri->Mon (3-day) weekend but flags a holiday
# week, a multi-day outage, or a missed capture.
STALE_GAP_DAYS = 4

ATR_LENGTH = 14
# Dollar tolerance for "near a dealer level": one strike-ish or 0.25% of
# spot, whichever is larger -- mirrors the existing air-gap tolerance
# convention in strategies/dealer_positioning/levels.py (_air_gap_score).
NEAR_LEVEL_MIN_BAND = 0.50
NEAR_LEVEL_PCT_BAND = 0.0025

SUMMARY_COLUMNS = [
    "captured_at",
    "snapshot_date",
    "symbol",
    "scope",
    "spot",
    "call_wall",
    "put_wall",
    "gamma_flip",
    "nearest_magnet",
    "gex_concentration_index",
    "total_option_oi",
    "total_option_volume",
]

LADDER_COLUMNS = [
    "captured_at",
    "snapshot_date",
    "symbol",
    "scope",
    "spot",
    "strike",
    "call_oi",
    "put_oi",
    "call_volume",
    "put_volume",
    "call_iv",
    "put_iv",
    "call_delta",
    "put_delta",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LevelDynamicsResult:
    summary_path: Path
    by_strike_path: Path
    summary_rows: int
    by_strike_rows: int
    symbols: int
    snapshot_dates: int


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _iter_snapshot_dirs(snapshot_root: Path) -> list[Path]:
    if not snapshot_root.exists():
        return []
    return sorted(p for p in snapshot_root.iterdir() if p.is_dir())


def _load_history(snapshot_root: Path, filename: str, columns: list[str]) -> pd.DataFrame:
    """Concatenate one parquet file across every snapshot date directory.

    Skips directories with a missing, unreadable, or empty (zero-row)
    parquet -- an empty file is a real capture failure (see 2026-07-07 and
    2026-07-22 in the real history), not a value to fabricate or fill.
    """
    frames: list[pd.DataFrame] = []
    for day_dir in _iter_snapshot_dirs(snapshot_root):
        path = day_dir / filename
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - one bad file should not fail the whole build
            logger.warning("skipping unreadable %s: %s", path, exc)
            continue
        if frame.empty:
            continue
        keep = [c for c in columns if c in frame.columns]
        frames.append(frame[keep].copy())
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_snapshot_history(snapshot_root: Path = SNAPSHOT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and lightly normalize the full persisted snapshot history."""
    summary = _load_history(snapshot_root, "dealer_level_summary.parquet", SUMMARY_COLUMNS)
    ladder = _load_history(snapshot_root, "dealer_strike_ladder.parquet", LADDER_COLUMNS)
    summary = _normalize(summary)
    ladder = _normalize(ladder)
    if not summary.empty:
        summary = summary.drop_duplicates(["symbol", "scope", "snapshot_date"], keep="last")
    if not ladder.empty:
        ladder = ladder.drop_duplicates(["symbol", "scope", "snapshot_date", "strike"], keep="last")
    return summary, ladder


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["scope"] = frame["scope"].astype(str)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], errors="coerce", utc=True)
    return frame.dropna(subset=["snapshot_date", "symbol", "scope"])


# ---------------------------------------------------------------------------
# ATR (as of the last daily bar strictly before snapshot_date)
# ---------------------------------------------------------------------------

def _atr_series(df: pd.DataFrame, length: int = ATR_LENGTH) -> pd.Series:
    """Standard rolling-mean true-range ATR. Mirrors the local ATR helper
    convention used elsewhere in this repo (e.g.
    strategies/momentum_expansion/features/feature_matrix_4h.py::_atr,
    strategies/momentum_expansion/backtest/simulate.py::_atr) -- there is no
    shared/importable ATR utility in this codebase; every module defines its
    own small copy of the same formula, and this module follows that
    established pattern rather than introducing a new cross-module import.
    """
    high, low, prior_close = df["high"], df["low"], df["close"].shift(1)
    true_range = pd.concat(
        [(high - low), (high - prior_close).abs(), (low - prior_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(length).mean()


def _build_atr_lookup(summary: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float | None]:
    """One ATR-history load per symbol, reused across every snapshot_date
    and scope for that symbol (avoids re-reading the same 1d parquet)."""
    lookup: dict[tuple[str, pd.Timestamp], float | None] = {}
    for symbol in sorted(summary["symbol"].unique()):
        try:
            bars = load_1d(symbol)
        except Exception:
            bars = pd.DataFrame()
        dates = sorted(summary.loc[summary["symbol"] == symbol, "snapshot_date"].unique())
        for as_of in dates:
            key = (symbol, pd.Timestamp(as_of))
            if bars.empty:
                lookup[key] = None
                continue
            prior = bars[bars.index.normalize() < pd.Timestamp(as_of).tz_localize("UTC")]
            if prior.empty:
                lookup[key] = None
                continue
            atr = _atr_series(prior).iloc[-1]
            lookup[key] = float(atr) if pd.notna(atr) else None
    return lookup


# ---------------------------------------------------------------------------
# Summary-level dynamics
# ---------------------------------------------------------------------------

_PRIOR_ROW_COLUMNS = ["snapshot_date", "call_wall", "put_wall", "gamma_flip", "gex_concentration_index", "total_option_oi"]


def _prior_rows(work: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Shift each (symbol, scope) group by `periods` *available* rows (not
    calendar days) -- this is the "actual prior available snapshot" rule: a
    missing calendar date contributes no row at all to `work` (it was sorted
    from whatever snapshot directories actually existed and were non-empty),
    so shift() naturally lands on the true prior available snapshot instead
    of forward-filling across the gap. Uses the groupby-native `.shift()`
    (not `.apply()`) so the result stays aligned to `work`'s original row
    order and index -- `.apply()` on a grouped frame is not guaranteed to
    preserve row order once groups are re-concatenated."""
    return work.groupby(["symbol", "scope"], sort=False)[_PRIOR_ROW_COLUMNS].shift(periods)


def compute_summary_dynamics(
    summary: pd.DataFrame,
    ladder: pd.DataFrame,
    *,
    atr_lookup: dict[tuple[str, pd.Timestamp], float | None] | None = None,
) -> pd.DataFrame:
    if summary.empty:
        return summary.assign(
            **{
                col: pd.Series(dtype="float64")
                for col in [
                    "wall_change_1d",
                    "wall_change_1d_gap_days",
                    "wall_change_3d",
                    "wall_change_3d_gap_days",
                    "gex_concentration_change",
                    "gamma_flip_velocity",
                    "distance_to_call_wall_atr",
                    "distance_to_put_wall_atr",
                    "level_stability_days",
                    "oi_change_by_dte",
                    "volume_to_prior_oi",
                    "iv_skew_change",
                    "near_level_option_volume_share",
                ]
            }
        )

    work = summary.sort_values(["symbol", "scope", "snapshot_date"]).reset_index(drop=True)

    prior1 = _prior_rows(work, 1)
    prior3 = _prior_rows(work, 3)

    work["prior_snapshot_date"] = prior1["snapshot_date"]
    work["prior_gap_days"] = (work["snapshot_date"] - work["prior_snapshot_date"]).dt.days
    work["has_prior_snapshot_1d"] = work["prior_snapshot_date"].notna()
    work["prior_gap_stale"] = work["prior_gap_days"].isna() | (work["prior_gap_days"] > STALE_GAP_DAYS)

    work["third_snapshot_date"] = prior3["snapshot_date"]
    work["third_gap_days"] = (work["snapshot_date"] - work["third_snapshot_date"]).dt.days
    work["has_prior_snapshot_3d"] = work["third_snapshot_date"].notna()
    work["third_gap_stale"] = work["third_gap_days"].isna() | (work["third_gap_days"] > STALE_GAP_DAYS)

    call_wall_change_1d = (work["call_wall"] - prior1["call_wall"]).abs()
    put_wall_change_1d = (work["put_wall"] - prior1["put_wall"]).abs()
    work["wall_change_1d"] = pd.concat([call_wall_change_1d, put_wall_change_1d], axis=1).mean(axis=1, skipna=True)
    work["wall_change_1d_gap_days"] = work["prior_gap_days"]

    call_wall_change_3d = (work["call_wall"] - prior3["call_wall"]).abs()
    put_wall_change_3d = (work["put_wall"] - prior3["put_wall"]).abs()
    work["wall_change_3d"] = pd.concat([call_wall_change_3d, put_wall_change_3d], axis=1).mean(axis=1, skipna=True)
    work["wall_change_3d_gap_days"] = work["third_gap_days"]

    work["gex_concentration_change"] = work["gex_concentration_index"] - prior1["gex_concentration_index"]

    gamma_flip_diff = work["gamma_flip"] - prior1["gamma_flip"]
    safe_gap = work["prior_gap_days"].where(work["prior_gap_days"] > 0)
    work["gamma_flip_velocity"] = gamma_flip_diff / safe_gap

    work["oi_change_by_dte"] = work["total_option_oi"] - prior1["total_option_oi"]
    prior_total_oi = prior1["total_option_oi"]
    work["volume_to_prior_oi"] = work["total_option_volume"] / prior_total_oi.where(prior_total_oi > 0)

    work["level_stability_days"] = _level_stability_days(work, prior1)

    if atr_lookup is None:
        atr_lookup = _build_atr_lookup(work)
    atr_values = [atr_lookup.get((row.symbol, row.snapshot_date)) for row in work.itertuples()]
    work["atr_14d"] = pd.Series(atr_values, index=work.index, dtype="float64")
    min_denom = (work["spot"].abs() * 1e-6).clip(lower=1e-9)
    denom = work["atr_14d"].where(work["atr_14d"].notna() & (work["atr_14d"] > 0), min_denom)
    work["distance_to_call_wall_atr"] = (work["spot"] - work["call_wall"]).abs() / denom
    work["distance_to_put_wall_atr"] = (work["spot"] - work["put_wall"]).abs() / denom
    work.loc[work["atr_14d"].isna(), ["distance_to_call_wall_atr", "distance_to_put_wall_atr"]] = np.nan

    skew_now, skew_change = _iv_skew(ladder, work[["symbol", "scope", "snapshot_date"]])
    work = work.merge(skew_now, on=["symbol", "scope", "snapshot_date"], how="left")
    work["iv_skew_change"] = skew_change

    near_share = _near_level_volume_share(ladder, work)
    work = work.merge(near_share, on=["symbol", "scope", "snapshot_date"], how="left")

    work["available_at"] = work["captured_at"]
    return work


def _level_stability_days(work: pd.DataFrame, prior1: pd.DataFrame) -> pd.Series:
    """Count of consecutive most-recent snapshots (including the current
    one) in which (call_wall, put_wall) is unchanged. If either the current
    or the prior value is missing, the pair is treated as changed (not
    stable) -- a missing level is never assumed equal to itself."""
    same = (
        work["call_wall"].notna()
        & work["put_wall"].notna()
        & prior1["call_wall"].notna()
        & prior1["put_wall"].notna()
        & (work["call_wall"] == prior1["call_wall"])
        & (work["put_wall"] == prior1["put_wall"])
    )
    changed = ~same
    grouped_block = changed.groupby([work["symbol"], work["scope"]], sort=False).cumsum()
    stability = grouped_block.groupby([work["symbol"], work["scope"], grouped_block], sort=False).cumcount() + 1
    return pd.Series(stability.values, index=work.index)


def _iv_skew(ladder: pd.DataFrame, keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """25-delta-ish risk-reversal skew: nearest-to-|0.25|-delta put IV minus
    nearest-to-0.25-delta call IV, per (symbol, scope, snapshot_date). Falls
    back to NaN when delta is unavailable for a side."""
    group_cols = ["symbol", "scope", "snapshot_date"]
    if ladder.empty:
        empty = keys.drop_duplicates().assign(iv_skew_25d=np.nan)
        return empty, pd.Series(np.nan, index=keys.index)

    calls = ladder.dropna(subset=["call_delta", "call_iv"]).copy()
    calls["_dist"] = (calls["call_delta"] - 0.25).abs()
    best_calls = calls.sort_values("_dist").groupby(group_cols, as_index=False, sort=False).first()
    best_calls = best_calls[group_cols + ["call_iv"]].rename(columns={"call_iv": "call_iv_25d"})

    puts = ladder.dropna(subset=["put_delta", "put_iv"]).copy()
    puts["_dist"] = (puts["put_delta"] + 0.25).abs()
    best_puts = puts.sort_values("_dist").groupby(group_cols, as_index=False, sort=False).first()
    best_puts = best_puts[group_cols + ["put_iv"]].rename(columns={"put_iv": "put_iv_25d"})

    skew = best_calls.merge(best_puts, on=group_cols, how="outer")
    skew["iv_skew_25d"] = skew["put_iv_25d"] - skew["call_iv_25d"]

    skew_sorted = skew.sort_values(["symbol", "scope", "snapshot_date"])
    prior_skew = skew_sorted.groupby(["symbol", "scope"], sort=False)["iv_skew_25d"].shift(1)
    skew_sorted = skew_sorted.assign(iv_skew_change=skew_sorted["iv_skew_25d"] - prior_skew)

    merged = keys.merge(
        skew_sorted[group_cols + ["iv_skew_25d", "iv_skew_change"]], on=group_cols, how="left"
    )
    return merged[group_cols + ["iv_skew_25d"]], merged["iv_skew_change"]


def _near_level_volume_share(ladder: pd.DataFrame, work: pd.DataFrame) -> pd.DataFrame:
    """Fraction of a scope's total option volume transacted at strikes
    within a small tolerance of any already-computed dealer level
    (call_wall, put_wall, gamma_flip, nearest_magnet). This describes where
    volume clusters relative to persisted levels; it is not a claim about
    who transacted or why."""
    group_cols = ["symbol", "scope", "snapshot_date"]
    if ladder.empty:
        return work[group_cols].drop_duplicates().assign(near_level_option_volume_share=np.nan)

    levels = work[group_cols + ["spot", "call_wall", "put_wall", "gamma_flip", "nearest_magnet"]].drop_duplicates(
        group_cols
    )
    merged = ladder.merge(levels, on=group_cols, how="inner", suffixes=("", "_lvl"))
    band = np.maximum(NEAR_LEVEL_MIN_BAND, merged["spot"].abs() * NEAR_LEVEL_PCT_BAND)

    def _near(level_col: str) -> pd.Series:
        level = merged[level_col]
        return level.notna() & ((merged["strike"] - level).abs() <= band)

    is_near = _near("call_wall") | _near("put_wall") | _near("gamma_flip") | _near("nearest_magnet")
    total_volume = merged["call_volume"].fillna(0.0) + merged["put_volume"].fillna(0.0)
    merged = merged.assign(_total_volume=total_volume, _near_volume=total_volume.where(is_near, 0.0))

    agg = merged.groupby(group_cols, as_index=False).agg(
        _total=("_total_volume", "sum"), _near=("_near_volume", "sum")
    )
    agg["near_level_option_volume_share"] = agg["_near"] / agg["_total"].where(agg["_total"] > 0)
    return agg[group_cols + ["near_level_option_volume_share"]]


# ---------------------------------------------------------------------------
# Strike-level dynamics
# ---------------------------------------------------------------------------

def compute_strike_dynamics(summary_dynamics: pd.DataFrame, ladder: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, scope, snapshot_date, strike) open-interest and volume
    change vs the actual prior available snapshot for that (symbol, scope).
    NaN where the strike did not exist in the prior snapshot (new strike, or
    no prior snapshot at all) -- a missing prior strike is never treated as
    zero open interest."""
    columns = [
        "symbol",
        "scope",
        "snapshot_date",
        "available_at",
        "strike",
        "call_oi",
        "put_oi",
        "call_oi_change",
        "put_oi_change",
        "oi_change_by_strike",
        "volume_to_prior_oi",
        "prior_snapshot_date",
        "prior_gap_days",
        "prior_gap_stale",
    ]
    if ladder.empty or summary_dynamics.empty:
        return pd.DataFrame(columns=columns)

    prior_map = summary_dynamics[
        ["symbol", "scope", "snapshot_date", "prior_snapshot_date", "prior_gap_days", "prior_gap_stale", "captured_at"]
    ].rename(columns={"captured_at": "available_at"})

    current = ladder.merge(prior_map, on=["symbol", "scope", "snapshot_date"], how="left")

    prior_side = ladder[["symbol", "scope", "snapshot_date", "strike", "call_oi", "put_oi"]].rename(
        columns={
            "snapshot_date": "prior_snapshot_date",
            "call_oi": "prior_call_oi",
            "put_oi": "prior_put_oi",
        }
    )
    merged = current.merge(prior_side, on=["symbol", "scope", "prior_snapshot_date", "strike"], how="left")

    merged["call_oi_change"] = merged["call_oi"] - merged["prior_call_oi"]
    merged["put_oi_change"] = merged["put_oi"] - merged["prior_put_oi"]
    merged["oi_change_by_strike"] = merged[["call_oi_change", "put_oi_change"]].sum(axis=1, min_count=1)

    prior_total_oi = merged["prior_call_oi"].fillna(0.0) + merged["prior_put_oi"].fillna(0.0)
    prior_available = merged["prior_call_oi"].notna() | merged["prior_put_oi"].notna()
    total_volume = merged["call_volume"].fillna(0.0) + merged["put_volume"].fillna(0.0)
    merged["volume_to_prior_oi"] = np.where(
        prior_available & (prior_total_oi > 0), total_volume / prior_total_oi.where(prior_total_oi > 0), np.nan
    )

    return merged[columns].sort_values(["symbol", "scope", "snapshot_date", "strike"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build(snapshot_root: Path = SNAPSHOT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, ladder = load_snapshot_history(snapshot_root)
    summary_dynamics = compute_summary_dynamics(summary, ladder)
    strike_dynamics = compute_strike_dynamics(summary_dynamics, ladder)
    return summary_dynamics, strike_dynamics


def run(
    *,
    snapshot_root: Path = SNAPSHOT_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> LevelDynamicsResult:
    summary_dynamics, strike_dynamics = build(snapshot_root)
    if summary_dynamics.empty:
        raise RuntimeError(f"no snapshot history found under {snapshot_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "level_dynamics_summary.parquet"
    by_strike_path = output_root / "level_dynamics_by_strike.parquet"
    summary_dynamics.to_parquet(summary_path, index=False)
    strike_dynamics.to_parquet(by_strike_path, index=False)

    return LevelDynamicsResult(
        summary_path=summary_path,
        by_strike_path=by_strike_path,
        summary_rows=len(summary_dynamics),
        by_strike_rows=len(strike_dynamics),
        symbols=int(summary_dynamics["symbol"].nunique()),
        snapshot_dates=int(summary_dynamics["snapshot_date"].nunique()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dealer-positioning level-dynamics (change) features.")
    parser.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    result = run(snapshot_root=Path(args.snapshot_root), output_root=Path(args.output_root))
    print(
        "dealer level-dynamics build complete: "
        f"symbols={result.symbols} snapshot_dates={result.snapshot_dates} "
        f"summary_rows={result.summary_rows} by_strike_rows={result.by_strike_rows} "
        f"summary={result.summary_path} by_strike={result.by_strike_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
