"""Read the level-dynamics history and attach it to a snapshot frame.

``scripts/build_level_dynamics.py`` has computed snapshot-over-snapshot change,
velocity, and stability features since 2026-07 and nothing has ever read them.
This module is the consumer side: a small, freshness-bounded join so the
rankings and the research exports can use features that were already built,
tested, and paid for.

Two rules are load-bearing, both borrowed from the theme-context join that had
to learn them the hard way:

* **Carry is bounded.** A dynamics row is joined to a snapshot only within
  ``MAX_CARRY_DAYS``; past that the columns go null rather than silently
  describing three-week-old structure as current.
* **Staleness is visible.** Every joined row carries
  ``dynamics_days_since_refresh`` so a consumer can see how old the answer is
  instead of inferring freshness from the fact that a value exists.

The join fails soft: a missing or empty dynamics artifact returns the input
frame with null dynamics columns and a logged reason, because dealer rankings
must keep working when an optional enrichment is absent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = REPO / "Data" / "dealer_positioning" / "level_dynamics"
SUMMARY_PATH = DYNAMICS_ROOT / "level_dynamics_summary.parquet"

# A dynamics row older than this is not carried forward onto a newer snapshot.
# Three trading days: long enough to bridge a holiday weekend or one failed
# capture, short enough that "wall moved yesterday" cannot mean last week.
MAX_CARRY_DAYS = 3

# The columns worth carrying. Deliberately a subset: these are the ones that say
# something the capture-time features do not already say.
DYNAMICS_FEATURE_COLUMNS = [
    "wall_change_1d",
    "wall_change_3d",
    "gex_concentration_change",
    "gamma_flip_velocity",
    "level_stability_days",
    "distance_to_call_wall_atr",
    "distance_to_put_wall_atr",
    "iv_skew_25d",
    "iv_skew_change",
    "near_level_option_volume_share",
    "volume_to_prior_oi",
    "atr_14d",
]

JOIN_KEYS = ["symbol", "scope"]

logger = logging.getLogger(__name__)


def load_dynamics_summary(path: Path = SUMMARY_PATH) -> pd.DataFrame:
    """Load the dynamics summary, or an empty frame when it is absent."""
    path = Path(path)
    if not path.exists():
        logger.info("level dynamics not joined: %s does not exist", path.name)
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - an unreadable optional feed must not kill the ranking
        logger.warning("level dynamics not joined: %s unreadable (%s)", path.name, type(exc).__name__)
        return pd.DataFrame()
    if frame.empty:
        logger.info("level dynamics not joined: %s is empty", path.name)
    return frame


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["scope"] = out["scope"].astype(str)
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["snapshot_date"])


def join_level_dynamics(
    frame: pd.DataFrame,
    *,
    path: Path = SUMMARY_PATH,
    max_carry_days: int = MAX_CARRY_DAYS,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Attach the newest dynamics row at or before each snapshot date.

    Never reads a dynamics row dated after the snapshot it is joined to, so the
    result is usable at the snapshot's own decision time.
    """
    columns = list(columns or DYNAMICS_FEATURE_COLUMNS)
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    if "dynamics_days_since_refresh" not in out.columns:
        out["dynamics_days_since_refresh"] = pd.NA

    dynamics = load_dynamics_summary(path)
    required = {"symbol", "scope", "snapshot_date"}
    if dynamics.empty or not required.issubset(dynamics.columns) or not required.issubset(out.columns):
        return out

    # Join from the keys only. Carrying the pre-created placeholder columns into
    # the merge makes pandas suffix the real values as `_x`/`_y`, and the empty
    # placeholder wins silently -- an all-null join that looks like it worked.
    left = _normalize(out[["symbol", "scope", "snapshot_date"]])
    right = _normalize(dynamics)
    available = [c for c in columns if c in right.columns]
    if not available or left.empty or right.empty:
        return out

    right = right[JOIN_KEYS + ["snapshot_date"] + available].copy()
    right = right.sort_values("snapshot_date").drop_duplicates(JOIN_KEYS + ["snapshot_date"], keep="last")
    right = right.rename(columns={"snapshot_date": "dynamics_date"})
    left = left.sort_values("snapshot_date")

    merged = pd.merge_asof(
        left.reset_index().rename(columns={"index": "_row"}),
        right.sort_values("dynamics_date"),
        left_on="snapshot_date",
        right_on="dynamics_date",
        by=JOIN_KEYS,
        direction="backward",
        allow_exact_matches=True,
    )
    merged["dynamics_days_since_refresh"] = (
        merged["snapshot_date"] - merged["dynamics_date"]
    ).dt.days

    # Bounded carry: beyond the limit the enrichment is dropped, not aged.
    too_old = merged["dynamics_days_since_refresh"] > int(max_carry_days)
    merged.loc[too_old, available] = pd.NA
    merged.loc[too_old, "dynamics_days_since_refresh"] = pd.NA

    merged = merged.set_index("_row").sort_index()
    for col in available:
        out[col] = merged[col]
    out["dynamics_days_since_refresh"] = merged["dynamics_days_since_refresh"]

    joined = int(merged[available[0]].notna().sum())
    logger.info(
        "level dynamics joined: %d of %d rows within %d-day carry", joined, len(out), max_carry_days
    )
    return out
