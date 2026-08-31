"""Derive intraday change features from the one-minute gamma ladder archive.

The dealer runner has been writing a gamma ladder per symbol per minute since
2026-06-12 -- roughly 330 snapshots per symbol per session across SPY, QQQ, IWM,
GLD and SLV -- and nothing has ever derived a change feature from any of it. The
nightly ``build_level_dynamics`` covers day-over-day movement; this covers the
session.

What it measures is *structure migration*, on the argument that where a wall is
going may say more than where it is:

    09:45 call wall 610 -> 10:15 612 -> 11:00 615 -> 11:45 618

is a different observation from ``call_wall = 618``.

Causality rules, both enforced rather than assumed:

* Every feature at time ``t`` is built only from snapshots at or before ``t``.
  Deltas look backward; nothing reads a later file.
* A gap wider than ``MAX_GAP_MINUTES`` yields a null delta instead of one
  spanning the gap, so a polling outage cannot masquerade as a violent
  intraday move.

CLI::

    python -m strategies.dealer_positioning.scripts.build_intraday_level_dynamics \
        --symbols SPY QQQ --since 2026-08-01
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from strategies.dealer_positioning.topology import concentration_index

REPO = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = REPO / "Data" / "dealer_positioning"
OUTPUT_ROOT = ARCHIVE_ROOT / "intraday_dynamics"

DEFAULT_SYMBOLS = ("SPY", "QQQ", "IWM", "GLD", "SLV")

# Horizons, in minutes, over which changes are measured.
HORIZONS = (1, 5, 15, 30)

# A delta is not computed across a gap wider than this. The runner polls every
# 60s, so anything past 5 minutes means the poller was down, not that structure
# moved that fast.
MAX_GAP_MINUTES = 5

# Levels whose migration is tracked.
LEVEL_COLUMNS = ("call_wall", "put_wall", "nearest_magnet")

_TIMESTAMP_RE = re.compile(r"gamma_ladder_(\d{8}T\d{6})")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntradayDynamicsResult:
    output_path: Path
    symbols: int
    rows: int


def _parse_snapshot_time(path: Path) -> datetime | None:
    match = _TIMESTAMP_RE.search(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _snapshot_features(ladder: pd.DataFrame) -> dict[str, float | None]:
    """Collapse one ladder into the per-snapshot scalars whose changes matter."""
    if ladder.empty:
        return {}
    work = ladder.copy()
    for col in ("strike", "spot", "net_gex", "total_abs_gex", "call_gex", "put_gex", "call_iv", "put_iv"):
        if col not in work.columns:
            work[col] = float("nan")
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["strike"])
    if work.empty:
        return {}

    spot = float(work["spot"].dropna().iloc[0]) if work["spot"].notna().any() else float("nan")
    total_abs = float(work["total_abs_gex"].abs().sum())
    net = float(work["net_gex"].sum())

    above = work[work["strike"] > spot]
    below = work[work["strike"] < spot]
    call_wall = _strike_of_max(above, "call_gex")
    put_wall = _strike_of_min(below, "put_gex")
    nearest_magnet = float(work.loc[work["net_gex"].abs().idxmax(), "strike"])

    near = work[(work["strike"] - spot).abs() <= spot * 0.01] if pd.notna(spot) and spot > 0 else work.iloc[0:0]
    return {
        "spot": spot,
        "estimated_net_gex": net,
        "total_abs_gamma": total_abs,
        "dealer_imbalance": (net / total_abs) if total_abs else None,
        "gamma_density_1pct": (float(near["total_abs_gex"].abs().sum()) / total_abs) if total_abs else None,
        "gamma_concentration": concentration_index(work["total_abs_gex"]),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "nearest_magnet": nearest_magnet,
        "atm_iv": _atm_iv(work, spot=spot),
    }


def _strike_of_max(frame: pd.DataFrame, col: str) -> float | None:
    if frame.empty or frame[col].max() <= 0:
        return None
    return float(frame.loc[frame[col].idxmax(), "strike"])


def _strike_of_min(frame: pd.DataFrame, col: str) -> float | None:
    if frame.empty or frame[col].min() >= 0:
        return None
    return float(frame.loc[frame[col].idxmin(), "strike"])


def _atm_iv(frame: pd.DataFrame, *, spot: float) -> float | None:
    if not pd.notna(spot) or spot <= 0 or frame.empty:
        return None
    idx = (frame["strike"] - spot).abs().idxmin()
    values = [frame.loc[idx, "call_iv"], frame.loc[idx, "put_iv"]]
    usable = [float(v) for v in values if pd.notna(v) and float(v) > 0]
    return float(sum(usable) / len(usable)) if usable else None


def load_symbol_snapshots(
    symbol: str,
    *,
    archive_root: Path = ARCHIVE_ROOT,
    since: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Collapse every archived ladder for one symbol into a time series."""
    directory = Path(archive_root) / symbol.upper() / "snapshots"
    if not directory.exists():
        logger.warning("no snapshot archive for %s", symbol)
        return pd.DataFrame()
    paths = sorted(directory.glob("gamma_ladder_*.csv"))
    if since:
        cutoff = pd.Timestamp(since, tz="UTC")
        paths = [p for p in paths if (ts := _parse_snapshot_time(p)) is not None and ts >= cutoff]
    if limit:
        paths = paths[-int(limit):]

    records: list[dict[str, float | None]] = []
    for path in paths:
        stamp = _parse_snapshot_time(path)
        if stamp is None:
            continue
        try:
            ladder = pd.read_csv(path)
        except Exception:  # noqa: BLE001 - one unreadable snapshot is not a failed build
            logger.debug("unreadable snapshot skipped: %s", path.name)
            continue
        features = _snapshot_features(ladder)
        if not features:
            continue
        features["captured_at"] = stamp
        features["symbol"] = symbol.upper()
        records.append(features)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).sort_values("captured_at").reset_index(drop=True)


def compute_intraday_dynamics(
    series: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    max_gap_minutes: int = MAX_GAP_MINUTES,
) -> pd.DataFrame:
    """Add backward-looking deltas at each horizon, nulled across polling gaps."""
    if series.empty:
        return series
    out = series.sort_values("captured_at").reset_index(drop=True).copy()
    out["session_date"] = pd.to_datetime(out["captured_at"]).dt.date

    delta_targets = [
        "estimated_net_gex",
        "total_abs_gamma",
        "dealer_imbalance",
        "gamma_density_1pct",
        "gamma_concentration",
        "atm_iv",
        *LEVEL_COLUMNS,
    ]

    for horizon in horizons:
        offset = pd.Timedelta(minutes=horizon)
        prior_time = pd.to_datetime(out["captured_at"]) - offset
        # merge_asof gives the newest snapshot at or before (t - horizon), which
        # keeps every lookup strictly backward-looking.
        reference = pd.merge_asof(
            pd.DataFrame({"target": prior_time, "_row": out.index}).sort_values("target"),
            out[["captured_at", *delta_targets]].sort_values("captured_at"),
            left_on="target",
            right_on="captured_at",
            direction="backward",
        ).set_index("_row").sort_index()

        gap = (prior_time - pd.to_datetime(reference["captured_at"])).dt.total_seconds() / 60.0
        too_wide = gap.isna() | (gap.abs() > float(max_gap_minutes))
        # A delta must also stay inside one session; overnight is not intraday.
        crossed_session = out["session_date"] != pd.to_datetime(reference["captured_at"]).dt.date

        for col in delta_targets:
            delta = out[col] - reference[col]
            delta[too_wide | crossed_session] = pd.NA
            out[f"delta_{col}_{horizon}m"] = delta

    return out


def run(
    *,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    archive_root: Path = ARCHIVE_ROOT,
    output_root: Path = OUTPUT_ROOT,
    since: str | None = None,
    limit: int | None = None,
) -> IntradayDynamicsResult:
    frames = []
    for symbol in symbols:
        series = load_symbol_snapshots(symbol, archive_root=archive_root, since=since, limit=limit)
        if series.empty:
            continue
        frames.append(compute_intraday_dynamics(series))
        logger.info("%s: %d snapshots", symbol, len(series))
    if not frames:
        raise RuntimeError("no intraday snapshots found for any requested symbol")

    combined = pd.concat(frames, ignore_index=True)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "intraday_level_dynamics.parquet"
    combined.to_parquet(output_path, index=False)
    return IntradayDynamicsResult(
        output_path=output_path,
        symbols=int(combined["symbol"].nunique()),
        rows=int(len(combined)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--since", default=None, help="ISO date; only snapshots at or after it")
    parser.add_argument("--limit", type=int, default=None, help="use only the newest N snapshots per symbol")
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(
        symbols=tuple(s.upper() for s in args.symbols),
        archive_root=args.archive_root,
        output_root=args.output_root,
        since=args.since,
        limit=args.limit,
    )
    print(f"wrote {result.rows} rows for {result.symbols} symbols -> {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
