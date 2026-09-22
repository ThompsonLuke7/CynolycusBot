#!/usr/bin/env python
"""Build the empirical sector-assignment cache for the whole universe.

The repo already has a point-in-time sector resolver
(`signals.market_regime.sector_map.empirical_sector_for`): it correlates a
ticker's trailing daily returns against each sector ETF and takes the best
match, cached per calendar month. It has simply never been run --
`Data/shared/market_regime/sector_assignments.parquet` does not exist, and
`SECTOR_RESOLVER_ENABLED` defaults to False.

That gap has a cost. The curated `SECTOR_MAP` covers 95 of 2,903 names (3.3%),
so the nervous system's sector concentration limit buckets ~97% of positions as
`UNALLOCATED` -- and `UNALLOCATED` deliberately never vetoes, which makes the
sector arm of that gate inert.

Why a batch builder rather than calling the resolver in a loop: the per-ticker
function rewrites the ENTIRE cache parquet on every call, so 2,900 names means
2,900 full rewrites. This computes them all and writes once, reusing
`correlate_to_sectors` unchanged so the result is identical to what the
per-ticker path would produce.

    .venv/bin/python scripts/build_sector_assignments.py --as-of 2026-09-11

POINT-IN-TIME: the correlation cutoff is the snapshot MONTH's first calendar
day, not `as_of`, so the assignment is identical whichever day of the month it
is queried on and nothing later in the month can leak in. That is the
resolver's own rule, preserved here.

CIRCULARITY WARNING: these assignments are themselves derived from trailing
return correlation. They are the right input for a sector EXPOSURE limit, but
they are a weak control for anything else built from correlation -- notably the
correlation peer groups in `signals/peer_structure/`. Do not treat
"beats sector" as an independent check when the sector is itself a correlation
cluster.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.shared_universe.universe import load_universe_as_of  # noqa: E402
from signals.market_regime.config import (  # noqa: E402
    SECTOR_ASSIGNMENTS_PATH,
    SECTOR_ETFS_LIST,
    SECTOR_RESOLVER_MIN_OBS,
    SECTOR_RESOLVER_WINDOW_DAYS,
)
from signals.market_regime.sector_map import (  # noqa: E402
    SECTOR_MAP,
    _daily_log_returns,
    _month_start,
    correlate_to_sectors,
)
from signals.market_regime.timeutil import atomic_write_parquet  # noqa: E402


logger = logging.getLogger("sector_assignments")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=None, help="Session (YYYY-MM-DD).")
    parser.add_argument("--window-days", type=int, default=SECTOR_RESOLVER_WINDOW_DAYS)
    parser.add_argument("--min-obs", type=int, default=SECTOR_RESOLVER_MIN_OBS)
    parser.add_argument("--out", default=None, help="Override the cache path.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute this month even if it is already cached. The resolver is "
             "deterministic within a month (the correlation cutoff is the month's "
             "first day), so this changes nothing unless the bar cache changed.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()
    today = pd.Timestamp.today().normalize()
    if as_of > today:
        # A future --as-of writes a snapshot LABELLED for a month that has not
        # happened, built from whatever bars exist now. `empirical_sector_for`
        # keys on (ticker, snapshot_month) and would then serve that premature
        # row instead of recomputing when the month actually arrives. Refuse it.
        logger.error(
            "--as-of %s is in the future (today is %s); refusing to write a "
            "snapshot for a month that has not happened",
            as_of.date(), today.date(),
        )
        return 2
    snapshot_month = _month_start(as_of)
    out_path = Path(args.out) if args.out else Path(SECTOR_ASSIGNMENTS_PATH)

    # Cheap exit for a recurring caller. The correlation cutoff is the month's
    # first day, so re-running inside the same month reproduces the same answer
    # bar-for-bar; a weekly job should not spend two minutes proving that.
    existing = pd.DataFrame()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing["snapshot_month"] = pd.to_datetime(existing["snapshot_month"])
        if not args.force and (existing["snapshot_month"] == snapshot_month).any():
            cached = int((existing["snapshot_month"] == snapshot_month).sum())
            logger.info(
                "%s already holds %d rows for %s; nothing to do (use --force to "
                "recompute)", out_path, cached, snapshot_month.date(),
            )
            return 0

    universe = load_universe_as_of(
        as_of.tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59)
    )
    tickers = sorted(universe["ticker"].astype(str).str.upper().unique())
    logger.info("universe as of %s: %d tickers", as_of.date(), len(tickers))

    sector_returns: dict[str, pd.Series] = {}
    for etf in SECTOR_ETFS_LIST:
        try:
            sector_returns[etf] = _daily_log_returns(etf)
        except FileNotFoundError:
            logger.warning("[%s] no cached daily bars for sector ETF — skipped", etf)
    if not sector_returns:
        logger.error("no sector ETF bars available; nothing to correlate against")
        return 1
    logger.info("sector ETFs available: %d (%s)", len(sector_returns), ",".join(sorted(sector_returns)))

    rows: list[dict] = []
    resolved = curated = missing = insufficient = 0
    for i, ticker in enumerate(tickers, 1):
        if i % 500 == 0:
            logger.info("  %d/%d ...", i, len(tickers))
        try:
            returns = _daily_log_returns(ticker)
        except FileNotFoundError:
            missing += 1
            rows.append({"ticker": ticker, "snapshot_month": snapshot_month,
                         "sector_etf": None, "correlation": None, "n_obs": 0})
            continue
        best_etf, best_corr, n_obs = correlate_to_sectors(
            returns, sector_returns, asof=snapshot_month,
            window_days=args.window_days, min_obs=args.min_obs,
        )
        if best_etf is None:
            insufficient += 1
        else:
            resolved += 1
        # The curated map stays authoritative where it has an opinion; the
        # resolver fills the 97% it does not cover. Recorded either way so the
        # disagreement rate is measurable rather than hidden.
        rows.append({
            "ticker": ticker,
            "snapshot_month": snapshot_month,
            "sector_etf": SECTOR_MAP.get(ticker, best_etf),
            "correlation": best_corr,
            "n_obs": n_obs,
            "resolver_sector_etf": best_etf,
            "curated": ticker in SECTOR_MAP,
        })
        if ticker in SECTOR_MAP:
            curated += 1

    frame = pd.DataFrame(rows)
    # MERGE, never overwrite. `empirical_sector_for` keys its lookups on
    # (ticker, snapshot_month), so the prior months ARE the point-in-time
    # record -- replacing the file wholesale would silently destroy every
    # earlier snapshot and leave historical lookups answering with today's
    # sectors. Only this month's rows are replaced.
    if not existing.empty:
        kept = existing[existing["snapshot_month"] != snapshot_month]
        frame = pd.concat([kept, frame], ignore_index=True)
        logger.info(
            "merged: kept %d rows from %d earlier month(s)",
            len(kept), kept["snapshot_month"].nunique(),
        )
    atomic_write_parquet(frame, out_path, index=False)

    current = frame[frame["snapshot_month"] == snapshot_month]
    agree = current[current["curated"].fillna(False) & current["resolver_sector_etf"].notna()]
    agreement = (
        float((agree["sector_etf"] == agree["resolver_sector_etf"]).mean())
        if not agree.empty else float("nan")
    )
    logger.info("wrote %s (%d rows)", out_path, len(frame))
    logger.info(
        "resolved %d | curated %d | no bars %d | insufficient history %d",
        resolved, curated, missing, insufficient,
    )
    logger.info("coverage: %.1f%% of the universe now has a sector",
                100.0 * current["sector_etf"].notna().mean())
    logger.info(
        "curated-vs-resolver agreement on the %d curated names with a resolver "
        "opinion: %.1f%%  (a low number means the resolver is not reproducing "
        "the hand map and should not be trusted blindly)",
        len(agree), 100.0 * agreement,
    )
    print(current["sector_etf"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
