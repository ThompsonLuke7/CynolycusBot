"""CLI to build the daily market-regime and sector-state tables.

Usage:
    python -m signals.market_regime.build [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Writes ``Data/shared/market_regime/daily_regime.parquet`` and
``sector_state.parquet`` atomically (temp file + rename, matching
strategies/intraday_structure/state_store.py's pattern).

``--start``/``--end`` filter the OUTPUT rows only; every rolling window is
always computed over the ticker's full cached history first, so trimming the
output never truncates a composite's lookback (which would reintroduce
warmup NaNs that shouldn't be there).
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import logging
from pathlib import Path
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from .config import DAILY_REGIME_PATH, SECTOR_STATE_PATH
from .daily_regime import build_daily_regime
from .nervous_system_adapter import persist_market_regime_outputs
from .sector_state import build_sector_state
from .timeutil import atomic_write_parquet

if TYPE_CHECKING:
    from core.nervous_system.persistence.uow import UnitOfWork

logger = logging.getLogger(__name__)

_STATE_STORE_UNAVAILABLE = "MARKET_REGIME_STATE_STORE_UNAVAILABLE"

# Match the freshness rule that consumes these states
# (config/freshness.py MVP_POLICY_DEFAULTS: MARKET and SECTOR are both 96h), so
# a state stops being valid at exactly the point the policy engine would start
# calling it stale. A Friday publication stays usable through Monday's open.
_STATE_VALIDITY = timedelta(hours=96)


def _default_valid_until(available_at: datetime) -> datetime:
    return available_at + _STATE_VALIDITY


# The state store is a causal cache for live decisions, not a second copy of the
# history. `get_state_candidates_for_snapshot` filters only on
# `available_at <= decision_time` and deliberately leaves expiry to the pure
# selector, so every published row is loaded and deserialised on EVERY snapshot
# build — 58+ of them in a single Meta run. Publishing the full tables (1,526
# MARKET + 16,786 SECTOR rows) would make each build scan six years of history
# to select one row. Publish a window comfortably wider than the 96h validity so
# a long weekend or a couple of missed nightlies still resolve, and no wider.
# Backfills should pass --start/--end explicitly.
_PUBLISH_LOOKBACK_DAYS = 10


def _recent_for_publication(df: pd.DataFrame, *, lookback_days: int) -> pd.DataFrame:
    if df.empty or lookback_days <= 0:
        return df
    cutoff = pd.Timestamp(df["date"].max()) - pd.Timedelta(days=lookback_days)
    # Boolean masking retains the producer's original index, which
    # _attach_source_lineage relies on for its record locators.
    return df[df["date"] >= cutoff]


def _publish_states(
    regime: pd.DataFrame,
    sector_state: pd.DataFrame,
    *,
    valid_until_for: Callable[[datetime], datetime],
) -> str:
    """Publish MARKET and SECTOR states, owning the unit of work ourselves.

    ``main`` used to publish only when a caller handed it a ``UnitOfWork``, and
    the sole production invocation is ``python -m signals.market_regime.build``
    from ``scripts/nightly_data_readiness.sh`` — which passes nothing. So
    ``persist_market_regime_outputs`` was unreachable in production and
    ``state_records`` never held a MARKET or SECTOR row, which is two of the
    three required states ``SnapshotBuilder`` could not resolve while Meta ran
    dark on 2026-08-20/21.

    Publication failure never fails the build: the two Parquet tables are the
    durable artifacts and are already on disk by the time this runs. The outcome
    is returned so the caller can log it, exactly as
    ``core.broker_equity_snapshot.capture_from_env`` does for PORTFOLIO.
    """

    try:
        from core.nervous_system.config.runtime import NervousSystemSettings
        from core.nervous_system.persistence.database import (
            create_database_engine,
            create_session_factory,
        )
        from core.nervous_system.persistence.uow import UnitOfWork

        # No argument, so from_env() falls back to .env for names the process
        # environment does not define. The server does not export CYNOLYCUS_*.
        settings = NervousSystemSettings.from_env()
        session_factory = create_session_factory(create_database_engine(settings))
    except Exception as exc:  # noqa: BLE001 - the Parquet tables must survive
        logger.warning(
            "market-regime state publication skipped (%s): %s",
            _STATE_STORE_UNAVAILABLE,
            type(exc).__name__,
        )
        return "FAILED"

    try:
        with UnitOfWork(session_factory) as uow:
            market_count, sector_count = persist_market_regime_outputs(
                regime,
                sector_state,
                unit_of_work=uow,
                valid_until_for=valid_until_for,
            )
            uow.commit()
    except Exception as exc:  # noqa: BLE001 - the Parquet tables must survive
        logger.warning(
            "market-regime state publication failed (%s): %s",
            _STATE_STORE_UNAVAILABLE,
            type(exc).__name__,
        )
        return "FAILED"

    logger.info(
        "published nervous-system states: %d MARKET, %d SECTOR",
        market_count,
        sector_count,
    )
    return "PUBLISHED"


def _filter_dates(df: pd.DataFrame, *, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df


def _artifact_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attach_source_lineage(
    df: pd.DataFrame, *, path: Path, source_id: str, table_name: str
) -> None:
    df.attrs["source_id"] = source_id
    df.attrs["content_hash"] = _artifact_sha256(path)
    # The filtered frame retains the producer's original index.  Publish an
    # explicit locator map after the Parquet write so attrs never alter output.
    df.attrs["record_locators"] = {
        index: f"{table_name}:row:{index}" for index in df.index
    }


def main(
    argv: list[str] | None = None,
    *,
    unit_of_work: "UnitOfWork | None" = None,
    valid_until_for: Callable[[datetime], datetime] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="Inclusive output start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Inclusive output end date (YYYY-MM-DD)")
    parser.add_argument("--out-regime", default=str(DAILY_REGIME_PATH), help="daily_regime.parquet output path")
    parser.add_argument("--out-sector-state", default=str(SECTOR_STATE_PATH), help="sector_state.parquet output path")
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Write the Parquet tables only; skip nervous-system state publication "
             "(for offline rebuilds and backfills, where publishing a historical "
             "window as if it were current would misdate the state store).",
    )
    parser.add_argument(
        "--publish-lookback-days",
        type=int,
        default=_PUBLISH_LOOKBACK_DAYS,
        help="How many trailing days of rows to publish as nervous-system states "
             f"(default {_PUBLISH_LOOKBACK_DAYS}). 0 publishes every output row; "
             "only do that for a deliberate backfill.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Building daily_regime table...")
    regime = build_daily_regime()
    logger.info("Building sector_state table...")
    sector_state = build_sector_state()

    regime = _filter_dates(regime, start=args.start, end=args.end)
    sector_state = _filter_dates(sector_state, start=args.start, end=args.end)

    out_regime = Path(args.out_regime)
    out_sector_state = Path(args.out_sector_state)
    atomic_write_parquet(regime, out_regime, index=False)
    atomic_write_parquet(sector_state, out_sector_state, index=False)

    publish = not args.no_publish
    if unit_of_work is not None and valid_until_for is None:
        raise ValueError("valid_until_for is required when publishing nervous-system states")
    if publish or unit_of_work is not None:
        _attach_source_lineage(
            regime,
            path=out_regime,
            source_id=str(out_regime),
            table_name="daily_regime",
        )
        _attach_source_lineage(
            sector_state,
            path=out_sector_state,
            source_id=str(out_sector_state),
            table_name="sector_state",
        )
        if unit_of_work is not None:
            # A caller-owned unit of work stays caller-owned: it commits, not us,
            # and it decides its own window.
            persist_market_regime_outputs(
                regime,
                sector_state,
                unit_of_work=unit_of_work,
                valid_until_for=valid_until_for,
            )
        else:
            lookback = int(args.publish_lookback_days)
            _publish_states(
                _recent_for_publication(regime, lookback_days=lookback),
                _recent_for_publication(sector_state, lookback_days=lookback),
                valid_until_for=valid_until_for or _default_valid_until,
            )

    logger.info(
        "daily_regime: %d rows [%s .. %s] -> %s",
        len(regime),
        regime["date"].min() if len(regime) else None,
        regime["date"].max() if len(regime) else None,
        out_regime,
    )
    logger.info(
        "sector_state: %d rows [%s .. %s] -> %s",
        len(sector_state),
        sector_state["date"].min() if len(sector_state) else None,
        sector_state["date"].max() if len(sector_state) else None,
        out_sector_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
