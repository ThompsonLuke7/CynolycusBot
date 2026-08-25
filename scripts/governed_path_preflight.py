#!/usr/bin/env python3
"""Would the governed path accept an order right now? Ask before the open.

Read-only. Builds no intent, submits nothing, writes nothing.

Why this exists. The Meta Ranker's pre-open flush was blocked from 2026-08-18 to
2026-08-24. Six separate causes were found and fixed one at a time, each
diagnosed only after a live session had already been lost, because the only
signal a blocked order produced was

    POLICY_VETO (SNAPSHOT_INVALID, SNAPSHOT_REQUIRED_STATE_MISSING)

The final cause was found by probing the INTRADAY decision parameters — the case
that worked — instead of the pre-open flush parameters, which were the case that
was broken. That distinction is the whole point of this script:

    --case intraday   decision bar = today's 14:00 UTC bar, decided ~14:20 ET
    --case flush      decision bar = the last COMPLETED 4H bar (18:00 UTC),
                      decided at the next session's 09:35 ET pre-open flush

Those resolve different states and fail differently. `flush` is the one that
carries every deferred entry, and it is the one nobody was testing.

Usage:
    PYTHONPATH=. python scripts/governed_path_preflight.py
    PYTHONPATH=. python scripts/governed_path_preflight.py --case intraday
    PYTHONPATH=. python scripts/governed_path_preflight.py --tickers PSIG,CRWD

Exit code is 0 when every checked ticker resolves a valid snapshot, 1 otherwise,
so it can gate a deploy or wake a scheduler.
"""
from __future__ import annotations

import argparse
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ET = ZoneInfo("America/New_York")
_UTC = timezone.utc

# The two decision shapes the 4H modules actually produce. Kept here rather than
# derived, because the point is to pin what production does, and a derivation
# that drifts from the runner would defeat the exercise.
_BAR_HOURS_UTC = (14, 18)          # 10:00 and 14:00 ET
_FLUSH_TIME_ET = dtime(9, 35)      # combined_server's pre-open flush slot
_INTRADAY_LAG = timedelta(minutes=20)   # runners fire ~20m after the bar closes


def _last_completed_bar(now_utc: datetime) -> datetime:
    """The most recent 4H bar whose window has closed."""
    from core.calendar.us_market_calendar import is_trading_day, prev_trading_day

    et = now_utc.astimezone(_ET)
    day = et.date() if is_trading_day(et.date()) else prev_trading_day(et.date())
    while True:
        for hour in reversed(_BAR_HOURS_UTC):
            bar = datetime(day.year, day.month, day.day, hour, tzinfo=_UTC)
            # A bar stamped at its START closes four hours later.
            if bar + timedelta(hours=4) <= now_utc:
                return bar
        day = prev_trading_day(day)


def _next_flush_time(now_utc: datetime) -> datetime:
    from core.calendar.us_market_calendar import is_trading_day, next_trading_day

    et = now_utc.astimezone(_ET)
    day = et.date()
    if not is_trading_day(day) or et.timetz().replace(tzinfo=None) >= _FLUSH_TIME_ET:
        day = next_trading_day(day)
    return datetime.combine(day, _FLUSH_TIME_ET, tzinfo=_ET).astimezone(_UTC)


def _case_parameters(case: str, now_utc: datetime) -> tuple[datetime, datetime, str]:
    """(decision_bar, decision_time, human label) for one decision shape."""
    bar = _last_completed_bar(now_utc)
    if case == "intraday":
        return bar, bar + timedelta(hours=4) + _INTRADAY_LAG, "intraday 4H run"
    return bar, _next_flush_time(now_utc), "pre-open flush of a deferred entry"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("flush", "intraday", "both"), default="both")
    parser.add_argument("--tickers", default="",
                        help="Comma-separated. Default: today's Meta top-K plus every "
                             "name with a queued entry or exit, which is the population "
                             "that actually gets submitted.")
    parser.add_argument("--strategy", default="meta_ranker")
    parser.add_argument("--profile", default="meta_4h_1420@1")
    args = parser.parse_args(argv)

    from core.nervous_system.config.freshness import get_snapshot_profile
    from core.nervous_system.config.runtime import NervousSystemSettings
    from core.nervous_system.context.diagnosis import diagnose_snapshot
    from core.nervous_system.context.snapshot_builder import SnapshotBuilder
    from core.nervous_system.persistence.database import (
        create_database_engine, create_session_factory,
    )
    from core.nervous_system.persistence.uow import UnitOfWork
    from signals.market_regime.config import SECTOR_ETFS_LIST

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        tickers = _default_tickers()
    if not tickers:
        print("no tickers to check (no queued orders and no readable state)")
        return 0

    now_utc = datetime.now(_UTC)
    cases = ("flush", "intraday") if args.case == "both" else (args.case,)

    try:
        settings = NervousSystemSettings.from_env()
        factory = create_session_factory(create_database_engine(settings))
    except Exception as exc:  # noqa: BLE001
        print(f"GOVERNED PATH UNREACHABLE: {type(exc).__name__} — nothing can submit")
        return 1

    failures = 0
    profile = get_snapshot_profile(args.profile)
    for case in cases:
        bar, decision_time, label = _case_parameters(case, now_utc)
        print(f"\n=== {case}: {label}")
        print(f"    decision_bar  {bar:%Y-%m-%d %H:%M}Z")
        print(f"    decision_time {decision_time:%Y-%m-%d %H:%M}Z "
              f"({decision_time.astimezone(_ET):%H:%M ET %a})")
        with UnitOfWork(factory) as uow:
            builder = SnapshotBuilder(uow.states, sector_entity_ids=tuple(SECTOR_ETFS_LIST))
            for ticker in tickers:
                try:
                    snapshot = builder.build(
                        strategy_id=args.strategy, entity_id=ticker,
                        decision_time=decision_time, decision_bar=bar, profile=profile,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  {ticker:<8} BUILD FAILED {type(exc).__name__}")
                    failures += 1
                    continue
                diagnosis = diagnose_snapshot(snapshot)
                if snapshot.valid:
                    print(f"  {ticker:<8} ok")
                    continue
                failures += 1
                blocking = ", ".join(
                    f"{item.state_type} {item.status}" for item in diagnosis.blocking
                )
                print(f"  {ticker:<8} BLOCKED  {blocking}")
                for item in diagnosis.blocking:
                    print(f"             {item.describe()}")
            # Snapshots are persisted idempotently by the builder; roll back so a
            # preflight never leaves rows behind.
            uow.rollback()

    print()
    if failures:
        print(f"PREFLIGHT FAILED: {failures} blocked snapshot(s)")
        return 1
    print("PREFLIGHT OK: every checked ticker resolves a valid snapshot")
    return 0


def _default_tickers() -> list[str]:
    """Today's targets plus anything already queued — the real submit population."""
    import json

    names: set[str] = set()
    inference = REPO_ROOT / "Data/inference/meta_ranker"
    for name in ("pending_open_entries.json", "pending_exit_orders.json"):
        path = inference / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        for entry in payload.get("entries", []):
            ticker = entry.get("ticker") or entry.get("order_symbol") or ""
            # An OCC symbol carries its underlying as the leading alphabetic run.
            head = "".join(ch for ch in str(ticker) if ch.isalpha())
            if head:
                names.add(head.upper() if len(head) < len(str(ticker)) else str(ticker).upper())
    audit = REPO_ROOT / "Data/inference/meta_ranker/live_signal_audit.jsonl"
    if audit.exists():
        try:
            for line in reversed(audit.read_text().splitlines()):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") == "order_plan" and row.get("targets"):
                    names.update(str(t).upper() for t in row["targets"])
                    break
        except Exception:  # noqa: BLE001
            pass
    return sorted(names)


if __name__ == "__main__":
    raise SystemExit(main())
