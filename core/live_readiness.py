"""Live-entry readiness and freshness gates.

Nightly/pre-open jobs do the expensive cache preparation. Live order paths use
this module to require a recent successful readiness stamp before opening new
positions. Risk-reducing sells are allowed to continue; buys are removed from the
plan when readiness is not proven.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from core.calendar import prev_trading_day

DEFAULT_READINESS_PATH = Path(__file__).resolve().parents[1] / "Data/readiness/latest_success.json"
DEFAULT_MAX_AGE_HOURS = 96.0
_ET = ZoneInfo("America/New_York")


def write_readiness_success(*, job: str, path: Path | None = None) -> dict:
    path = path or DEFAULT_READINESS_PATH
    payload = {
        "job": job,
        "status": "success",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return payload


def readiness_status(
    *,
    path: Path | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> tuple[bool, str, dict]:
    path = path or DEFAULT_READINESS_PATH
    if os.getenv("CYNOLYCUS_READINESS_REQUIRED", "1") == "0":
        return True, "readiness gate disabled by CYNOLYCUS_READINESS_REQUIRED=0", {}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return False, f"missing readiness stamp {path}: {exc}", {}
    if payload.get("status") != "success":
        return False, f"readiness status is {payload.get('status')!r}", payload
    raw_ts = payload.get("completed_at_utc")
    try:
        completed = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=timezone.utc)
    except Exception as exc:
        return False, f"invalid readiness timestamp {raw_ts!r}: {exc}", payload
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    completed_utc = completed.astimezone(timezone.utc)
    age_hours = (now_utc - completed_utc).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        return False, f"readiness stamp is {age_hours:.1f}h old (> {max_age_hours:.1f}h)", payload

    # A broad hour limit is needed across weekends, but by itself it lets a
    # Sunday stamp authorize Tuesday entries after Monday's refresh failed.
    # Require proof generated after the most recently completed trading
    # session's regular close.  Weekend/holiday stamps naturally satisfy the
    # preceding session threshold.
    now_et = now_utc.astimezone(_ET)
    prior_session = prev_trading_day(now_et.date())
    required_after = datetime.combine(prior_session, time(16, 0), tzinfo=_ET).astimezone(timezone.utc)
    if completed_utc < required_after:
        return False, (
            "readiness stamp predates latest completed trading session "
            f"({prior_session.isoformat()} 16:00 ET)"
        ), payload
    return True, f"readiness stamp OK ({age_hours:.1f}h old)", payload


def _drop_skipped_new_entries(new_managed: dict | None, skipped_symbols: set[str]) -> None:
    if not new_managed or not skipped_symbols:
        return
    for ticker, state in list(new_managed.items()):
        route = state.get("route", "equity")
        order_symbol = state.get("occ") if route == "option" else state.get("symbol", ticker)
        if order_symbol in skipped_symbols and int(state.get("runs_held", 0) or 0) == 0:
            new_managed.pop(ticker, None)


def filter_entry_orders_for_readiness(
    plan: Iterable[tuple],
    *,
    new_managed: dict | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> tuple[list[tuple], set[str], str]:
    """Remove BUY orders from ``plan`` unless the readiness stamp is current."""
    items = list(plan or [])
    buy_symbols = {str(p[0]) for p in items if len(p) >= 3 and str(p[1]).lower() == "buy"}
    if not buy_symbols:
        return items, set(), "no entry orders"
    ok, reason, _payload = readiness_status(max_age_hours=max_age_hours)
    if ok:
        return items, set(), reason
    skipped = buy_symbols
    _drop_skipped_new_entries(new_managed, skipped)
    kept = [p for p in items if str(p[0]) not in skipped]
    return kept, skipped, reason


def main() -> None:
    ap = argparse.ArgumentParser(description="Read/write live data-readiness stamp.")
    ap.add_argument("--write-success", action="store_true")
    ap.add_argument("--job", default="nightly_data_readiness")
    args = ap.parse_args()
    if args.write_success:
        print(json.dumps(write_readiness_success(job=args.job), indent=2, sort_keys=True))
    else:
        ok, reason, payload = readiness_status()
        print(json.dumps({"ok": ok, "reason": reason, "payload": payload}, indent=2, sort_keys=True))
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
