"""Nightly: attach forward outcomes to archived decisions.

Reads ``Data/research/decision_state/<module>/<date>/*.parquet`` and writes
``Data/research/decision_state_outcomes/<module>/<date>.parquet`` (definitions
in ``core.decision_state_outcomes``). A date is recomputed until all its
horizons have matured, then left alone. Outcomes are derived data and live in
their own tree; the raw archive is only ever read.

    .venv/bin/python -m scripts.decision_state.attach_outcomes            # last 45 days
    .venv/bin/python -m scripts.decision_state.attach_outcomes --days 400 --module meta_ranker
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.decision_state_archive import archive_root, list_partitions  # noqa: E402
from core.decision_state_outcomes import (  # noqa: E402
    HORIZONS,
    OUTCOME_SCHEMA,
    is_final,
    last_complete_session,
    session_dates,
    ticker_outcomes,
)

logger = logging.getLogger("decision_state.outcomes")
BARS_1H = REPO / "Data" / "shared" / "bars" / "1h"
BARS_1D = REPO / "Data" / "shared" / "bars" / "1d"
DEFAULT_OUT = REPO / "Data" / "research" / "decision_state_outcomes"


def _read_bars(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _flagged(daily: pd.DataFrame | None) -> set[pd.Timestamp]:
    """Non-organic corporate-action gap sessions (same guard as the Stage 2/3 labels)."""
    if daily is None or len(daily) < 2:
        return set()
    from strategies.momentum_expansion.ablation.stage2_labels import flagged_sessions
    return flagged_sessions(daily.set_index("timestamp"))


def outcomes_for_rows(rows: pd.DataFrame, *, spy: pd.DataFrame, calendar: pd.DatetimeIndex,
                      last_complete, bars_1h: Path = BARS_1H, bars_1d: Path = BARS_1D) -> pd.DataFrame:
    """Outcomes for every (run_id, ticker) in ``rows``, reading each ticker's
    bars once however many modules and dates mention it."""
    decisions = rows[["run_id", "ticker", "decision_ts"]].drop_duplicates(["run_id", "ticker"])
    parts = []
    for ticker, grp in decisions.groupby("ticker", sort=True):
        bars = spy if ticker == "SPY" else _read_bars(bars_1h / f"{ticker}.parquet")
        daily = _read_bars(bars_1d / f"{ticker}.parquet")
        parts.append(ticker_outcomes(grp, bars, calendar=calendar, last_complete=last_complete, spy=spy,
                                     daily=daily, flagged=_flagged(daily)))
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["run_id", "ticker"])
    frame["schema_version"] = OUTCOME_SCHEMA
    frame["computed_at"] = pd.Timestamp.now(tz="UTC")
    return frame


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run(*, days: int, module: str | None, root: Path | None, out_root: Path, force: bool) -> dict:
    spy = _read_bars(BARS_1H / "SPY.parquet")
    if spy is None or spy.empty:
        raise SystemExit("ABORT: no SPY 1H bars; cannot build the session calendar")
    calendar = pd.DatetimeIndex(sorted(set(session_dates(spy["timestamp"]))))
    last_complete = last_complete_session(calendar)
    start = (datetime.now(timezone.utc) - timedelta(days=days)).astimezone().strftime("%Y-%m-%d")
    by_date: dict[tuple[str, str], list[Path]] = {}
    for p in list_partitions(root, module, start=start):
        by_date.setdefault((p.parent.parent.name, p.parent.name), []).append(p)
    stats = {"partitions": 0, "skipped_final": 0, "rows": 0, "ok": 0, "immature": 0}
    pending = []
    for (mod, date), paths in sorted(by_date.items()):
        out_path = out_root / mod / f"{date}.parquet"
        if out_path.exists() and not force and is_final(pd.read_parquet(out_path)):
            stats["skipped_final"] += 1
            continue
        part = pd.concat([pd.read_parquet(p, columns=["run_id", "ticker", "decision_ts", "module"])
                          for p in paths], ignore_index=True)
        part["_out"] = str(out_path)
        pending.append(part)
    if not pending:
        return stats
    rows = pd.concat(pending, ignore_index=True)
    outcomes = outcomes_for_rows(rows, spy=spy, calendar=calendar, last_complete=last_complete)
    keyed = rows[["run_id", "ticker", "module", "_out"]].drop_duplicates(["run_id", "ticker"]).merge(
        outcomes, on=["run_id", "ticker"], how="left", validate="one_to_one")
    for out_path, frame in keyed.groupby("_out", sort=True):
        frame = frame.drop(columns=["_out"]).reset_index(drop=True)
        _write(frame, Path(out_path))
        stats["partitions"] += 1
        stats["rows"] += len(frame)
        stats["ok"] += int((frame.get(f"status_{HORIZONS[0]}") == "ok").sum())
        stats["immature"] += int((frame.get(f"status_{HORIZONS[-1]}") == "immature").sum())
        logger.info("%d decisions -> %s", len(frame), out_path)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=45, help="Archive dates to (re)consider, counting back from today.")
    ap.add_argument("--module", default=None)
    ap.add_argument("--root", default=None, help="Archive root (default: CYNOLYCUS_DECISION_ARCHIVE_ROOT or Data/research/decision_state).")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true", help="Recompute dates already final.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.root) if args.root else archive_root()
    stats = run(days=args.days, module=args.module, root=root, out_root=Path(args.out_root), force=args.force)
    print(f"decision-state outcomes: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
