"""Read-only logic and data-integrity audit for one trading session.

Each check is the mechanical form of a defect that reached the live paper stack
and went unseen until a human looked at a chart (see
docs/INTEGRITY_CHECKLIST.md for the catalogue and the history of each). The
daily review runs this first and reports every WARN/FAIL.

    .venv/bin/python scripts/integrity_audit.py                 # latest session
    .venv/bin/python scripts/integrity_audit.py --date 2026-11-23
    .venv/bin/python scripts/integrity_audit.py --json

Exit status is 1 when any check FAILs, else 0. Nothing here writes, orders or
mutates state.
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from core.calendar import latest_session_date, prev_trading_day

REPO = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
IS_DIR = REPO / "Data/inference/intraday_structure"
UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"
BARS_1D = REPO / "Data/shared/bars/1d"
META_MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
SERVER_LOGS = REPO / "logs/live_server"
CONTEXT_DIR = IS_DIR / "decision_context"
PRE_ENTRY = {"SETUP_DETECTED", "ARMED", "CONFIRMED"}
OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL | INFO | SKIP
    detail: str


def _et_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(ET).date()
    except ValueError:
        return None


def _lines_for_date(path: Path, day: date, *, key: str = "timestamp") -> list[dict]:
    """Rows of an append-only JSONL whose ``key`` falls on ``day`` (ET), reading only the tail.

    Several of these logs are near a gigabyte, so the start of the day is found
    by growing a tail window until its first row predates ``day``.
    """
    if not path.exists():
        return []
    size = path.stat().st_size
    window = 8 << 20
    with path.open("rb") as fh:
        while True:
            start = max(0, size - window)
            fh.seek(start)
            if start:
                fh.readline()
            first = fh.readline()
            try:
                first_day = _et_date(json.loads(first).get(key))
            except (ValueError, AttributeError):
                first_day = None
            if start == 0 or (first_day is not None and first_day < day):
                break
            window *= 2
        fh.seek(start)
        if start:
            fh.readline()
        out = []
        for raw in fh:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if _et_date(row.get(key)) == day:
                out.append(row)
        return out


# --- checks ------------------------------------------------------------------------

def check_state_store() -> Check:
    try:
        with socket.create_connection(("127.0.0.1", 55432), timeout=2):
            return Check("state_store", "PASS", "127.0.0.1:55432 accepts connections")
    except OSError as exc:
        return Check("state_store", "FAIL", f"127.0.0.1:55432 unreachable ({exc.__class__.__name__}); "
                     "docker compose -f compose.nervous-system.yaml up -d postgres")


def check_carried_setups(day: date) -> Check:
    path = IS_DIR / "state.json"
    if not path.exists():
        return Check("intraday_carried_setups", "SKIP", "no state.json")
    setups = json.loads(path.read_text()).get("setups", [])
    carried = [s for s in setups if s.get("state") in PRE_ENTRY
               and (_et_date(s.get("state_entered_at")) or day) < day]
    if not carried:
        return Check("intraday_carried_setups", "PASS", f"no pre-entry setup older than {day}")
    oldest = min(carried, key=lambda s: s["state_entered_at"])
    return Check("intraday_carried_setups", "FAIL",
                 f"{len(carried)} pre-entry setups from earlier sessions hold their slots "
                 f"({len({s['ticker'] for s in carried})} tickers); oldest {oldest['setup_id']} "
                 f"{oldest['state']} since {oldest['state_entered_at'][:10]}")


def check_stuck_abstainer(day: date, *, limit: int = 200) -> Check:
    rows = _lines_for_date(IS_DIR / "abstentions.jsonl", day)
    if not rows:
        return Check("intraday_stuck_abstainer", "SKIP", f"no abstentions on {day}")
    per = Counter(r.get("setup_id") for r in rows)
    worst = [(sid, n) for sid, n in per.most_common(3) if n > limit]
    if not worst:
        return Check("intraday_stuck_abstainer", "PASS",
                     f"{len(rows):,} abstentions; busiest setup {per.most_common(1)[0][1]} (limit {limit})")
    return Check("intraday_stuck_abstainer", "WARN",
                 "a setup refused on hundreds of bars is stuck, not selective: "
                 + ", ".join(f"{sid} x{n}" for sid, n in worst))


def check_same_session_rearm(day: date) -> Check:
    rows = _lines_for_date(IS_DIR / "transitions.jsonl", day)
    if not rows:
        return Check("intraday_rearm_rate", "SKIP", f"no transitions on {day}")
    closed_at: dict[str, str] = {}
    rearmed: set[str] = set()
    for r in sorted(rows, key=lambda x: x["timestamp"]):
        sid, to = r["setup_id"], r.get("to_state")
        t = datetime.fromisoformat(r["timestamp"]).astimezone(ET)
        if to in {"CLOSED", "INVALIDATED", "EXHAUSTED"} and t.hour < 15 and t.time() >= datetime.min.replace(hour=9, minute=30).time():
            closed_at.setdefault(sid, r["timestamp"])
        elif to in {"SETUP_DETECTED", "ARMED"} and sid in closed_at and r["timestamp"] > closed_at[sid]:
            rearmed.add(sid)
    if not closed_at:
        return Check("intraday_rearm_rate", "INFO", "no RTH closes before 15:00")
    rate = len(rearmed) / len(closed_at)
    status = "WARN" if rate < 0.15 else "PASS"
    return Check("intraday_rearm_rate", status,
                 f"{len(rearmed)}/{len(closed_at)} setups closed before 15:00 re-armed the same session "
                 f"({rate:.0%}; was 6% when revival only ran on registration)")


SELECTED = re.compile(r"intraday execution: selected (\S+) for (\S+) \(spot [\d.]+, strike [\d.]+, otm ([+-]?[\d.]+)%")
UNFILLED = re.compile(r"intraday execution: entry for (\S+) did not fill")


def check_option_moneyness(day: date, *, max_otm: float = 0.03) -> Check:
    """Every contract the intraday executor picked, filled or not.

    Primary source is the executor's "selected" log line; booked exits (whose
    spot is in the ledger) cover days before that line existed. An unfilled
    pick with no spot is reported by name, since only its fill status is known.
    """
    picks: dict[str, float] = {}
    unfilled: list[str] = []
    prefix = day.isoformat()
    for log in sorted(SERVER_LOGS.glob("server_*.log"))[-3:]:
        with log.open(errors="replace") as fh:
            for line in fh:
                if not line.startswith(prefix) or "intraday execution:" not in line:
                    continue
                if (m := SELECTED.search(line)):
                    picks[m.group(1)] = float(m.group(3)) / 100.0
                elif (m := UNFILLED.search(line)):
                    unfilled.append(m.group(1))
    for r in _lines_for_date(IS_DIR / "closed_trades.jsonl", day, key="ts"):
        m = OCC.match(str(r.get("order_symbol") or ""))
        spot = r.get("u_entry")
        if r.get("module") != "intraday_structure" or not m or not spot or r["order_symbol"] in picks:
            continue
        strike, cp = int(m.group(4)) / 1000.0, m.group(3)
        picks[r["order_symbol"]] = (strike / float(spot) - 1.0) if cp == "C" else (1.0 - strike / float(spot))
    unknown = [o for o in unfilled if o not in picks]
    if not picks:
        return Check("option_moneyness", "INFO" if unknown else "SKIP",
                     f"no measurable picks on {day}" + (f"; unfilled, spot unknown: {', '.join(unknown)}" if unknown else ""))
    far = [(s, o) for s, o in picks.items() if o > max_otm]
    detail = f"{len(picks)} contracts, median OTM {statistics.median(picks.values()):+.1%}"
    if far:
        detail += f"; >{max_otm:.0%} OTM: " + ", ".join(f"{s} ({o:+.1%})" for s, o in far)
    if unknown:
        detail += f"; unfilled, spot unknown: {', '.join(unknown)}"
    return Check("option_moneyness", "WARN" if far else "PASS", detail)


def check_bars_held_clock(day: date) -> Check:
    rows = _lines_for_date(IS_DIR / "closed_setups.jsonl", day, key="exit_time")
    pairs = []
    for r in rows:
        if r.get("entry_time") and r.get("exit_time") and r.get("bars_held") is not None:
            minutes = (datetime.fromisoformat(r["exit_time"]) - datetime.fromisoformat(r["entry_time"])).total_seconds() / 60
            pairs.append((int(r["bars_held"]), minutes))
    if not pairs:
        return Check("ledger_bars_held_clock", "SKIP", f"no entered setups closed on {day}")
    bad = sum(b > 2 * m + 5 for b, m in pairs) / len(pairs)
    status = "WARN" if bad > 0.10 else "PASS"
    return Check("ledger_bars_held_clock", status,
                 f"{bad:.0%} of {len(pairs)} rows hold more bars than minutes elapsed allow "
                 "(bars_held must count from entry)")


def check_universe_reference(day: date) -> Check:
    if not UNIVERSE.exists():
        return Check("universe_px_freshness", "SKIP", "no shared_universe.csv")
    u = pd.read_csv(UNIVERSE)
    if "px_as_of" not in u.columns:
        return Check("universe_px_freshness", "FAIL",
                     "px has no as-of date: the universe predates the bar-cache refresh "
                     "(rebuild via nightly discovery --rebuild-universe)")
    elig = u[u["is_eligible"].astype(str).str.lower().isin({"true", "1"})]
    elig = elig.sort_values("avg_dollar_volume_20d", ascending=False).head(750)
    as_of = pd.to_datetime(elig["px_as_of"], errors="coerce").dt.date
    floor = prev_trading_day(day)
    stale = (as_of.isna() | (as_of < floor)).mean()
    legacy = (elig.get("px_source", pd.Series(dtype=str)) != "bars_1d").mean()
    status = "FAIL" if stale > 0.25 else ("WARN" if stale > 0.05 else "PASS")
    return Check("universe_px_freshness", status,
                 f"top-750 eligible: {stale:.0%} with px older than {floor}; {legacy:.0%} not from the bar cache")


def check_daily_cache(day: date) -> Check:
    if not UNIVERSE.exists():
        return Check("bars_1d_freshness", "SKIP", "no shared_universe.csv")
    u = pd.read_csv(UNIVERSE, usecols=lambda c: c in {"ticker", "is_eligible", "avg_dollar_volume_20d"})
    elig = u[u["is_eligible"].astype(str).str.lower().isin({"true", "1"})]
    tickers = elig.sort_values("avg_dollar_volume_20d", ascending=False)["ticker"].head(300)
    last = []
    for t in tickers:
        p = BARS_1D / f"{t}.parquet"
        if p.exists():
            ts = pd.read_parquet(p, columns=["timestamp"])["timestamp"]
            last.append(pd.to_datetime(ts, utc=True).max().tz_convert(ET).date())
    if not last:
        return Check("bars_1d_freshness", "SKIP", "no cached daily bars")
    fresh = sum(d >= day for d in last) / len(last)
    behind = sum(d < prev_trading_day(day) for d in last) / len(last)
    status = "FAIL" if behind > 0.10 else ("WARN" if fresh < 0.90 else "PASS")
    return Check("bars_1d_freshness", status,
                 f"top-300 names: {fresh:.0%} have the {day} bar, {behind:.0%} are more than a session behind "
                 "(the 22:15 readiness run writes the day's bar; tomorrow's gaps are measured from it)")


def check_refresher(day: date) -> Check:
    failed: Counter[str] = Counter()
    ok: Counter[str] = Counter()
    prefix = day.isoformat()
    for log in sorted(SERVER_LOGS.glob("server_*.log"))[-3:]:
        with log.open(errors="replace") as fh:
            for line in fh:
                if line.startswith(prefix) and "[refresher]" in line:
                    m = re.search(r"\[refresher\] (\w+) (OK|FAILED)", line)
                    if m:
                        (ok if m.group(2) == "OK" else failed)[m.group(1)] += 1
    if not failed and not ok:
        return Check("refresher_steps", "SKIP", f"no refresher lines dated {day}")
    if not failed:
        return Check("refresher_steps", "PASS", f"all refresher steps OK ({dict(ok)})")
    return Check("refresher_steps", "WARN" if "matrix" not in failed else "FAIL",
                 f"failures {dict(failed)} vs OK {dict(ok)}; see '[refresher] <step> failure output' lines")


def check_meta_matrix(day: date) -> Check:
    if not META_MATRIX.exists():
        return Check("meta_matrix_full_bar", "SKIP", f"no {META_MATRIX.name}")
    idx = pd.read_parquet(META_MATRIX, columns=[]).index
    ts = pd.to_datetime(idx.get_level_values("timestamp"), utc=True)
    counts = pd.Series(1, index=ts).groupby(level=0).size().sort_index()
    expected = float(counts.tail(40).median())
    full = counts[counts >= 0.5 * expected]
    latest = full.index[-1].tz_convert(ET)
    status = "PASS" if latest.date() >= day else "WARN"
    return Check("meta_matrix_full_bar", status, f"latest fully populated bar {latest:%Y-%m-%d %H:%M} ET")


def check_decision_context(day: date) -> Check:
    """The engine's per-decision context records exist, and carry dealer levels and the regime.

    Catches a recorder that is not running (e.g. a server not restarted after a
    deploy) and the 2026-11 pattern of index ETFs deciding with no dealer levels.
    """
    path = CONTEXT_DIR / f"{day:%Y%m%d}.jsonl"
    if not path.exists():
        moved = _lines_for_date(IS_DIR / "transitions.jsonl", day)
        if moved:
            return Check("decision_context", "WARN",
                         f"no decision-context file for {day} although {len(moved):,} setup transitions were logged "
                         "(recorder off, or the server predates it)")
        return Check("decision_context", "SKIP", f"no intraday activity on {day}")
    kinds: Counter[str] = Counter()
    index_rth = index_no_dealer = no_regime = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            kinds[r.get("kind")] += 1
            if r.get("daily_regime") is None:
                no_regime += 1
            t = datetime.fromisoformat(r["bar_ts"]).astimezone(ET)
            if r.get("instrument_class") == "index_etf" and (9, 30) <= (t.hour, t.minute) < (16, 0):
                index_rth += 1
                if (r.get("options") or {}).get("source") in (None, "none"):
                    index_no_dealer += 1
    total = sum(kinds.values())
    size_mb = path.stat().st_size / 1e6
    detail = f"{total:,} records ({dict(kinds)}), {size_mb:.1f} MB"
    problems = []
    if index_rth and index_no_dealer / index_rth > 0.20:
        problems.append(f"{index_no_dealer}/{index_rth} index-ETF RTH decisions had NO dealer levels")
    if no_regime:
        problems.append(f"{no_regime} records without a daily regime row")
    return Check("decision_context", "WARN" if problems else "PASS", detail + ("; " + "; ".join(problems) if problems else ""))


CHECKS = (
    ("state_store", lambda d: check_state_store()),
    ("intraday_carried_setups", check_carried_setups),
    ("intraday_stuck_abstainer", check_stuck_abstainer),
    ("intraday_rearm_rate", check_same_session_rearm),
    ("option_moneyness", check_option_moneyness),
    ("ledger_bars_held_clock", check_bars_held_clock),
    ("universe_px_freshness", check_universe_reference),
    ("bars_1d_freshness", check_daily_cache),
    ("refresher_steps", check_refresher),
    ("meta_matrix_full_bar", check_meta_matrix),
    ("decision_context", check_decision_context),
)


def run(day: date) -> list[Check]:
    out = []
    for name, fn in CHECKS:
        try:
            out.append(fn(day))
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the rest
            out.append(Check(name, "FAIL", f"check crashed: {exc.__class__.__name__}: {exc}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", type=date.fromisoformat, default=None, help="session date (default: latest)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    day = args.date or latest_session_date()
    checks = run(day)
    if args.json:
        print(json.dumps({"date": day.isoformat(), "checks": [asdict(c) for c in checks]}, indent=2))
    else:
        print(f"integrity audit for {day}")
        width = max(len(c.name) for c in checks)
        for c in checks:
            print(f"  {c.status:<4}  {c.name:<{width}}  {c.detail}")
    return 1 if any(c.status == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
