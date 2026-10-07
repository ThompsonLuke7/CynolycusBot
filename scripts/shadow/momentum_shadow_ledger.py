"""Forward shadow ledger for the top-300 momentum blends (research/long_horizon_discount_2026-09-25).

Research only. It places no orders and reads nothing from any broker account. The only network call is the
market-data bar fetch. It records what each arm WOULD hold, so the 2019-2026 backtest gets an out-of-sample
check that starts 2026-10-02.

SCHEDULE (why any run day works)
  A decision falls on the last trading session of every 4th week, counted from the anchor week of 2026-10-02.
  A run never asks "is today the day". It asks "which scheduled decisions have closed and are not in the
  ledger yet", and records each of them using bars up to that decision date only. So Friday evening,
  Saturday, Sunday, Monday before or after the open, or a week late all write the SAME row, and a missed
  weekend is caught up by the next run instead of being skipped. In the three off-weeks a run records nothing.
  A week counts as closed 30 minutes after its last session's 16:00 ET close. A Friday run before that sees
  last week as the latest closed week and does nothing; the decision stays due for the next run.

ARMS (fixed on 2026-10-03, before any forward data existed)
  core_sat_300  (primary)  70% SPY + 30% equal-weight top 20 by 12-1 momentum among the 300 most liquid stocks
  core_sat_1000            the same blend on the liquid-1000
  mom_top300 / mom_1000    the two momentum sleeves alone
  spy                      benchmark
Features match research 02_build_panel: mom_12_1 = close[t-21] / close[t-252] - 1; liquidity = 60-day median
dollar volume. Lives are split as in build_pit_universe (zero-volume padding, reused tickers, >10x jumps).
Entry is the first session's open after the decision date.

    PYTHONPATH=. .venv/bin/python scripts/shadow/momentum_shadow_ledger.py --auto      # the weekly-refresh stage
    PYTHONPATH=. .venv/bin/python scripts/shadow/momentum_shadow_ledger.py --mark      # score the ledger now
    PYTHONPATH=. .venv/bin/python scripts/shadow/momentum_shadow_ledger.py --status    # schedule only, no network

Files: research/long_horizon_discount_2026-09-25/shadow/ledger.csv (the record) and marks_latest.txt;
       Data/research/shadow_bars/{run date}/ (full-universe bars; only the newest 2 are kept) and marks/.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.calendar import is_trading_day, prev_trading_day  # noqa: E402
from scripts.research_data.build_pit_universe import split_lives  # noqa: E402
from scripts.research_data.fetch_research_bars_1d import UNIVERSE_CSV, fetch_ticker  # noqa: E402

ET = ZoneInfo("America/New_York")
ANCHOR = date(2026, 10, 2)            # first decision; the 4-week grid is counted from its week
CADENCE_WEEKS = 4
WEEK_CLOSED_AT = time(16, 30)         # 16:00 ET close + 30 min for the daily bar to be final
BARS_ROOT = REPO / "Data" / "research" / "shadow_bars"
SHADOW_DIR = REPO / "research" / "long_horizon_discount_2026-09-25" / "shadow"
LEDGER = SHADOW_DIR / "ledger.csv"
N_TOP, SLEEVE = 20, 0.30
LOOKBACK_DAYS = 620
KEEP_SNAPSHOTS = 2
MIN_TRADABLE = 1000


# ---------- schedule (pure functions of a clock and the ledger) ----------
def week_end_session(d: date) -> date:
    """Last trading session of d's Monday-Friday week. A Saturday or Sunday belongs to the week just ended."""
    fri = d + timedelta(days=4 - d.weekday())
    return fri if is_trading_day(fri) else prev_trading_day(fri)


def latest_closed_week_end(now: datetime) -> date:
    """The most recent week-ending session whose close (plus the data buffer) is behind `now`."""
    et = now.astimezone(ET)
    w = week_end_session(et.date())
    if et < datetime.combine(w, WEEK_CLOSED_AT, ET):
        w = week_end_session(w - timedelta(days=7))
    return w


def last_completed_session(now: datetime) -> date:
    """The latest session whose daily bar is final at `now`."""
    et = now.astimezone(ET)
    d = et.date()
    if not is_trading_day(d) or et < datetime.combine(d, WEEK_CLOSED_AT, ET):
        d = prev_trading_day(d)
    return d


def scheduled_decisions(through: date, anchor: date = ANCHOR) -> list[date]:
    """Every scheduled decision date from the anchor week up to `through`, one per CADENCE_WEEKS."""
    out, monday = [], anchor - timedelta(days=anchor.weekday())
    while (d := week_end_session(monday)) <= through:
        out.append(d)
        monday += timedelta(weeks=CADENCE_WEEKS)
    return out


def due_decisions(now: datetime, recorded: set[date], anchor: date = ANCHOR) -> list[date]:
    """Scheduled decisions that have closed and are missing from the ledger, oldest first."""
    return [d for d in scheduled_decisions(latest_closed_week_end(now), anchor) if d not in recorded]


def next_decision(now: datetime, anchor: date = ANCHOR) -> date:
    closed = latest_closed_week_end(now)
    monday = anchor - timedelta(days=anchor.weekday())
    while (d := week_end_session(monday)) <= closed:
        monday += timedelta(weeks=CADENCE_WEEKS)
    return d


# ---------- bars ----------
def _with_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(date=pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET).dt.normalize().dt.tz_localize(None))


def load_asof(path: Path, asof: date) -> pd.DataFrame | None:
    """The security's current life using bars up to `asof` only (so a later split or reuse cannot leak back)."""
    if not path.exists():
        return None
    df = _with_date(pd.read_parquet(path))
    lives = split_lives(df[df["date"] <= pd.Timestamp(asof)])
    return lives[-1].set_index("date")[["open", "close", "volume"]] if lives else None


def load_life_at(path: Path, when: pd.Timestamp, through: date) -> pd.DataFrame | None:
    """For marking: the life that was trading on `when`, with completed sessions only."""
    if not path.exists():
        return None
    df = _with_date(pd.read_parquet(path))
    for life in split_lives(df[df["date"] <= pd.Timestamp(through)]):
        if life["date"].iloc[0] <= when <= life["date"].iloc[-1]:
            return life.set_index("date")[["open", "close"]]
    return None


def fetch_bars(tickers: list[str], out_dir: Path, now: datetime, start: datetime, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    end = (now - timedelta(minutes=20)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    start_s = start.astimezone(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = dict(ex.map(lambda t: fetch_ticker(t, end, force, out_dir, start_s), tickers))
    bad = {t: r for t, r in res.items() if not (isinstance(r, int) or r == "cached")}
    if "SPY" in bad or len(bad) > 0.05 * len(tickers):
        raise RuntimeError(f"bar fetch failed for {len(bad)} of {len(tickers)} tickers (SPY ok: {'SPY' not in bad}); "
                           f"first errors: {list(bad.items())[:3]}")


def universe_tickers() -> list[str]:
    uni = pd.read_csv(UNIVERSE_CSV)
    return sorted(set(uni.loc[uni["type"].fillna("Stock") != "ETF", "ticker"].dropna().astype(str).str.upper()) | {"SPY"})


def prune_snapshots(keep: int = KEEP_SNAPSHOTS) -> list[str]:
    """Full-universe bar snapshots are 80+ MB each and re-fetchable; the ledger is the record."""
    dirs = sorted(d for d in BARS_ROOT.iterdir() if d.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name))
    gone = [d.name for d in dirs[:-keep]] if keep else []
    for name in gone:
        shutil.rmtree(BARS_ROOT / name)
    return gone


# ---------- decision ----------
def compute_targets(bars: Path, asof: date, tickers: list[str], min_tradable: int = MIN_TRADABLE) -> pd.DataFrame:
    """Target weights of every arm for the decision at the close of `asof`, from bars dated <= asof."""
    spy = load_asof(bars / "SPY.parquet", asof)
    if spy is None or spy.index.max() != pd.Timestamp(asof):
        raise RuntimeError(f"no completed SPY bar for {asof} in {bars} (last: "
                           f"{None if spy is None else spy.index.max().date()}); rerun after the close")
    rows = []
    for t in tickers:
        if t == "SPY":
            continue
        df = load_asof(bars / f"{t}.parquet", asof)
        if df is None or df.index.max() != pd.Timestamp(asof):
            continue   # no bar on the decision date: not tradable that day
        c = df["close"]
        rows.append({"ticker": t, "ref_close": c.iloc[-1], "dv60": (c * df["volume"]).rolling(60).median().iloc[-1],
                     "mom_12_1": c.iloc[-22] / c.iloc[-253] - 1 if len(c) >= 253 else float("nan")})
    u = pd.DataFrame(rows).dropna(subset=["dv60"])
    if len(u) < min_tradable:
        raise RuntimeError(f"only {len(u)} tradable stocks on {asof}; expected at least {min_tradable}")
    u["dv_rank"] = u["dv60"].rank(ascending=False)
    spy_close = spy["close"].iloc[-1]
    out = [{"arm": "spy", "ticker": "SPY", "weight": 1.0, "ref_close": spy_close}]
    keep = ["arm", "ticker", "weight", "ref_close", "mom_12_1", "dv_rank"]
    for arm, blend, n_liq in (("mom_1000", "core_sat_1000", 1000), ("mom_top300", "core_sat_300", 300)):
        s = u[u["dv_rank"] <= n_liq].nlargest(N_TOP, "mom_12_1")
        if len(s) < N_TOP or s["mom_12_1"].isna().any():
            raise RuntimeError(f"{arm}: fewer than {N_TOP} names with a 12-1 momentum value on {asof}")
        out += s.assign(arm=arm, weight=1 / N_TOP)[keep].to_dict("records")
        out += s.assign(arm=blend, weight=SLEEVE / N_TOP)[keep].to_dict("records")
        out.append({"arm": blend, "ticker": "SPY", "weight": 1 - SLEEVE, "ref_close": spy_close})
    return pd.DataFrame(out).assign(decision_date=asof.isoformat())


def read_ledger(ledger: Path) -> pd.DataFrame:
    return pd.read_csv(ledger) if ledger.exists() else pd.DataFrame(columns=["arm", "ticker", "weight", "decision_date"])


def recorded_dates(ledger: Path) -> set[date]:
    return {date.fromisoformat(d) for d in read_ledger(ledger)["decision_date"].dropna().unique()}


def append_decision(ledger: Path, rows: pd.DataFrame, bars: Path, now: datetime, replace: bool = False) -> None:
    old = read_ledger(ledger)
    d = rows["decision_date"].iloc[0]
    if d in set(old["decision_date"]):
        if not replace:
            raise SystemExit(f"{d} is already in the ledger (use --force to replace it)")
        old = old[old["decision_date"] != d]
    rows = rows.assign(bars_snapshot=bars.name, recorded_at_utc=now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger.with_suffix(".csv.tmp")
    out = rows if old.empty else pd.concat([old, rows], ignore_index=True)
    out.sort_values(["decision_date", "arm"], kind="stable").to_csv(tmp, index=False)
    tmp.replace(ledger)


def record(decisions: list[date], ledger: Path, now: datetime, bars: Path | None, replace: bool = False) -> None:
    tickers = universe_tickers()
    if bars is None:
        bars = BARS_ROOT / now.astimezone(ET).strftime("%Y-%m-%d")
        fetch_bars(tickers, bars, now, now - timedelta(days=LOOKBACK_DAYS), force=False)
    for d in decisions:
        rows = compute_targets(bars, d, tickers)
        append_decision(ledger, rows, bars, now, replace)
        late = (now.astimezone(ET).date() - d).days
        s = rows[rows["arm"] == "mom_top300"]
        print(f"recorded decision {d} ({late} day(s) after it) from {bars.name} bars | top-300 sleeve: "
              + " ".join(f"{t}({m:+.0%})" for t, m in zip(s["ticker"], s["mom_12_1"])))


# ---------- mark ----------
def mark(ledger: Path, now: datetime, bars: Path | None = None) -> str:
    led = read_ledger(ledger)
    if led.empty:
        return "ledger is empty"
    through = last_completed_session(now)
    if bars is None:
        bars = BARS_ROOT / "marks"
        # 120 days before the first decision: split_lives drops any life shorter than 30 sessions
        first = datetime.combine(date.fromisoformat(led["decision_date"].min()), time(), timezone.utc) - timedelta(days=120)
        fetch_bars(sorted(led["ticker"].unique()), bars, now, first, force=True)
    spy = load_life_at(bars / "SPY.parquet", pd.Timestamp(through), through)
    if spy is None:
        raise RuntimeError(f"no SPY bar through {through} in {bars}")
    cal = spy.index
    decisions = sorted(led["decision_date"].unique())
    entry = {d: int(cal.searchsorted(pd.Timestamp(d), side="right")) for d in decisions}
    rows, notes = [], []
    for i, d in enumerate(decisions):
        e = entry[d]
        if e >= len(cal):
            notes.append(f"decision {d}: its entry session has not completed yet")
            continue
        nxt = entry[decisions[i + 1]] if i + 1 < len(decisions) else len(cal)
        closed = nxt < len(cal)                       # the next decision's entry open exists: this period is finished
        for arm, g in led[led["decision_date"] == d].groupby("arm"):
            ret = 0.0
            for t, w in zip(g["ticker"], g["weight"]):
                life = load_life_at(bars / f"{t}.parquet", cal[e], through)
                if life is None or cal[e] not in life.index:
                    notes.append(f"{d} {arm} {t}: no bar at the entry open, weight left in cash")
                    continue
                if closed and cal[nxt] in life.index:
                    exit_px = life["open"].loc[cal[nxt]]
                else:                                   # open period, or the name stopped trading: last close
                    exit_px = life["close"].loc[:cal[min(nxt, len(cal) - 1)]].iloc[-1]
                ret += w * (exit_px / life["open"].loc[cal[e]] - 1)
            rows.append({"decision": d, "entry": cal[e].date().isoformat(),
                         "status": "closed" if closed else "open", "arm": arm, "ret": ret})
    lines = [f"momentum shadow ledger | marked through {through} | {len(decisions)} decision(s) | "
             f"next scheduled decision {next_decision(now)}"]
    if rows:
        r = pd.DataFrame(rows)
        tab = r.pivot(index=["decision", "entry", "status"], columns="arm", values="ret")
        lines.append("period returns (entry open -> next entry open; the open period -> last close):")
        lines.append(tab.map(lambda v: f"{v:+.2%}").to_string())
        cum = (1 + tab).prod() - 1
        lines.append("cumulative since the first entry: " + "  ".join(f"{a} {v:+.2%}" for a, v in cum.items()))
        if "spy" in cum:
            lines.append("vs SPY: " + "  ".join(f"{a} {v - cum['spy']:+.2%}" for a, v in cum.items() if a != "spy"))
    lines += notes
    txt = "\n".join(lines)
    (ledger.parent / "marks_latest.txt").write_text(txt + "\n")
    return txt


def status(ledger: Path, now: datetime) -> str:
    rec = recorded_dates(ledger)
    due = due_decisions(now, rec)
    return (f"now {now.astimezone(ET):%Y-%m-%d %H:%M %Z} | latest closed week ends {latest_closed_week_end(now)} | "
            f"recorded {sorted(d.isoformat() for d in rec)} | due now {[d.isoformat() for d in due]} | "
            f"next scheduled decision {next_decision(now)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--auto", action="store_true", help="record every due decision (none in off-weeks), then mark")
    ap.add_argument("--snapshot", action="store_true", help="record one decision now: --date, else the latest closed week end")
    ap.add_argument("--mark", action="store_true")
    ap.add_argument("--status", action="store_true", help="print the schedule; no network, no writes")
    ap.add_argument("--date", type=date.fromisoformat, help="decision date for --snapshot")
    ap.add_argument("--force", action="store_true", help="replace an existing decision date in the ledger")
    ap.add_argument("--now", type=datetime.fromisoformat, help="pretend this is the current time (ISO; ET if no offset)")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--bars-dir", type=Path, help="use this existing bar directory instead of fetching")
    args = ap.parse_args()
    now = datetime.now(timezone.utc) if args.now is None else (args.now if args.now.tzinfo else args.now.replace(tzinfo=ET))
    print(status(args.ledger, now))
    if args.auto:
        due = due_decisions(now, recorded_dates(args.ledger))
        if due:
            record(due, args.ledger, now, args.bars_dir)
            if args.bars_dir is None:
                gone = prune_snapshots()
                if gone:
                    print(f"pruned old bar snapshots: {gone}")
        else:
            print("no decision is due this week; nothing recorded")
    if args.snapshot:
        record([args.date or latest_closed_week_end(now)], args.ledger, now, args.bars_dir, replace=args.force)
    if args.auto or args.mark:
        print(mark(args.ledger, now, args.bars_dir))


if __name__ == "__main__":
    main()
