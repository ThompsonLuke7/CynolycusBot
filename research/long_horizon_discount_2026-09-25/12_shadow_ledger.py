"""Forward shadow of the momentum blends: a record of what each arm WOULD hold from 2026-10 on.

Research only. It places no orders and reads nothing from any broker account. The only network
call is the market-data bar fetch. Every number in Parts 3-4 comes from one 2019-2026 window, and
this ledger is the out-of-sample check on it.

  --snapshot   Fetch fresh SIP adjusted daily bars (~620 calendar days) for today's shared universe into
               Data/research/shadow_bars/{asof}/, rank the universe as of the last completed session,
               and append each arm's target weights to shadow/ledger.csv. Entry is the NEXT session's open.
  --mark       Price every recorded decision from its entry open to the next decision's entry open (the
               last one to the latest close), chain the periods and compare each arm with SPY.

Arms, fixed on 2026-10-03 before any forward data exists:
  core_sat_300  (primary)  70% SPY + 30% equal-weight top 20 by 12-1 momentum among the 300 most liquid stocks
  core_sat_1000            the same blend on the liquid-1000 (the arm pre-specified in Part 3)
  mom_top300 / mom_1000    the two momentum sleeves alone
  spy                      benchmark
Features match 02_build_panel: mom_12_1 = close[t-21] / close[t-252] - 1, liquidity = 60-day median dollar
volume. Lives are split as in build_pit_universe (zero-volume padding, reused tickers, >10x jumps).
Cadence: run --snapshot every 4 weeks after a Friday close. It uses the last completed session.

    .venv/bin/python research/long_horizon_discount_2026-09-25/12_shadow_ledger.py --snapshot
    .venv/bin/python research/long_horizon_discount_2026-09-25/12_shadow_ledger.py --mark
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from scripts.research_data.build_pit_universe import split_lives  # noqa: E402
from scripts.research_data.fetch_research_bars_1d import UNIVERSE_CSV, fetch_ticker  # noqa: E402

BARS_ROOT = REPO / "Data" / "research" / "shadow_bars"
LEDGER = HERE / "shadow" / "ledger.csv"
N_TOP, SLEEVE = 20, 0.30
LOOKBACK_DAYS = 620


def load(bars: Path, t: str) -> pd.DataFrame | None:
    p = bars / f"{t}.parquet"
    if not p.exists():
        return None
    lives = split_lives(pd.read_parquet(p))
    if not lives:
        return None
    df = lives[-1]
    df = df.assign(date=pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
                   .dt.normalize().dt.tz_localize(None))
    return df.set_index("date")[["open", "close", "volume"]]


def latest_bars() -> Path:
    dirs = sorted(d for d in BARS_ROOT.iterdir() if d.is_dir())
    if not dirs:
        raise FileNotFoundError(f"no snapshot under {BARS_ROOT}; run --snapshot first")
    return dirs[-1]


def snapshot(force: bool) -> None:
    uni = pd.read_csv(UNIVERSE_CSV)
    tickers = sorted(set(uni.loc[uni["type"].fillna("Stock") != "ETF", "ticker"].dropna().astype(str).str.upper()) | {"SPY"})
    now = datetime.now(timezone.utc)
    end = now.strftime("%Y-%m-%dT00:00:00Z")
    start = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%dT00:00:00Z")
    bars = BARS_ROOT / now.strftime("%Y-%m-%d")
    bars.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = dict(ex.map(lambda t: fetch_ticker(t, end, False, bars, start), tickers))
    bad = {t: r for t, r in res.items() if not (isinstance(r, int) or r == "cached")}
    if "SPY" in bad or len(bad) > 0.05 * len(tickers):
        raise RuntimeError(f"bar fetch failed for {len(bad)} of {len(tickers)} tickers (SPY ok: {'SPY' not in bad})")
    asof = load(bars, "SPY").index.max()
    rows = []
    for t in tickers:
        df = load(bars, t)
        if t == "SPY" or df is None or df.index.max() != asof:
            continue   # no bar on the decision date: not tradable that day
        c = df["close"]
        rows.append({"ticker": t, "ref_close": c.iloc[-1], "dv60": (c * df["volume"]).rolling(60).median().iloc[-1],
                     "mom_12_1": c.iloc[-22] / c.iloc[-253] - 1 if len(c) >= 253 else float("nan")})
    u = pd.DataFrame(rows).dropna(subset=["dv60"])
    u["dv_rank"] = u["dv60"].rank(ascending=False)
    if len(u) < 1000:
        raise RuntimeError(f"only {len(u)} tradable stocks on {asof.date()}; expected > 1000")
    sleeves = {"mom_1000": u[u["dv_rank"] <= 1000].nlargest(N_TOP, "mom_12_1"),
               "mom_top300": u[u["dv_rank"] <= 300].nlargest(N_TOP, "mom_12_1")}
    spy_close = load(bars, "SPY")["close"].iloc[-1]
    out = [{"arm": "spy", "ticker": "SPY", "weight": 1.0, "ref_close": spy_close}]
    for arm, blend in (("mom_1000", "core_sat_1000"), ("mom_top300", "core_sat_300")):
        s = sleeves[arm].assign(arm=arm, weight=1 / N_TOP)
        out += s[["arm", "ticker", "weight", "ref_close", "mom_12_1", "dv_rank"]].to_dict("records")
        out += s.assign(arm=blend, weight=SLEEVE / N_TOP)[["arm", "ticker", "weight", "ref_close", "mom_12_1", "dv_rank"]].to_dict("records")
        out.append({"arm": blend, "ticker": "SPY", "weight": 1 - SLEEVE, "ref_close": spy_close})
    new = pd.DataFrame(out).assign(decision_date=asof.date().isoformat(), bars_snapshot=bars.name)
    old = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()
    if len(old) and asof.date().isoformat() in set(old["decision_date"]):
        if not force:
            raise SystemExit(f"{asof.date()} is already in the ledger (use --force to replace it)")
        old = old[old["decision_date"] != asof.date().isoformat()]
    LEDGER.parent.mkdir(exist_ok=True)
    pd.concat([old, new], ignore_index=True).to_csv(LEDGER, index=False)
    print(f"decision {asof.date()} | {len(u)} tradable stocks | fetch failures {len(bad)}")
    for arm in ("mom_top300", "mom_1000"):
        s = sleeves[arm]
        print(f"{arm:11s}: " + " ".join(f"{t}({m:+.0%})" for t, m in zip(s["ticker"], s["mom_12_1"])))


def mark() -> None:
    led = pd.read_csv(LEDGER, parse_dates=["decision_date"])
    bars = latest_bars()
    px = {t: load(bars, t) for t in led["ticker"].unique()}
    cal = px["SPY"].index
    decisions = sorted(led["decision_date"].unique())
    rows = []
    for i, d in enumerate(decisions):
        e = cal.searchsorted(pd.Timestamp(d), side="right")
        if e >= len(cal):
            print(f"decision {pd.Timestamp(d).date()}: no session after it yet, nothing to mark")
            continue
        nxt = cal.searchsorted(pd.Timestamp(decisions[i + 1]), side="right") if i + 1 < len(decisions) else None
        for arm, g in led[led["decision_date"] == d].groupby("arm"):
            ret = 0.0
            for t, w in zip(g["ticker"], g["weight"]):
                df = px[t]
                if df is None or cal[e] not in df.index:
                    continue   # no fill at the entry open: that weight stays in cash
                last = df["open"].get(cal[nxt]) if nxt is not None and nxt < len(cal) else None
                exit_px = last if last is not None else df["close"].iloc[-1]
                ret += w * (exit_px / df["open"].loc[cal[e]] - 1)
            rows.append({"decision": pd.Timestamp(d).date(), "entry": cal[e].date(), "arm": arm, "ret": ret})
    if not rows:
        return
    r = pd.DataFrame(rows).pivot(index="decision", columns="arm", values="ret")
    print(f"marked with {bars.name} bars through {cal.max().date()}\n\nperiod returns:")
    print(r.map(lambda v: f"{v:+.2%}").to_string())
    print("\ncumulative: " + "  ".join(f"{a} {v:+.2%}" for a, v in ((1 + r).prod() - 1).items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--mark", action="store_true")
    ap.add_argument("--force", action="store_true", help="replace an existing decision date in the ledger")
    args = ap.parse_args()
    if args.snapshot:
        snapshot(args.force)
    if args.mark:
        mark()


if __name__ == "__main__":
    main()
