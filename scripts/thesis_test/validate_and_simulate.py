"""Thesis test, step 3-4: validate the option price series, then simulate the hold.

STEP 3 IS NOT OPTIONAL. `research/options_experiment/10_RETRACTION_option_pnl_invalid.md`
records an entire study retracted because option "prices" were stale trade prints
-- corr(option return, underlying return) was +0.09 when it should be ~+0.9 for a
long call. Option daily bars are TRADE bars: on a thin contract a bar exists only
on days something traded, so a price read on any other day is a stale print and
any P&L built from it is fiction. Every contract is screened on:

  * coverage      bars present / trading days in the window
  * correlation   corr(option daily return, underlying daily return); a long call
                  must be strongly positive
  * staleness     share of consecutive bars with an identical close

Contracts that fail are EXCLUDED and counted, not silently averaged in.

STEP 4 then replays the thesis on the survivors: buy the ~35-45 DTE monthly at the
signal, hold N trading days, and exit on whichever comes first -- the horizon, a
premium stop, or expiry. Compared against the live policy (21 DTE, -39% premium
stop, median 2.0-day hold) and against the same entries expressed in shares.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "research/execution_quality/data"
PATHS = DATA / "thesis_contract_paths.jsonl"
BARS_1D = REPO / "Data/shared/bars/1d"

MIN_COVERAGE = 0.60         # bars on at least 60% of the window's trading days
MIN_CORR = 0.50             # a long call must track its underlying
MAX_STALE = 0.40            # at most 40% of steps with an unchanged close
_u: dict[str, pd.DataFrame | None] = {}


def underlying(ticker: str):
    if ticker not in _u:
        p = BARS_1D / f"{ticker}.parquet"
        d = None
        if p.exists():
            d = pd.read_parquet(p, columns=["timestamp", "open", "high", "low", "close"])
            d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
                "America/New_York").dt.date.astype(str)
            d = d.drop_duplicates("date").set_index("date")
        _u[ticker] = d
    return _u[ticker]


def validate(row) -> tuple[bool, dict]:
    bars = row.get("bars") or []
    if len(bars) < 6:
        return False, {"reason": "too_few_bars", "n_bars": len(bars)}
    ud = underlying(row["ticker"])
    if ud is None:
        return False, {"reason": "no_underlying"}
    dates = [b["t"] for b in bars]
    have = [d for d in dates if d in ud.index]
    if len(have) < 6:
        return False, {"reason": "no_date_overlap"}
    o = pd.Series([b["c"] for b in bars if b["t"] in ud.index], index=have, dtype=float)
    u = ud.loc[have, "close"].astype(float)
    o_ret, u_ret = o.pct_change().dropna(), u.pct_change().dropna()
    idx = o_ret.index.intersection(u_ret.index)
    if len(idx) < 5:
        return False, {"reason": "too_few_returns"}
    corr = float(np.corrcoef(o_ret.loc[idx], u_ret.loc[idx])[0, 1]) if o_ret.loc[idx].std() else float("nan")
    stale = float((o.diff().fillna(1) == 0).mean())
    # Window length in trading days, from the underlying's own calendar.
    span = ud.loc[(ud.index >= dates[0]) & (ud.index <= dates[-1])]
    coverage = len(bars) / max(1, len(span))
    stats = {"n_bars": len(bars), "coverage": round(coverage, 3),
             "corr_with_underlying": None if np.isnan(corr) else round(corr, 3),
             "stale_share": round(stale, 3)}
    if coverage < MIN_COVERAGE:
        return False, {**stats, "reason": "low_coverage"}
    if np.isnan(corr) or corr < MIN_CORR:
        return False, {**stats, "reason": "corr_too_low"}
    if stale > MAX_STALE:
        return False, {**stats, "reason": "too_stale"}
    return True, stats


def simulate(row, hold_days: int, prem_stop: float | None):
    """Buy at the first bar's close, hold `hold_days` trading days, exit on the
    first of: premium stop, horizon, last available bar."""
    bars = row["bars"]
    entry = float(bars[0]["c"])
    if entry <= 0:
        return None
    path = bars[1:hold_days + 1]
    if not path:
        return None
    for b in path:
        if prem_stop is not None and float(b["l"]) <= entry * (1.0 - prem_stop):
            return {"ret": -prem_stop, "exit": "stop", "days": path.index(b) + 1}
    last = float(path[-1]["c"])
    return {"ret": last / entry - 1.0, "exit": "horizon", "days": len(path)}


def shares_return(row, hold_days: int):
    ud = underlying(row["ticker"])
    if ud is None:
        return None
    dates = [b["t"] for b in row["bars"] if b["t"] in ud.index]
    if len(dates) < 2:
        return None
    seq = ud.loc[dates[0]:].head(hold_days + 1)["close"].astype(float)
    if len(seq) < 2:
        return None
    return float(seq.iloc[-1] / seq.iloc[0] - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holds", default="5,10,15,20")
    args = ap.parse_args()
    rows = [json.loads(l) for l in PATHS.open() if l.strip()]
    with_bars = [r for r in rows if r.get("n_bars")]
    print(f"entries fetched: {len(rows)}   with option bars: {len(with_bars)}\n")

    good, rejected = [], []
    for r in with_bars:
        ok, st = validate(r)
        (good if ok else rejected).append({**r, "_v": st})
    print("=" * 92)
    print("STEP 3 — price-series validation (the check the 2026-07 retraction exists to enforce)")
    print("=" * 92)
    import collections
    print(f"  usable contracts : {len(good)} / {len(with_bars)}")
    for reason, n in collections.Counter(x["_v"].get("reason") for x in rejected).most_common():
        print(f"  rejected: {reason:18s} {n}")
    if good:
        print(f"\n  median coverage  : {np.median([g['_v']['coverage'] for g in good]):.0%}")
        print(f"  median corr(opt, underlying): "
              f"{np.median([g['_v']['corr_with_underlying'] for g in good]):.3f}"
              "   <- must be ~+0.9 for a long call; +0.09 is what invalidated the 2026-07 study")
        print(f"  median stale share: {np.median([g['_v']['stale_share'] for g in good]):.0%}")
    if not good:
        print("\nNo usable contracts. Stop here rather than reporting P&L on stale prints.")
        return

    print("\n" + "=" * 92)
    print("STEP 4 — the thesis replayed on REAL option prices")
    print("=" * 92)
    print(f"{'policy':34s} {'n':>4s} {'median':>8s} {'mean':>8s} {'win%':>6s} "
          f"{'p90':>8s} {'total $/1k':>11s}")
    def report(label, rets):
        rets = np.array([x for x in rets if x is not None and np.isfinite(x)])
        if len(rets) == 0:
            print(f"{label:34s}   no rows"); return
        print(f"{label:34s} {len(rets):4d} {np.median(rets):8.1%} {rets.mean():8.1%} "
              f"{np.mean(rets > 0):6.0%} {np.percentile(rets, 90):8.1%} "
              f"{1000 * rets.sum():11,.0f}")

    for h in [int(x) for x in args.holds.split(",")]:
        report(f"OPTION hold {h}d, -39% stop (live)", [(_r or {}).get("ret") for _r in
                                                       (simulate(g, h, 0.39) for g in good)])
        report(f"OPTION hold {h}d, -60% stop", [(_r or {}).get("ret") for _r in
                                                (simulate(g, h, 0.60) for g in good)])
        report(f"OPTION hold {h}d, NO stop", [(_r or {}).get("ret") for _r in
                                              (simulate(g, h, None) for g in good)])
        report(f"SHARES hold {h}d", [shares_return(g, h) for g in good])
        print()

    out = DATA / "thesis_validation.json"
    out.write_text(json.dumps({"usable": len(good), "fetched": len(with_bars),
                               "rejects": collections.Counter(
                                   x["_v"].get("reason") for x in rejected)}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
