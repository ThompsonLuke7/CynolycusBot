"""DTE floor as a RULE (not a model), regime routing rules, and the long-horizon put test.

All three studies use only valid data: real recorded fills + dense underlying daily
bars + the point-in-time regime panel. No option repricing anywhere (see
`10_RETRACTION_option_pnl_invalid.md`).

STUDY A -- how long does the move take, and what predicts it?
    Correlates signal-time observables against `days_to_mfe` (trading days until the
    underlying reaches its favorable peak). The output is a RULE for a DTE floor,
    not a trained model.

STUDY B -- regime routing rules.
    Tests simple, stated-in-advance rules on the real fills (e.g. "calls only when
    risk_appetite_z > 0"), reporting what each would have done to the book.

STUDY C -- are puts just on the wrong timescale?
    User hypothesis: puts work on thematic/cycle timescales (MSTR 500 -> 100), not on
    a 2-day swing clock. Extends the favorable-excursion measurement to 60/90/120/180
    days to see whether downside moves resume extending on a longer horizon.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "Data/analysis/multi_ticker_swing_live/paired_option_trades.csv"
BARS = REPO / "Data/shared/bars/1d"
REGIME = REPO / "Data/shared/market_regime/daily_regime.parquet"
OUT = REPO / "research/options_experiment/data/dte_rule_study.parquet"

LONG_HORIZONS = (5, 10, 20, 30, 45, 60, 90, 120, 180)


def load() -> pd.DataFrame:
    d = pd.read_csv(SRC)
    for c in ["entry_price_option", "pnl_dollars", "pnl_pct_option", "dte_at_entry",
              "entry_price_underlying", "holding_minutes", "atr_at_entry"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["entry_time"] = pd.to_datetime(d.entry_time, utc=True)
    d["dir"] = np.where(d.option_type == "C", 1, -1)
    return d[d.entry_price_option > 0].copy()


def paths(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tk, g in d.groupby("ticker"):
        p = BARS / f"{tk}.parquet"
        if not p.exists():
            continue
        b = pd.read_parquet(p)
        if "timestamp" not in b.columns:
            b = b.reset_index()
        b["timestamp"] = pd.to_datetime(b.timestamp, utc=True)
        b = b.sort_values("timestamp")
        ts = b.timestamp.values
        hi, lo, cl = b.high.to_numpy(), b.low.to_numpy(), b.close.to_numpy()
        for r in g.itertuples(index=False):
            i = int(np.searchsorted(ts, np.datetime64(r.entry_time.tz_localize(None)), side="right"))
            if i >= len(ts) - 5:
                continue
            S = r.entry_price_underlying
            if not np.isfinite(S) or S <= 0:
                continue
            # realized vol before entry (signal-time observable)
            j0 = max(0, i - 20)
            rv = float(np.std(np.diff(np.log(cl[j0:i])))) * np.sqrt(252) if i - j0 > 5 else np.nan
            rec = {"symbol": r.symbol, "ticker": tk, "entry_time": r.entry_time, "dir": r.dir,
                   "option_type": r.option_type, "dte": r.dte_at_entry, "pnl": r.pnl_dollars,
                   "ret": r.pnl_pct_option, "spot": S, "rv20": rv,
                   "atr_pct": (r.atr_at_entry / S) if np.isfinite(getattr(r, "atr_at_entry", np.nan)) else np.nan}
            for H in LONG_HORIZONS:
                j = min(i + max(int(H / 1.45), 1), len(ts))
                if j <= i:
                    continue
                arr = hi[i:j] if r.dir > 0 else lo[i:j]
                fav = ((arr.max() - S) / S) if r.dir > 0 else ((S - arr.min()) / S)
                rec[f"mfe_{H}"] = fav
                if H == 60:
                    k = int(np.argmax(arr)) if r.dir > 0 else int(np.argmin(arr))
                    rec["days_to_mfe60"] = k * 1.45          # trading bars -> calendar days
                    # first day the move exceeded +10%
                    thr = np.where(((arr - S) / S >= 0.10) if r.dir > 0
                                   else ((S - arr) / S >= 0.10))[0]
                    rec["days_to_10pct"] = (thr[0] * 1.45) if len(thr) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    u = paths(d)
    reg = pd.read_parquet(REGIME)
    reg["available_at"] = pd.to_datetime(reg.available_at, utc=True)
    rc = [c for c in reg.columns if c.endswith("_z")
          and not c.endswith(("_n_components", "_stale_days"))]
    u = pd.merge_asof(u.sort_values("entry_time"), reg[["available_at"] + rc].sort_values("available_at"),
                      left_on="entry_time", right_on="available_at", direction="backward")
    u.to_parquet(OUT, index=False)
    print(f"trades with path + regime: {len(u)}")

    # ------------------------------------------------------------------ STUDY A
    print("\n" + "=" * 74)
    print("STUDY A -- how long does the move take, and what predicts it?")
    print("=" * 74)
    v = u[u.days_to_10pct.notna()]
    print(f"\ntrades that ever reached +10% within 60d: {len(v)}/{len(u)} ({len(v)/len(u):.0%})")
    if len(v):
        print(f"  days to reach +10%: median {v.days_to_10pct.median():.0f}, "
              f"p25 {v.days_to_10pct.quantile(.25):.0f}, p75 {v.days_to_10pct.quantile(.75):.0f}, "
              f"p90 {v.days_to_10pct.quantile(.90):.0f}")
        print(f"  -> a DTE floor below ~{v.days_to_10pct.quantile(.50):.0f}d misses half of "
              f"the +10% moves that do occur")

    print("\ncorrelation of SIGNAL-TIME observables with days-to-+10% (lower = faster mover):")
    cands = ["rv20", "atr_pct", "spot"] + rc
    cc = []
    for c in cands:
        if c in v.columns and v[c].notna().sum() > 40:
            r = v[[c, "days_to_10pct"]].dropna()
            if len(r) > 40:
                cc.append((c, float(np.corrcoef(r[c], r.days_to_10pct)[0, 1]), len(r)))
    cc.sort(key=lambda x: -abs(x[1]))
    for c, r, n in cc[:8]:
        print(f"  {c:<28} r={r:+.3f}  (n={n})")
    if cc and abs(cc[0][1]) < 0.20:
        print("\n  -> NOTHING observable predicts speed well (|r| < 0.20 for every candidate).")
        print("     A per-trade adaptive DTE is therefore not supportable. A flat FLOOR is.")

    print("\nDTE floor candidates -- what fraction of the achievable +10% moves each captures:")
    for floor in (2, 7, 14, 21, 30, 45):
        cap = (v.days_to_10pct <= floor).mean()
        print(f"  floor {floor:>2}d: captures {cap:.0%} of the +10% moves that occur")

    # ------------------------------------------------------------------ STUDY B
    print("\n" + "=" * 74)
    print("STUDY B -- simple regime routing rules on the real fills")
    print("=" * 74)
    base_c = u[u.option_type == "C"]
    print(f"\nbaseline calls: n={len(base_c)} win {(base_c.pnl>0).mean():.0%} "
          f"total ${base_c.pnl.sum():,.0f}")
    rules = {
        "calls only when risk_appetite_z > 0": (u.option_type == "C") & (u.risk_appetite_z > 0),
        "calls only when risk_appetite_z > 0.5": (u.option_type == "C") & (u.risk_appetite_z > 0.5),
        "calls only when liquidity_stress_z < 0": (u.option_type == "C") & (u.liquidity_stress_z < 0),
        "calls when risk-ON and low stress": (u.option_type == "C") & (u.risk_appetite_z > 0) & (u.liquidity_stress_z < 0),
    }
    if "breadth_z" in u.columns:
        rules["calls only when breadth_z > 0"] = (u.option_type == "C") & (u.breadth_z > 0)
    print(f"\n{'rule':<42}{'n':>5}{'win':>7}{'total P&L':>12}{'$/trade':>10}")
    for name, m in rules.items():
        s = u[m.fillna(False)]
        if len(s) < 20:
            print(f"{name:<42}{len(s):>5}   (too few)")
            continue
        print(f"{name:<42}{len(s):>5}{(s.pnl>0).mean():>7.0%}${s.pnl.sum():>11,.0f}"
              f"${s.pnl.mean():>9,.0f}")
    print(f"{'(all calls, no rule)':<42}{len(base_c):>5}{(base_c.pnl>0).mean():>7.0%}"
          f"${base_c.pnl.sum():>11,.0f}${base_c.pnl.mean():>9,.0f}")

    # ------------------------------------------------------------------ STUDY C
    print("\n" + "=" * 74)
    print("STUDY C -- are puts simply on the wrong timescale?")
    print("=" * 74)
    print("\nmedian favorable underlying excursion by horizon:")
    print(f"  {'horizon':>8}{'calls':>10}{'puts':>10}{'put/call':>10}")
    for H in LONG_HORIZONS:
        c = f"mfe_{H}"
        if c not in u:
            continue
        cm = u[u.option_type == "C"][c].median()
        pm = u[u.option_type == "P"][c].median()
        if np.isfinite(cm) and np.isfinite(pm) and cm > 0:
            print(f"  {H:>6}d {100*cm:>9.1f}%{100*pm:>9.1f}%{pm/cm:>10.2f}")
    print("\n  share of PUT trades whose underlying eventually fell >20%:")
    for H in (30, 60, 90, 120, 180):
        c = f"mfe_{H}"
        if c in u:
            p = u[u.option_type == "P"]
            print(f"    within {H:>3}d: {(p[c] >= 0.20).mean():.0%}")

    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
