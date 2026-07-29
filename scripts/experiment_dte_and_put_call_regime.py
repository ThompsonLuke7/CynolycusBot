"""Two studies on the 575 REAL option fills, using only valid data.

Valid-data discipline (see 10_RETRACTION_option_pnl_invalid.md): option price bars
cannot mark positions on this universe. So neither study reprices an option. Both
use (a) the real recorded fills/outcomes and (b) UNDERLYING daily bars, which are
dense and reliable, plus (c) the point-in-time market-regime panel.

STUDY A -- DTE: was "nearest expiry" cutting the move off?
    Live selection is `min(listed expiry)` with _MIN_DTE_DAYS = 0, i.e. always the
    nearest, including 0/1DTE. Not adaptive, no regime input. For each real trade
    this measures WHEN the favorable underlying move actually happened, and how much
    additional favorable excursion existed beyond the expiry that was chosen.
    It answers "would more time have helped?" without pricing an option.

STUDY B -- puts vs calls: real effect or regime artifact?
    Puts lost -$32,912 vs calls +$5,523. This tests whether that survives
    conditioning on market regime, direction of SPY over the holding window, and
    whether puts were simply traded in the wrong regimes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "Data/analysis/multi_ticker_swing_live/paired_option_trades.csv"
BARS = REPO / "Data/shared/bars/1d"
REGIME = REPO / "Data/shared/market_regime/daily_regime.parquet"
OUT = REPO / "research/options_experiment/data/dte_putcall_study.parquet"

HORIZONS = (1, 2, 3, 5, 10, 20, 30, 45)


def load_trades() -> pd.DataFrame:
    d = pd.read_csv(SRC)
    for c in ["entry_price_option", "exit_price_option", "pnl_dollars", "pnl_pct_option",
              "dte_at_entry", "entry_price_underlying", "holding_minutes"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["entry_time"] = pd.to_datetime(d.entry_time, utc=True)
    d["exit_time"] = pd.to_datetime(d.exit_time, utc=True)
    d["dir"] = np.where(d.option_type == "C", 1, -1)
    return d[d.entry_price_option > 0].copy()


def underlying_paths(d: pd.DataFrame) -> pd.DataFrame:
    """For each trade, favorable underlying excursion at several horizons."""
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
            if i >= len(ts):
                continue
            S = r.entry_price_underlying
            if not np.isfinite(S) or S <= 0:
                continue
            rec = {"symbol": r.symbol, "ticker": tk, "entry_time": r.entry_time,
                   "dir": r.dir, "dte": r.dte_at_entry, "pnl": r.pnl_dollars,
                   "ret": r.pnl_pct_option, "option_type": r.option_type,
                   "hold_hours": r.holding_minutes / 60.0}
            # bars are daily; horizon in calendar days ~ trading days * 1.45
            for H in HORIZONS:
                j = min(i + max(int(H / 1.45), 1), len(ts))
                if j <= i:
                    continue
                fav = ((hi[i:j].max() - S) / S) if r.dir > 0 else ((S - lo[i:j].min()) / S)
                rec[f"mfe_{H}d"] = fav
                # when did the best move happen?
                arr = hi[i:j] if r.dir > 0 else lo[i:j]
                best = int(np.argmax(arr)) if r.dir > 0 else int(np.argmin(arr))
                rec[f"best_bar_{H}d"] = best
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    d = load_trades()
    u = underlying_paths(d)
    print(f"trades: {len(d)}, with underlying path: {len(u)}")

    # ---------------------------------------------------------------- STUDY A
    print("\n" + "=" * 74)
    print("STUDY A -- would MORE days to expiry have helped?")
    print("=" * 74)
    print(f"\nDTE actually chosen: median {u.dte.median():.0f}d, "
          f"{(u.dte <= 2).mean():.0%} at <=2 DTE  (selection is `min(expiry)`, not adaptive)")

    print("\nFavorable underlying excursion available, by horizon (median / p75):")
    for H in HORIZONS:
        c = f"mfe_{H}d"
        if c in u:
            print(f"  {H:>2}d: median {100*u[c].median():5.1f}%   p75 {100*u[c].quantile(.75):5.1f}%   "
                  f"share reaching +10%: {(u[c] >= 0.10).mean():.0%}")

    # the money question: how much move happened AFTER the chosen expiry?
    u = u.copy()
    u["mfe_at_expiry"] = u.apply(
        lambda r: r.get(f"mfe_{min(HORIZONS, key=lambda h: abs(h - max(r.dte, 1)))}d", np.nan), axis=1)
    for H in (10, 20, 30):
        col = f"mfe_{H}d"
        if col not in u:
            continue
        gain = u[col] - u.mfe_at_expiry
        share = (gain > 0.02).mean()
        print(f"\n  extending to {H}d: additional favorable excursion "
              f"median {100*gain.median():+.1f}pp, and {share:.0%} of trades gained >2pp more")

    print("\n  Did the move come AFTER the trade was closed?")
    for lbl, m in [("winners", u.pnl > 0), ("losers", u.pnl <= 0)]:
        s = u[m]
        if len(s) < 10:
            continue
        print(f"    {lbl:<8} n={len(s):3d}  MFE by 2d {100*s.mfe_2d.median():5.1f}%  "
              f"by 10d {100*s.mfe_10d.median():5.1f}%  by 30d {100*s.mfe_30d.median():5.1f}%")

    # ---------------------------------------------------------------- STUDY B
    print("\n" + "=" * 74)
    print("STUDY B -- are puts structurally bad, or was it the regime?")
    print("=" * 74)
    reg = pd.read_parquet(REGIME)
    reg["available_at"] = pd.to_datetime(reg.available_at, utc=True)
    reg = reg.sort_values("available_at")
    u2 = u.sort_values("entry_time")
    regime_cols = [c for c in reg.columns
                   if c.endswith("_z") and not c.endswith(("_n_components", "_stale_days"))]
    j = pd.merge_asof(u2, reg[["available_at"] + regime_cols],
                      left_on="entry_time", right_on="available_at", direction="backward")
    print(f"\nregime-joined trades: {j[regime_cols[0]].notna().sum()} / {len(j)}  "
          f"({len(regime_cols)} regime factors)")

    print("\nBaseline:")
    for t, g in j.groupby("option_type"):
        print(f"  {'calls' if t=='C' else 'puts '}: n={len(g):3d}  win {(g.pnl>0).mean():.0%}  "
              f"total ${g.pnl.sum():>9,.0f}  median ret {100*g.ret.median():+.0f}%")

    print("\nDid the UNDERLYING even move the right way? (this is the key control)")
    for t, g in j.groupby("option_type"):
        lbl = "calls" if t == "C" else "puts "
        print(f"  {lbl}: median favorable excursion by 2d = {100*g.mfe_2d.median():.1f}%, "
              f"by 10d = {100*g.mfe_10d.median():.1f}%")

    print("\nPuts vs calls conditioned on regime (risk_appetite_z tercile at entry):")
    if "risk_appetite_z" in j and j.risk_appetite_z.notna().sum() > 60:
        j["ra"] = pd.qcut(j.risk_appetite_z, 3, labels=["risk-OFF", "mid", "risk-ON"], duplicates="drop")
        t = j.groupby(["ra", "option_type"], observed=True).apply(
            lambda x: pd.Series({"n": len(x), "win": (x.pnl > 0).mean(),
                                 "total_pnl": x.pnl.sum(),
                                 "underlying_mfe_2d": 100 * x.mfe_2d.median()}),
            include_groups=False)
        print(t.round(2).to_string())

    j.to_parquet(OUT, index=False)
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
