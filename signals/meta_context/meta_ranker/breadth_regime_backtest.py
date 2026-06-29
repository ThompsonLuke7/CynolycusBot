"""Are some ENTRY DAYS systematically better than others?

User question: on a great entry day (e.g. 2026-04-01) nearly all of the top-15 combo
names go on to positive forward returns; on a bad day only 2-3 do and the rest are flat
/ negative. Is that day-level "breadth of success" predictable from the market regime
KNOWN AT ENTRY (no look-ahead)?

For each 4H scan bar we:
  1. score the matrix -> combo rank, take the top-K (default 15),
  2. measure the FORWARD outcome of that basket:
       breadth_pos   = frac of top-K with fwd_close_return > 0   ("how many worked")
       breadth_5pct  = frac with fwd_close_return >= +5%
       breadth_alpha = frac with fwd_max_alpha   > 0             ("beat the market")
       port_mean     = mean fwd_close_return of the basket       (the day's PnL)
  3. snapshot the ENTRY-TIME regime (market-wide regime_*/treasury_*/macro_* cols are
     ~constant within a bar; per-ticker descriptors are averaged over the basket).

Then we correlate breadth vs each regime feature (Spearman, across bars) and bucket bars
into bad / mid / good breadth terciles to see which regime each falls in. Regime is
observed at entry, breadth is the forward result => the correlation is predictive.

  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/breadth_regime_backtest.py --k 15
  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/breadth_regime_backtest.py --k 15 --since 2026-01-01 --show-day 2026-04-01
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from signals.meta_context.meta_ranker.score import score_frame, DEFAULT_MATRIX

# market-wide regime (constant within a bar) — the real "what regime was it" signal
MARKET_REGIME = [
    "regime_vix_z", "regime_vix_high", "regime_spy_trend", "regime_spy_ret_20",
    "treasury_spread_2s10s", "treasury_spread_3m10y", "treasury_inverted",
    "treasury_10y_change_5d", "treasury_10y",
    "days_to_high_impact_macro", "days_since_high_impact_macro",
    "macro_high_impact_today_count", "macro_high_impact_next_3d_count",
]
# per-ticker descriptors — averaged over the basket = "what kind of names got picked"
BASKET_DESC = ["theme_breadth", "beta_spy_60", "trend_persistence", "dollar_vol_pctile_252"]


def build_day_table(sc: pd.DataFrame, k: int) -> pd.DataFrame:
    sc = sc.copy()
    sc["rank"] = sc.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
    top = sc[sc["rank"] <= k].copy()
    rows = []
    for ts, g in top.groupby("timestamp"):
        fc = g["fwd_close_return"]
        row = {
            "timestamp": ts,
            "n": len(g),
            "breadth_pos": (fc > 0).mean(),
            "breadth_5pct": (fc >= 0.05).mean(),
            "breadth_alpha": (g["fwd_max_alpha"] > 0).mean(),
            "port_mean": fc.mean(),
            "port_med": fc.median(),
            "worst": fc.min(),
            "best": fc.max(),
        }
        for c in MARKET_REGIME:
            if c in g.columns:
                row[c] = g[c].mean()
        for c in BASKET_DESC:
            if c in g.columns:
                row["bk_" + c] = g[c].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--since", default="2025-05-01")
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--show-day", default=None, help="ET/UTC date to dump the top-K names + fwd returns")
    args = ap.parse_args()

    m = pd.read_parquet(args.matrix).reset_index()
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
    m = m[m["timestamp"] >= args.since]
    m = m[m["fwd_close_return"].notna() & m["fwd_max_return"].notna()].copy()
    print(f"scoring {m['timestamp'].nunique()} labeled bars since {args.since} ({len(m):,} rows)...")
    sc = score_frame(m)

    day = build_day_table(sc, args.k)
    print(f"\n=== day-level basket (top-{args.k}) outcome distribution, {len(day)} bars ===")
    desc = day[["breadth_pos", "breadth_5pct", "breadth_alpha", "port_mean", "port_med"]].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    print((desc * 100).round(1).to_string())
    print("\nReading: breadth_pos p10..p90 shows how much the 'how many of 15 worked' varies day to day.")

    # ---- predictive correlation: entry-time regime vs forward breadth ----
    feats = [c for c in MARKET_REGIME + ["bk_" + c for c in BASKET_DESC] if c in day.columns]
    cor = []
    for c in feats:
        x = day[c]
        if x.nunique() < 3:
            continue
        cor.append({
            "feature": c,
            "spearman_vs_breadth_pos": x.corr(day["breadth_pos"], method="spearman"),
            "spearman_vs_port_mean": x.corr(day["port_mean"], method="spearman"),
        })
    cor = pd.DataFrame(cor)
    cor["abs"] = cor["spearman_vs_breadth_pos"].abs()
    cor = cor.sort_values("abs", ascending=False).drop(columns="abs")
    pd.set_option("display.width", 200)
    print(f"\n=== entry-time regime -> forward breadth (Spearman across {len(day)} bars) ===")
    print(cor.round(3).to_string(index=False))
    print("Positive = that condition at entry precedes a broader/higher basket outcome.")

    # ---- terciles: bad vs good breadth days, and the regime they sat in ----
    day["bucket"] = pd.qcut(day["breadth_pos"], 3, labels=["bad", "mid", "good"], duplicates="drop")
    grp = day.groupby("bucket", observed=True)
    summ_cols = ["breadth_pos", "port_mean", "port_med", "worst"] + feats
    summ = grp[summ_cols].mean()
    for c in ["breadth_pos", "port_mean", "port_med", "worst"]:
        summ[c] = (summ[c] * 100).round(1)
    print(f"\n=== regime by breadth tercile (mean of each col; returns/breadth in %) ===")
    print(summ.round(3).T.to_string())
    print("\nCompare the 'bad' vs 'good' columns: a feature that monotonically shifts is a day-quality tell.")

    if args.show_day:
        d0 = pd.Timestamp(args.show_day, tz="UTC")
        sub = sc[(sc["timestamp"] >= d0) & (sc["timestamp"] < d0 + pd.Timedelta(days=1))]
        if len(sub):
            sub = sub.copy()
            sub["rank"] = sub.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
            for ts, g in sub.groupby("timestamp"):
                g = g[g["rank"] <= args.k].sort_values("rank")
                print(f"\n--- {ts}  top-{args.k} ---")
                cols = ["rank", "ticker", "s_combo", "fwd_close_return", "fwd_max_return", "fwd_max_alpha"]
                show = g[cols].copy()
                for c in ["fwd_close_return", "fwd_max_return", "fwd_max_alpha"]:
                    show[c] = (show[c] * 100).round(1)
                print(show.to_string(index=False))
                fc = g["fwd_close_return"]
                print(f"  breadth_pos={ (fc>0).mean():.2f}  port_mean={fc.mean()*100:.1f}%  "
                      f"regime_vix_z={g['regime_vix_z'].mean():.2f}  spy_trend={g['regime_spy_trend'].mean():.2f}")
        else:
            print(f"\n(no bars found for {args.show_day})")


if __name__ == "__main__":
    main()
