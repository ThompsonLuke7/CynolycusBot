"""Does an entry-day GATE help ACROSS regimes (not just the 2025-26 bull) — and does it
cut more losers than winners?

The shipped meta matrix only goes back to 2025-05 (all bull). To see bear/chop we use the
MOMENTUM module OOF (`expansion_v1/oof_preds.parquet`, 2022-11 -> 2026-05, 4391 bars, leak-free
out-of-fold) as the selection proxy: it is the dominant meta leg, carries the same forward
labels, and already holds `trend_persistence`. We attach a REAL at-entry market regime from
SPY 4H bars (backward-looking only -> no look-ahead).

For each bar: rank by momentum score, take top-K, then
  breadth_pos = frac of top-K with fwd_close_return > 0      (the "how many worked" the user means)
  port_mean   = mean fwd_close_return of the basket
  basket_tp   = mean trend_persistence of the picks          (candidate gate)
  market_tp   = mean trend_persistence of ALL names that bar (regime proxy, at entry)
  + SPY regime: above 50-bar MA, 20-bar return, drawdown from 60-bar high, realized vol.

Two outputs:
  1. REGIME TABLE — breadth/return by SPY regime and by period; does trend even exist off-bull?
  2. GATE EVAL — for each candidate gate (known at entry), losers_cut vs winners_cut, the win
     rate of what's cut, and how much total return is retained. A good gate cuts mostly losers.

  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/gate_regime_backtest.py --k 15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
MOM_OOF = REPO / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet"
SPY = REPO / "Data/shared/bars/4h/SPY.parquet"


def spy_regime() -> pd.DataFrame:
    b = pd.read_parquet(SPY, columns=["timestamp", "close", "high", "low"])
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    b = b.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    c = b["close"]
    b["spy_ma50"] = c.rolling(50).mean()
    b["spy_above_50ma"] = (c > b["spy_ma50"]).astype(float)
    b["spy_ret_20"] = c.pct_change(20)
    b["spy_dd"] = c / c.rolling(60).max() - 1.0          # drawdown from trailing high
    b["spy_vol_20"] = c.pct_change().rolling(20).std()
    return b[["timestamp", "spy_above_50ma", "spy_ret_20", "spy_dd", "spy_vol_20"]]


def build(k: int) -> pd.DataFrame:
    oof = pd.read_parquet(MOM_OOF).reset_index()
    oof["timestamp"] = pd.to_datetime(oof["timestamp"], utc=True)
    oof = oof[oof["fwd_close_return"].notna()].copy()
    # market-wide trend persistence (regime proxy, backward-looking)
    mkt_tp = oof.groupby("timestamp")["trend_persistence"].mean().rename("market_tp")
    oof["rank"] = oof.groupby("timestamp")["score"].rank(ascending=False, method="first")
    top = oof[oof["rank"] <= k]
    g = top.groupby("timestamp")
    day = pd.DataFrame({
        "n": g.size(),
        "breadth_pos": g["fwd_close_return"].apply(lambda s: (s > 0).mean()),
        "port_mean": g["fwd_close_return"].mean(),
        "port_med": g["fwd_close_return"].median(),
        "worst": g["fwd_close_return"].min(),
        "basket_tp": g["trend_persistence"].mean(),
    }).join(mkt_tp).reset_index()
    reg = spy_regime()
    day = pd.merge_asof(day.sort_values("timestamp"), reg.sort_values("timestamp"),
                        on="timestamp", direction="backward")
    day["year"] = day["timestamp"].dt.year
    return day, top


def regime_label(r):
    if r["spy_above_50ma"] != 1:
        return "downtrend (<50ma)"
    if r["spy_dd"] <= -0.05:
        return "pullback (>50ma, dd<=-5%)"
    return "uptrend (>50ma, shallow)"


def gate_eval(top: pd.DataFrame, day: pd.DataFrame, gate_bars: pd.Index, name: str) -> dict:
    """Treat each top-K (bar,name) as a trade. gate_bars = timestamps the gate KEEPS."""
    r = top.set_index("timestamp")["fwd_close_return"]
    kept_mask = r.index.isin(gate_bars)
    kept, cut = r[kept_mask], r[~kept_mask]
    base_total = r.sum()
    winners_cut = int((cut > 0).sum())
    losers_cut = int((cut <= 0).sum())
    return {
        "gate": name,
        "bars_kept": int(day["timestamp"].isin(gate_bars).sum()),
        "bars_cut": int((~day["timestamp"].isin(gate_bars)).sum()),
        "trades_cut": len(cut),
        "win%_cut": round((cut > 0).mean() * 100, 1) if len(cut) else np.nan,
        "mean_cut%": round(cut.mean() * 100, 1) if len(cut) else np.nan,
        "losers/winners_cut": round(losers_cut / max(winners_cut, 1), 2),
        "kept_mean%": round(kept.mean() * 100, 1),
        "kept_win%": round((kept > 0).mean() * 100, 1),
        "kept_ret/std": round(kept.mean() / kept.std(), 3) if kept.std() else np.nan,
        "ret_retained%": round(kept.sum() / base_total * 100, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    day, top = build(args.k)
    print(f"momentum-OOF proxy: {len(day)} bars {day.timestamp.min().date()} -> {day.timestamp.max().date()}, "
          f"top-{args.k}/bar\n")

    # ---------- 1. regime / period breakdown ----------
    day["regime"] = day.apply(regime_label, axis=1)
    print("=== breadth & basket return by SPY regime AT ENTRY ===")
    rt = day.groupby("regime").agg(
        bars=("timestamp", "size"),
        breadth_pos=("breadth_pos", "mean"),
        port_mean=("port_mean", "mean"),
        worst=("worst", "mean"),
        basket_tp=("basket_tp", "mean"),
        market_tp=("market_tp", "mean"),
    )
    for c in ["breadth_pos", "port_mean", "worst"]:
        rt[c] = (rt[c] * 100).round(1)
    print(rt.round(3).to_string())

    print("\n=== same, by year (regime mix over time) ===")
    yt = day.groupby("year").agg(
        bars=("timestamp", "size"), breadth_pos=("breadth_pos", "mean"),
        port_mean=("port_mean", "mean"), pct_above50ma=("spy_above_50ma", "mean"),
        basket_tp=("basket_tp", "mean"), market_tp=("market_tp", "mean"),
    )
    for c in ["breadth_pos", "port_mean", "pct_above50ma"]:
        yt[c] = (yt[c] * 100).round(1)
    print(yt.round(3).to_string())

    # does trend_persistence predict breadth WITHIN each regime? (bull-artifact test)
    print("\n=== Spearman(basket_tp, breadth_pos) and Spearman(market_tp, breadth_pos) WITHIN regime ===")
    for rg, gd in day.groupby("regime"):
        print(f"  {rg:30} n={len(gd):4}  basket_tp->breadth {gd['basket_tp'].corr(gd['breadth_pos'],method='spearman'):+.3f}"
              f"   market_tp->breadth {gd['market_tp'].corr(gd['breadth_pos'],method='spearman'):+.3f}"
              f"   spy_dd->breadth {gd['spy_dd'].corr(gd['breadth_pos'],method='spearman'):+.3f}")

    # ---------- 2. gate evaluation (cut more losers than winners?) ----------
    print(f"\n=== GATE EVAL — each top-{args.k} (bar,name) is a trade; gate decides at entry ===")
    base = top["fwd_close_return"]
    print(f"BASELINE (no gate): trades={len(base)}  mean={base.mean()*100:.1f}%  win={ (base>0).mean()*100:.1f}%  "
          f"ret/std={base.mean()/base.std():.3f}")
    rows = []
    allbars = day["timestamp"]
    # quantile-based floors so thresholds are regime-agnostic
    btp = day["basket_tp"]; mtp = day["market_tp"]
    gates = {
        "SPY above 50ma":                 day.loc[day["spy_above_50ma"] == 1, "timestamp"],
        "SPY dd > -5%":                   day.loc[day["spy_dd"] > -0.05, "timestamp"],
        "SPY dd > -10%":                  day.loc[day["spy_dd"] > -0.10, "timestamp"],
        "SPY vol_20 <= 75pct":            day.loc[day["spy_vol_20"] <= day["spy_vol_20"].quantile(.75), "timestamp"],
        "basket_tp >= 25pct":             day.loc[btp >= btp.quantile(.25), "timestamp"],
        "basket_tp >= 50pct":             day.loc[btp >= btp.quantile(.50), "timestamp"],
        "market_tp >= 25pct":             day.loc[mtp >= mtp.quantile(.25), "timestamp"],
        "market_tp >= 50pct":             day.loc[mtp >= mtp.quantile(.50), "timestamp"],
        "above50ma & dd>-5%":             day.loc[(day["spy_above_50ma"]==1)&(day["spy_dd"]>-0.05), "timestamp"],
        "above50ma & market_tp>=25pct":   day.loc[(day["spy_above_50ma"]==1)&(mtp>=mtp.quantile(.25)), "timestamp"],
    }
    for name, bars in gates.items():
        rows.append(gate_eval(top, day, pd.Index(bars.unique()), name))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print(out.to_string(index=False))
    print("\nWANT: losers/winners_cut > 1 (cuts more losers), kept_mean% & kept_ret/std up vs baseline,")
    print("      ret_retained% high (not throwing away winners). A gate that cuts winners shows win%_cut high.")


if __name__ == "__main__":
    main()
