"""
Plots for the rankers-vs-classifiers / best-order-policy experiment.

For the overall-best model of a strategy (read from comparison_summary.json) it produces:
  - equity_curve.png        : cumulative P&L of the best family @ its best policy
  - clf_vs_ranker.png       : best ret/DD per family (classifiers vs rankers)
  - score_lift.png          : test-set decile lift (mean forward return by score decile)
  - example_trades.png      : 4H price panels for a few representative trades with
                              entry / TP / SL / exit markers + realized return.

Usage:
  .venv/bin/python -m strategies.momentum_expansion.backtest.plot_family_compare --strategy momentum
  .venv/bin/python -m strategies.momentum_expansion.backtest.plot_family_compare --strategy htf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.momentum_expansion.backtest import family_backtest as fb
from strategies.momentum_expansion.backtest.run_family_compare import STRATEGIES

REL_SRC = {"momentum": "fwd_max_return", "htf": "fwd_best_high_return"}


def _best(summary):
    b = summary["overall_best_by_ret_over_dd"]
    return (b["family"], int(b["seed"]), int(b["top_k"]),
            float(b["tp_atr_mult"]), float(b["sl_atr_mult"]), int(b["max_hold"]))


def main(strategy: str) -> None:
    spec = STRATEGIES[strategy]
    res = Path(spec["results_dir"])
    summary = json.loads((res / "comparison_summary.json").read_text())
    family, seed, top_k, tp, sl, hold = _best(summary)
    cfg = fb.StrategyConfig.from_manifest(
        spec["name"], spec["models_dir"], spec["matrix_path"],
        allow_short=spec["allow_short"], forward_window=spec["forward_window"])

    plot_dir = Path(spec["plot_dir"]); plot_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_parquet(res / "scores" / f"{family}_s{seed}.parquet")
    sig = fb.select_signals(scored[["timestamp", "ticker", "score"]], top_k, cfg.allow_short)
    bars = fb.BarCache()
    trades = fb.simulate(sig, bars, tp_mult=tp, sl_mult=sl, max_hold=hold)
    trades = trades.sort_values("exit_ts").reset_index(drop=True)

    title = f"{cfg.name} — best: {family} s{seed} (tp={tp} sl={sl} topk={top_k} hold={hold})"

    # 1) equity curve
    eq = fb.INITIAL_CAPITAL + trades["pnl_dollar"].cumsum()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(pd.to_datetime(trades["exit_ts"]), eq, lw=1.3)
    ax.set_title(f"Equity curve (gross, $1k/trade fixed notional)\n{title}", fontsize=10)
    ax.set_ylabel("equity ($)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plot_dir / "equity_curve.png", dpi=120); plt.close(fig)

    # 2) clf vs ranker (best ret/DD per family)
    bbf = pd.DataFrame(summary["best_by_family"])
    order = ["xgb_classifier", "xgb_ranker", "lgbm_classifier", "lgbm_ranker"]
    bbf = bbf.set_index("family").reindex(order)
    colors = ["#2b8cbe" if "classifier" in f else "#e34a33" for f in order]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(range(len(order)), bbf["ret_over_dd"].values, color=colors)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("best ret / |maxDD|")
    h2h = summary["clf_vs_ranker"]["matched_policy_head_to_head"]
    ax.set_title(f"Classifiers (blue) vs Rankers (red) — winner: {summary['clf_vs_ranker']['winner']}\n"
                 f"matched-policy clf>ranker: {h2h}", fontsize=9)
    for i, v in enumerate(bbf["ret_over_dd"].values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(plot_dir / "clf_vs_ranker.png", dpi=120); plt.close(fig)

    # 3) decile lift: mean forward return by score decile on the test set
    relcol = REL_SRC[strategy]
    sc = scored.dropna(subset=["score", relcol]).copy()
    sc["decile"] = pd.qcut(sc["score"], 10, labels=False, duplicates="drop")
    lift = sc.groupby("decile")[relcol].mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(lift.index, lift.values, color="#31a354")
    ax.axhline(sc[relcol].mean() * 100, color="k", ls="--", lw=1, label="test mean")
    ax.set_xlabel("model score decile (9 = highest)"); ax.set_ylabel(f"mean {relcol} (%)")
    ax.set_title(f"Score → forward-return lift (test set)\n{family} s{seed}", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(plot_dir / "score_lift.png", dpi=120); plt.close(fig)

    # 4) example trades: pick a spread (2 big winners, 1 stop-out, 1 short if available)
    ex = []
    longs = trades[trades.direction == 1]
    ex += list(longs.nlargest(2, "pnl_pct").index)
    ex += list(trades[trades.exit_reason == "sl"].nsmallest(1, "pnl_pct").index)
    if cfg.allow_short and (trades.direction == -1).any():
        ex += list(trades[trades.direction == -1].nlargest(1, "pnl_pct").index)
    else:
        ex += list(longs.iloc[[len(longs)//2]].index) if len(longs) else []
    ex = list(dict.fromkeys(ex))[:4]

    n = len(ex); fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2)); axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, ex):
        tr = trades.loc[idx]
        b = bars.get(tr.ticker)
        e0 = int(tr.entry_i); e1 = int(tr.exit_i)
        lo = max(0, e0 - 8); hi = min(len(b["ts_dt"]), e1 + 8)
        x = pd.to_datetime(b["ts_dt"][lo:hi])
        ax.plot(x, b["close"][lo:hi], color="#444", lw=1.1)
        ax.axhline(tr.tp_price, color="#1a9850", ls=":", lw=1, label="TP")
        ax.axhline(tr.sl_price, color="#d73027", ls=":", lw=1, label="SL")
        et = pd.to_datetime(tr.entry_ts); xt = pd.to_datetime(tr.exit_ts)
        ax.scatter([et], [tr.entry_price], marker="^" if tr.direction > 0 else "v",
                   color="#2b8cbe", s=70, zorder=5, label="entry")
        ax.scatter([xt], [tr.exit_price], marker="x", color="#000", s=70, zorder=5, label="exit")
        ax.set_title(f"{tr.ticker} {'LONG' if tr.direction>0 else 'SHORT'}  "
                     f"{tr.pnl_pct*100:+.1f}% [{tr.exit_reason}]", fontsize=9)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Example test trades — {family} s{seed}", fontsize=11)
    fig.tight_layout(); fig.savefig(plot_dir / "example_trades.png", dpi=120); plt.close(fig)

    print(f"[{strategy}] wrote 4 plots to {plot_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=list(STRATEGIES), required=True)
    main(p.parse_args().strategy)
