"""
Example-trade plots for the EV-optimization final configs (research/capstone/
ev_optimization_4h.md). Re-simulates the already-frozen test-window configs
(no re-tuning — same policy as the E4 one-shot run) to recover per-trade
detail for plotting and for reporting avg-win/avg-loss magnitudes, since the
one-shot run only kept aggregate metrics.

Reuses strategies/momentum_expansion/backtest/plot_family_compare.py's
example-trades panel convention (price path + TP/SL lines + entry/exit
markers) and the same StrategyConfig / BarCache / simulate engine as
family_backtest.py so the plotted trades match the reported numbers exactly.

Usage:
  PYTHONPATH=. .venv/bin/python backtests/ev_experiments_4h_plots.py --strategy all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtests.ev_experiments_4h import _cfg, _deployed
from scripts.capstone.family_backtest_clean import compute_cutoffs, load_window
from strategies.momentum_expansion.backtest import family_backtest as fb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtests" / "ev_experiments_4h"

FINAL_CONFIGS = {
    "momentum": dict(label="final_balanced_k3_z2", top_k=3, conviction_z=2.0,
                     side="long_only", tp=2.0, sl=5.0, hold=75),
    "htf": dict(label="final_ev_k5_z1_long", top_k=5, conviction_z=1.0,
               side="long_only", tp=6.0, sl=5.0, hold=25),
}
BASELINE_CONFIGS = {
    "momentum": dict(label="baseline_clean", top_k=5, conviction_z=None,
                     side="long_only", tp=2.0, sl=4.0, hold=75),
    "htf": dict(label="baseline_clean", top_k=20, conviction_z=None,
               side="both", tp=5.0, sl=2.0, hold=25),
}


def simulate_test(strategy: str, config: dict) -> tuple[pd.DataFrame, fb.BarCache, str, int]:
    cfg = _cfg(strategy)
    train_end, test_start = compute_cutoffs(cfg)
    df = load_window(cfg, "test", train_end, test_start)
    family, seed = _deployed(cfg)
    df["score"] = fb.score_family(df, cfg, family, seed)
    g = df.groupby("timestamp")["score"]
    df["conviction_z"] = (df["score"] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)

    allow_short = cfg.allow_short and config.get("side", "both") != "long_only"
    sig = fb.select_signals(df[["timestamp", "ticker", "score"]], int(config["top_k"]), allow_short)
    if config.get("conviction_z"):
        sig = sig.merge(df[["timestamp", "ticker", "conviction_z"]], on=["timestamp", "ticker"])
        sig = sig[sig["conviction_z"] * sig["direction"] >= float(config["conviction_z"])].drop(columns="conviction_z")

    cache = fb.BarCache()
    trades = fb.simulate(sig, cache, tp_mult=float(config["tp"]), sl_mult=float(config["sl"]),
                         max_hold=int(config["hold"]))
    trades = trades.sort_values("exit_ts").reset_index(drop=True)
    return trades, cache, family, seed


def report_win_loss_asymmetry(strategy: str, trades: pd.DataFrame, config: dict) -> None:
    wins = trades.loc[trades.pnl_pct > 0, "pnl_pct"]
    losses = trades.loc[trades.pnl_pct <= 0, "pnl_pct"]
    m = fb.metrics(trades)
    print(f"\n=== {strategy} [{config['label']}] win/loss asymmetry (test, n={len(trades)}) ===")
    print(f"  win_rate={m['win_rate']:.3f}  avg_trade_pct(EV)={m['avg_trade_pct']:+.3f}%  "
          f"profit_factor={m['profit_factor']:.3f}")
    print(f"  avg WIN  = {wins.mean()*100:+.3f}%  (n={len(wins)})")
    print(f"  avg LOSS = {losses.mean()*100:+.3f}%  (n={len(losses)})")
    print(f"  win:loss size ratio = {abs(wins.mean()/losses.mean()):.2f}x")
    print(f"  exit mix: tp={m['tp_rate']:.1%}  sl={m['sl_rate']:.1%}  time={m['time_rate']:.1%}")
    by_reason = trades.groupby("exit_reason")["pnl_pct"].mean() * 100
    print(f"  avg pnl by exit reason: {by_reason.round(2).to_dict()}")


def plot_examples(strategy: str, trades: pd.DataFrame, cache: fb.BarCache,
                  family: str, seed: int, config: dict, out_path: Path) -> None:
    """2 representative GOOD trades: best TP-hit winner + best time-stop winner
    (the latter shows the 'let winners run past TP-equivalent' dynamic that
    drives EV on the wide-stop configs — see ev_optimization_4h.md)."""
    ex = []
    tp_wins = trades[(trades.exit_reason == "tp") & (trades.pnl_pct > 0)]
    time_wins = trades[(trades.exit_reason == "time") & (trades.pnl_pct > 0)]
    if not tp_wins.empty:
        ex.append(("Best TP-hit winner", tp_wins.loc[tp_wins.pnl_pct.idxmax()]))
    if not time_wins.empty:
        ex.append(("Best time-stop winner", time_wins.loc[time_wins.pnl_pct.idxmax()]))
    if len(ex) < 2:
        rest = trades[trades.pnl_pct > 0].nlargest(2 - len(ex), "pnl_pct")
        ex += [("Winner", r) for _, r in rest.iterrows()]

    n = len(ex)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.6))
    axes = np.atleast_1d(axes)
    for ax, (tag, tr) in zip(axes, ex):
        b = cache.get(tr.ticker)
        e0, e1 = int(tr.entry_i), int(tr.exit_i)
        lo, hi = max(0, e0 - 10), min(len(b["ts_dt"]), e1 + 10)
        x = pd.to_datetime(b["ts_dt"][lo:hi])
        ax.plot(x, b["close"][lo:hi], color="#444", lw=1.3)
        ax.axhline(tr.tp_price, color="#1a9850", ls=":", lw=1.2, label="TP level")
        ax.axhline(tr.sl_price, color="#d73027", ls=":", lw=1.2, label="SL level")
        et, xt = pd.to_datetime(tr.entry_ts), pd.to_datetime(tr.exit_ts)
        ax.scatter([et], [tr.entry_price], marker="^" if tr.direction > 0 else "v",
                   color="#2b8cbe", s=110, zorder=5, label="entry", edgecolor="white")
        ax.scatter([xt], [tr.exit_price], marker="x", color="#000", s=110, zorder=5, label="exit")
        ax.set_title(f"{tag}: {tr.ticker} {'LONG' if tr.direction>0 else 'SHORT'}  "
                     f"{tr.pnl_pct*100:+.2f}%  [{tr.exit_reason}, {tr.bars_held} bars held]",
                     fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"{strategy}_expansion final config [{config['label']}]  "
        f"(top{config['top_k']} z>={config.get('conviction_z') or 0} {config['side']} "
        f"tp{config['tp']}/sl{config['sl']}/hold{config['hold']}) — {family} s{seed} — "
        f"FROZEN TEST WINDOW, one-shot",
        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[{strategy}] wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["momentum", "htf", "all"], default="all")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    strategies = ["momentum", "htf"] if args.strategy == "all" else [args.strategy]
    for s in strategies:
        trades, cache, family, seed = simulate_test(s, FINAL_CONFIGS[s])
        report_win_loss_asymmetry(s, trades, FINAL_CONFIGS[s])
        trades.to_parquet(OUT / f"{s}_final_frozen_test_trades.parquet", index=False)
        plot_examples(s, trades, cache, family, seed, FINAL_CONFIGS[s],
                     OUT / f"{s}_final_example_trades.png")

        base_trades, base_cache, bf, bs = simulate_test(s, BASELINE_CONFIGS[s])
        report_win_loss_asymmetry(s, base_trades, BASELINE_CONFIGS[s])


if __name__ == "__main__":
    main()
