"""
Capstone paper figure set — generated ONLY from the locked artifacts registered
in scripts/capstone/reproduce_results.py (no retraining, no network).

Every figure carries a provenance footer: date window, universe, split tag
(test(frozen) / wf-oof / paper / reference), benchmark, source artifacts, and
the git commit + generation date. Split tags follow results_lock.json.

Figure inventory (research/capstone/figures/):
  fig01  frozen-test equity curves vs SPY buy-and-hold
  fig02  frozen-test drawdown (underwater) curves vs SPY
  fig03  rolling 63-day Sharpe of daily booked P&L vs SPY
  fig04  selection-bias correction: biased vs val-selected/test-frozen results
  fig05  per-regime performance (SPY 200d-SMA risk-on/risk-off at entry)
  fig06  trade-return distributions (frozen test)
  fig07  hold-time distributions (frozen test + paper ledger)
  fig08  walk-forward OOF score-decile lift (mom / HTF / meta q / u / combo)
  fig09  meta ranker OOF calibration (quality & upside)
  fig10  feature importance — the five deployed winners
  fig11  meta exit-policy comparison (OOF-scored holdout, results_lock)
  fig12  swing paper-trading sessions 2026-05-28/29 (options ledger)

Conventions (stated on the figures):
  * Event-driven backtests book $1,000 notional per trade on a $100k base, NOT
    compounded, concurrency unconstrained — the same convention as the locked
    family_compare_clean / sweep_v2_clean artifacts.
  * P&L is booked at trade exit; days with no exits contribute zero.
  * Swing trade timestamps are recovered by positional index into the same raw
    30m caches the backtest simulated on, and every trade's entry price is
    verified against that bar (abort if <99% consistent).

Palette: light (paper) theme, validated with the dataviz palette checks
(lightness band, chroma floor, Machado-2009 CVD separation >= 12 dE on adjacent
pairs, WCAG contrast vs white; series are also direct-labeled).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/make_figures.py            # all
  PYTHONPATH=. .venv/bin/python scripts/capstone/make_figures.py --only fig01,fig08
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.shared_plotting import LIGHT_THEME, apply_mpl_defaults, save_figure, style_figure
from scripts.capstone.reproduce_results import ARTIFACTS, REPO, _read_oof

FIG_DIR = REPO / "research" / "capstone" / "figures"
LOCK_PATH = REPO / "research" / "capstone" / "results_lock.json"

THEME = LIGHT_THEME

# Fixed series colors — color follows the entity across every figure.
# Validated (Python port of dataviz validate_palette.js, surface #ffffff):
# lightness band PASS, chroma floor PASS, worst adjacent CVD dE >= 12 PASS.
COLORS = {
    "momentum": "#2a78d6",   # blue
    "htf_swing": "#199e70",  # aqua (dark step — clears 3:1 on white)
    "swing": "#4a3aa7",      # violet
    "meta": "#008300",       # green
    "spy": "#898781",        # neutral benchmark gray
    "biased": "#c3c2b7",     # pre-audit (selection-biased) values
}
LABELS = {
    "momentum": "Momentum 4H",
    "htf_swing": "HTF Swing 4H",
    "swing": "Swing 30m",
    "meta": "Meta Ranker",
    "spy": "SPY buy & hold",
}

BASE_K = 100.0          # $100k base equity, $1k notional per trade
BARS_PER_DAY_4H = 2.0   # regular-session 4H bars per trading day
BARS_PER_DAY_30M = 13.0


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


GIT_SHA = _git_sha()
GEN_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def footer(fig, *, window: str, universe: str, split: str, source: str,
           benchmark: str | None = None, note: str | None = None) -> None:
    """Standard provenance footer required on every capstone figure."""
    line1 = f"Window: {window}    Universe: {universe}    Split: {split}"
    if benchmark:
        line1 += f"    Benchmark: {benchmark}"
    line2 = f"Source: {source}    |    git {GIT_SHA}, generated {GEN_DATE} by scripts/capstone/make_figures.py"
    if note:
        line2 = f"{note}\n{line2}"
    fig.text(0.01, -0.015, line1 + "\n" + line2, ha="left", va="top",
             fontsize=7.2, color=THEME.muted_text, linespacing=1.5)


def _direct_label(ax, x, y, text, color):
    ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=8.5, fontweight="bold", va="center",
                annotation_clip=False)


# ---------------------------------------------------------------------------
# Loaders (locked artifacts only)
# ---------------------------------------------------------------------------

def load_mom_trades() -> tuple[pd.DataFrame, dict]:
    summ = json.loads((ARTIFACTS["mom_family_clean"]).read_text())
    path = ARTIFACTS["mom_family_clean"].parent / "xgb_classifier_s45_frozen_test_trades.parquet"
    df = pd.read_parquet(path).sort_values("exit_ts").reset_index(drop=True)
    assert len(df) == summ["deployed_winner_frozen_test"]["trades"], "momentum trades != locked count"
    return df, summ


def load_htf_trades() -> tuple[pd.DataFrame, dict]:
    summ = json.loads((ARTIFACTS["htf_family_clean"]).read_text())
    path = ARTIFACTS["htf_family_clean"].parent / "lgbm_classifier_s46_frozen_test_trades.parquet"
    df = pd.read_parquet(path).sort_values("exit_ts").reset_index(drop=True)
    assert len(df) == summ["deployed_winner_frozen_test"]["trades"], "HTF trades != locked count"
    return df, summ


def load_swing_trades() -> pd.DataFrame:
    """Swing clean trades + timestamps recovered from the raw 30m caches.

    best_v2_clean_trades.parquet stores positional bar indices into each
    ticker's raw 30m frame (sweep_v2.TickerData). Recover entry/exit
    timestamps and verify each trade's entry price against that bar — a
    regenerated/shifted cache would break the price check, so we abort unless
    >= 99% of trades are price-consistent.
    """
    from strategies.multi_ticker_swing.backtest.sweep_v2 import load_raw_30m

    tr = pd.read_parquet(ARTIFACTS["swing_bt_clean"].parent / "best_v2_clean_trades.parquet")
    entry_ts = np.full(len(tr), np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_ts = np.full(len(tr), np.datetime64("NaT"), dtype="datetime64[ns]")
    consistent = np.zeros(len(tr), dtype=bool)
    for ticker, sub in tr.groupby("ticker"):
        raw = load_raw_30m(ticker)
        ts = pd.to_datetime(raw["timestamp"].values).values
        o = raw["open"].values
        h = raw["high"].values
        lo = raw["low"].values
        c = raw["close"].values
        n = len(ts)
        for i, r in sub.iterrows():
            si, xi = int(r.signal_idx), int(r.exit_idx)
            if xi >= n:
                continue
            entry_ts[i] = ts[si]
            exit_ts[i] = ts[xi]
            e = r.entry_price
            near = (abs(c[si] - e) / e < 1e-3) or (si + 1 < n and abs(o[si + 1] - e) / e < 1e-3)
            in_bar = (lo[si] <= e <= h[si]) or (si + 1 < n and lo[si + 1] <= e <= h[si + 1])
            consistent[i] = near or in_bar
    frac = consistent.mean()
    if frac < 0.99:
        raise RuntimeError(f"swing index→timestamp mapping unverified: only {frac:.1%} price-consistent")
    tr["entry_ts"] = pd.to_datetime(entry_ts, utc=True)
    tr["exit_ts"] = pd.to_datetime(exit_ts, utc=True)
    tr["pnl_dollar"] = tr["pnl_pct"] * 1000.0
    tr = tr.dropna(subset=["entry_ts", "exit_ts"]).sort_values("exit_ts").reset_index(drop=True)
    print(f"  swing mapping: {frac:.2%} price-consistent, {len(tr)} trades, "
          f"{tr.entry_ts.min():%Y-%m-%d} → {tr.exit_ts.max():%Y-%m-%d}")
    return tr


def load_spy() -> pd.DataFrame:
    spy = pd.read_parquet(ARTIFACTS["spy_1d_bars"])[["timestamp", "close"]].copy()
    spy["date"] = pd.to_datetime(spy["timestamp"]).dt.tz_localize(None).dt.normalize()
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["sma200"] = spy["close"].rolling(200).mean()
    spy["risk_on"] = spy["close"] >= spy["sma200"]
    spy["ret"] = spy["close"].pct_change()
    return spy


def load_lock() -> dict:
    lock = json.loads(LOCK_PATH.read_text())
    return {(m["model"], m["metric"]): m for m in lock["metrics"]}


def strategies_bundle():
    mom, mom_summ = load_mom_trades()
    htf, htf_summ = load_htf_trades()
    swing = load_swing_trades()
    wins = {
        "momentum": (pd.Timestamp(mom_summ["test_window_start"]), mom.exit_ts.max()),
        "htf_swing": (pd.Timestamp(htf_summ["test_window_start"]), htf.exit_ts.max()),
        "swing": (swing.entry_ts.min(), swing.exit_ts.max()),
    }
    trades = {"momentum": mom, "htf_swing": htf, "swing": swing}
    universes = {
        "momentum": f"{mom.ticker.nunique()} tickers traded (top-5/bar of ranked pool)",
        "htf_swing": f"{htf.ticker.nunique()} tickers traded (top-20/bar of ranked pool)",
        "swing": f"{swing.ticker.nunique()} tickers traded (top-100 val-selected, blacklist-filtered)",
    }
    return trades, wins, universes


def equity_series(df: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    """$k equity: $100k base + cumulative $1k-notional trade P&L, booked at exit."""
    eq = BASE_K + df.groupby("exit_ts")["pnl_dollar"].sum().sort_index().cumsum() / 1000.0
    eq = pd.concat([pd.Series([BASE_K], index=[start]), eq])
    return eq


def daily_pnl(df: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    booked = df.groupby(df["exit_ts"].dt.normalize())["pnl_dollar"].sum()
    booked.index = booked.index.tz_localize(None)
    return booked.reindex(days, fill_value=0.0)


def _win_str(wins: dict) -> str:
    return " / ".join(f"{LABELS[k]} {a:%Y-%m-%d}→{b:%Y-%m-%d}" for k, (a, b) in wins.items())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig01_equity(trades, wins, universes, spy):
    fig, ax = plt.subplots(figsize=(12, 6))
    style_figure(fig, ax, THEME)
    t0 = min(w[0] for w in wins.values())
    t1 = max(w[1] for w in wins.values())
    s = spy[(spy.date >= t0.tz_localize(None)) & (spy.date <= t1.tz_localize(None))]
    spy_eq = BASE_K * s.close / s.close.iloc[0]
    ax.plot(s.date, spy_eq, color=COLORS["spy"], lw=1.6, ls="--", label=LABELS["spy"])
    for key in ("momentum", "htf_swing", "swing"):
        eq = equity_series(trades[key], wins[key][0])
        ax.plot(eq.index, eq.values, color=COLORS[key], lw=2.0, label=LABELS[key])
    ax.set_xlim(t0, t1 + pd.Timedelta(days=55))
    for key in ("momentum", "htf_swing", "swing"):
        eq = equity_series(trades[key], wins[key][0])
        _direct_label(ax, eq.index[-1], eq.values[-1], f"{LABELS[key]}  {eq.values[-1] - BASE_K:+.0f}k",
                      COLORS[key])
    _direct_label(ax, s.date.iloc[-1], spy_eq.iloc[-1], f"SPY {spy_eq.iloc[-1] - BASE_K:+.0f}k", COLORS["spy"])
    ax.axhline(BASE_K, color=THEME.spine, lw=0.8)
    ax.set_ylabel("Equity ($k) — $100k base, $1k notional per trade, not compounded")
    ax.set_title("Frozen-test equity curves vs SPY — policies selected on VAL, frozen on TEST")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    footer(fig,
           window=_win_str(wins),
           universe="; ".join(universes[k] for k in ("momentum", "htf_swing", "swing")),
           split="test(frozen)", benchmark="SPY (same $100k base)",
           source="family_compare_clean/*_frozen_test_trades.parquet, sweep_v2_clean/best_v2_clean_trades.parquet, Data/shared/bars/1d/SPY.parquet",
           note="P&L booked at trade exit; per-trade notional fixed at $1k; concurrent-position count unconstrained (see figures/README.md).")
    return save_figure(fig, FIG_DIR / "fig01_equity_curves.png", dpi=200, close=True)


def fig02_drawdown(trades, wins, universes, spy):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    style_figure(fig, ax, THEME)
    t0 = min(w[0] for w in wins.values())
    t1 = max(w[1] for w in wins.values())
    s = spy[(spy.date >= t0.tz_localize(None)) & (spy.date <= t1.tz_localize(None))]
    spy_eq = s.close / s.close.iloc[0]
    spy_dd = (spy_eq / spy_eq.cummax() - 1) * 100
    ax.plot(s.date, spy_dd, color=COLORS["spy"], lw=1.6, ls="--", label=LABELS["spy"])
    ax.fill_between(s.date, spy_dd, 0, color=COLORS["spy"], alpha=0.12, lw=0)
    for key in ("momentum", "htf_swing", "swing"):
        eq = equity_series(trades[key], wins[key][0])
        dd = (eq / eq.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, color=COLORS[key], lw=1.8, label=LABELS[key])
        print(f"  {key} max drawdown on $100k-base equity: {dd.min():.2f}%")
    ax.set_ylabel("Drawdown from peak equity (%)")
    ax.set_title("Frozen-test drawdown (underwater) curves — same equity convention as fig01")
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncols=4)
    footer(fig,
           window=_win_str(wins),
           universe="; ".join(universes[k] for k in ("momentum", "htf_swing", "swing")),
           split="test(frozen)", benchmark="SPY",
           source="same as fig01",
           note="Drawdown measured on the $100k-base booked-P&L equity. The locked swing max_dd_pct (-74.5%) instead uses the sweep's "
                "cumulative per-trade %-sum convention — see README reconciliation.")
    return save_figure(fig, FIG_DIR / "fig02_drawdown.png", dpi=200, close=True)


def fig03_rolling_sharpe(trades, wins, universes, spy):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    style_figure(fig, ax, THEME)
    win_n, min_p = 63, 45
    for key in ("momentum", "htf_swing", "swing"):
        a, b = wins[key]
        days = spy[(spy.date >= a.tz_localize(None)) & (spy.date <= b.tz_localize(None))]["date"]
        pnl = daily_pnl(trades[key], pd.DatetimeIndex(days))
        ret = pnl / (BASE_K * 1000.0)
        rs = ret.rolling(win_n, min_periods=min_p).mean() / ret.rolling(win_n, min_periods=min_p).std() * np.sqrt(252)
        ax.plot(rs.index, rs.values, color=COLORS[key], lw=1.8, label=LABELS[key])
    t0 = min(w[0] for w in wins.values()).tz_localize(None)
    t1 = max(w[1] for w in wins.values()).tz_localize(None)
    s = spy[(spy.date >= t0) & (spy.date <= t1)]
    rs_spy = (s.set_index("date")["ret"].rolling(win_n, min_periods=min_p).mean()
              / s.set_index("date")["ret"].rolling(win_n, min_periods=min_p).std() * np.sqrt(252))
    ax.plot(rs_spy.index, rs_spy.values, color=COLORS["spy"], lw=1.6, ls="--", label=LABELS["spy"])
    ax.axhline(0, color=THEME.spine, lw=0.8)
    ax.set_ylabel(f"Rolling {win_n}-day annualized Sharpe")
    ax.set_title("Rolling Sharpe of daily booked P&L (vs $100k base) — frozen test windows")
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncols=4)
    footer(fig,
           window=_win_str(wins),
           universe="; ".join(universes[k] for k in ("momentum", "htf_swing", "swing")),
           split="test(frozen)", benchmark="SPY daily close-to-close",
           source="same as fig01",
           note="Daily return = booked trade P&L / $100k; zero on days with no exits. Multi-day trade P&L lands on its exit day.")
    return save_figure(fig, FIG_DIR / "fig03_rolling_sharpe.png", dpi=200, close=True)


def fig04_selection_bias(lock):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    style_figure(fig, (ax1, ax2), THEME)

    # (a) return-over-max-drawdown, biased test-selected winner vs deployed winner frozen on test
    # biased values from research/capstone/leakage_audit.md §0.6 (test-picked grid edge / family)
    biased = {"momentum": 44.6, "htf_swing": 41.4}
    clean = {k: lock[(m, "clean_deployed_winner_ret_over_dd")]["value"]
             for k, m in (("momentum", "momentum"), ("htf_swing", "htf_swing"))}
    x = np.arange(2)
    w = 0.38
    ax1.bar(x - w / 2, [biased[k] for k in ("momentum", "htf_swing")], w,
            color=COLORS["biased"], label="selected on test — test(comp)")
    ax1.bar(x + w / 2, [clean[k] for k in ("momentum", "htf_swing")], w,
            color=[COLORS["momentum"], COLORS["htf_swing"]], label="selected on val, frozen — test(frozen)")
    for i, k in enumerate(("momentum", "htf_swing")):
        ax1.text(i - w / 2, biased[k] + 0.8, f"{biased[k]:.1f}x", ha="center", fontsize=9, color=THEME.muted_text)
        ax1.text(i + w / 2, clean[k] + 0.8, f"{clean[k]:.1f}x", ha="center", fontsize=9,
                 fontweight="bold", color=THEME.text)
    ax1.set_xticks(x, [LABELS["momentum"], LABELS["htf_swing"]])
    ax1.set_ylabel("Backtest return / max drawdown (x)")
    ax1.set_title("(a) Order-policy backtest, ret/DD")
    ax1.legend(frameon=False, fontsize=8.5)

    # (b) swing win rates: stale self-selected artifact vs val-selected/test-frozen
    cats = ["Long", "Short", "Combined"]
    b_vals = [lock[("swing", "bt_v2_long_win_rate")]["value"] * 100,
              lock[("swing", "bt_v2_short_win_rate")]["value"] * 100,
              lock[("swing", "bt_v2_combined_wr_sector_agg")]["value"] * 100]
    c_vals = [lock[("swing", "bt_v2_clean_long_win_rate")]["value"] * 100,
              lock[("swing", "bt_v2_clean_short_win_rate")]["value"] * 100,
              lock[("swing", "bt_v2_clean_win_rate")]["value"] * 100]
    x = np.arange(3)
    ax2.bar(x - w / 2, b_vals, w, color=COLORS["biased"], label="stale artifact, self-selected split")
    ax2.bar(x + w / 2, c_vals, w, color=COLORS["swing"], label="val-selected, test-frozen")
    for i in range(3):
        ax2.text(i - w / 2, b_vals[i] + 0.7, f"{b_vals[i]:.1f}", ha="center", fontsize=9, color=THEME.muted_text)
        ax2.text(i + w / 2, c_vals[i] + 0.7, f"{c_vals[i]:.1f}", ha="center", fontsize=9,
                 fontweight="bold", color=THEME.text)
    ax2.axhline(50, color=THEME.spine, lw=1.0, ls=":")
    ax2.text(-0.55, 50.8, "coin flip", fontsize=8, color=THEME.muted_text, ha="left")
    ax2.set_xticks(x, cats)
    ax2.set_ylim(0, 78)
    ax2.set_ylabel("Trade win rate (%)")
    ax2.set_title("(b) Swing 30m sweep_v2 win rate")
    ax2.legend(frameon=False, fontsize=8.5)

    fig.suptitle("What the leakage audit corrected — selection on the reporting split inflated every headline backtest",
                 fontsize=11.5, y=1.02)
    footer(fig,
           window="momentum test 2025-05-20→2026-05, HTF test 2025-05-30→2026-06, swing test 2025-08→2026-06",
           universe="same trade sets as fig01",
           split="test(comp)/artifact → test(frozen)",
           source="results_lock.json; biased ret/DD from leakage_audit.md §0.6",
           note="Same model, same data — only the split the policy was SELECTED on changed. Win rates are PnL-based, long/short n=2,126/1,970 (clean).")
    return save_figure(fig, FIG_DIR / "fig04_selection_bias_correction.png", dpi=200, close=True)


def fig05_regime(trades, wins, universes, spy):
    reg = spy.set_index("date")["risk_on"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    style_figure(fig, (ax1, ax2), THEME)
    keys = ("momentum", "htf_swing", "swing")
    x = np.arange(len(keys))
    w = 0.38
    stats = {}
    for key in keys:
        df = trades[key].copy()
        on = df["entry_ts"].dt.normalize().dt.tz_localize(None).map(reg)
        for flag, name in ((True, "on"), (False, "off")):
            sub = df[on == flag]
            stats[(key, name)] = (len(sub), (sub.pnl_pct > 0).mean() * 100 if len(sub) else np.nan,
                                  sub.pnl_pct.mean() * 100 if len(sub) else np.nan)
    for ax, stat_i, ylab, title in ((ax1, 1, "Win rate (%)", "(a) Win rate by entry-day regime"),
                                    (ax2, 2, "Mean trade P&L (%)", "(b) Mean trade P&L by entry-day regime")):
        on_v = [stats[(k, "on")][stat_i] for k in keys]
        off_v = [stats[(k, "off")][stat_i] for k in keys]
        ax.bar(x - w / 2, on_v, w, color=[COLORS[k] for k in keys], label="SPY ≥ 200d SMA (risk-on)")
        ax.bar(x + w / 2, off_v, w, color="white", edgecolor=[COLORS[k] for k in keys],
               linewidth=1.6, hatch="//", label="SPY < 200d SMA (risk-off)")
        for i, k in enumerate(keys):
            if np.isfinite(on_v[i]):
                ax.text(i - w / 2, on_v[i] + (0.6 if stat_i == 1 else 0.06), f"{on_v[i]:.1f}",
                        ha="center", fontsize=8.5, color=THEME.text)
            if np.isfinite(off_v[i]):
                ax.text(i + w / 2, off_v[i] + (0.6 if stat_i == 1 else 0.06), f"{off_v[i]:.1f}",
                        ha="center", fontsize=8.5, color=THEME.text)
        ax.set_xticks(x, [f"{LABELS[k]}\n(n={stats[(k, 'on')][0]:,}/{stats[(k, 'off')][0]:,})" for k in keys])
        if stat_i == 1:
            ax.axhline(50, color=THEME.spine, lw=1.0, ls=":")
        else:
            ax.axhline(0, color=THEME.spine, lw=0.8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Frozen-test performance by market regime at entry (SPY close vs 200-day SMA)", fontsize=11.5, y=1.02)
    footer(fig,
           window=_win_str(wins),
           universe="; ".join(universes[k] for k in keys),
           split="test(frozen)", benchmark="regime defined on SPY 1d closes",
           source="same trades as fig01; Data/shared/bars/1d/SPY.parquet",
           note="Regime rule is fixed and stated (no fitting): risk-on = SPY close ≥ its 200-day SMA on the entry day. n=(risk-on/risk-off) under each strategy.")
    return save_figure(fig, FIG_DIR / "fig05_regime_performance.png", dpi=200, close=True)


def fig06_return_dists(trades, wins, universes):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=False)
    style_figure(fig, axes, THEME)
    for ax, key in zip(axes, ("momentum", "htf_swing", "swing")):
        r = trades[key].pnl_pct * 100
        lo_c, hi_c = np.percentile(r, [0.5, 99.5])
        ax.hist(r.clip(lo_c, hi_c), bins=60, color=COLORS[key], alpha=0.85)
        ax.axvline(0, color=THEME.spine, lw=0.9)
        ax.axvline(r.mean(), color=THEME.text, lw=1.3, ls="--")
        ax.axvline(r.median(), color=THEME.text, lw=1.3, ls=":")
        ax.set_title(f"{LABELS[key]} — n={len(r):,}\nmean {r.mean():+.2f}% (––)   median {r.median():+.2f}% (··)",
                     fontsize=9.5)
        ax.set_xlabel("Trade return (%)")
    axes[0].set_ylabel("Trades")
    fig.suptitle("Frozen-test trade-return distributions (display clipped at 0.5/99.5 pct)", fontsize=11.5, y=1.04)
    footer(fig,
           window=_win_str(wins),
           universe="; ".join(universes[k] for k in ("momentum", "htf_swing", "swing")),
           split="test(frozen)",
           source="same trades as fig01",
           note="Clipping affects the display only; means/medians are computed on unclipped returns.")
    return save_figure(fig, FIG_DIR / "fig06_trade_return_distributions.png", dpi=200, close=True)


def fig07_hold_times(trades, wins, universes):
    paper = pd.read_csv(ARTIFACTS["swing_paper_closed"])
    hold_min = (pd.to_datetime(paper.closed_ts) - pd.to_datetime(paper.entry_time)).dt.total_seconds() / 60.0
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    style_figure(fig, axes, THEME)
    conv = {"momentum": ("bars_held", BARS_PER_DAY_4H), "htf_swing": ("bars_held", BARS_PER_DAY_4H),
            "swing": ("holding_bars", BARS_PER_DAY_30M)}
    for ax, key in zip(axes[:3], ("momentum", "htf_swing", "swing")):
        col, per_day = conv[key]
        d = trades[key][col] / per_day
        ax.hist(d.clip(upper=np.percentile(d, 99.5)), bins=40, color=COLORS[key], alpha=0.85)
        ax.axvline(d.median(), color=THEME.text, lw=1.3, ls=":")
        ax.set_title(f"{LABELS[key]} — backtest\nmedian {d.median():.1f} trading days (n={len(d):,})", fontsize=9.5)
        ax.set_xlabel("Hold (trading days)")
    ax = axes[3]
    ax.hist(hold_min, bins=40, color=COLORS["swing"], alpha=0.55)
    ax.axvline(hold_min.median(), color=THEME.text, lw=1.3, ls=":")
    ax.set_title(f"Swing 30m — PAPER options ledger\nmedian {hold_min.median():.0f} min (n={len(paper)}, 2 sessions)",
                 fontsize=9.5)
    ax.set_xlabel("Hold (minutes)")
    axes[0].set_ylabel("Trades")
    fig.suptitle("Hold-time distributions — backtest (frozen test) and swing paper sessions", fontsize=11.5, y=1.04)
    footer(fig,
           window=_win_str(wins) + " / paper 2026-05-28→29",
           universe="; ".join(universes[k] for k in ("momentum", "htf_swing", "swing")) + "; paper: live swing scanner picks",
           split="panels a–c test(frozen); panel d paper",
           source="same trades as fig01; multiticker_20260528_20260529_closed_performance_rebuilt.csv",
           note="4H bars converted at 2 bars/trading day, 30m at 13. Panel d is the options paper ledger (small n, 2 sessions — not comparable to the stock backtests).")
    return save_figure(fig, FIG_DIR / "fig07_hold_times.png", dpi=200, close=True)


def _oof_frames():
    """(name, frame, window, color) for the five OOF-scored models."""
    cols = ["timestamp", "ticker", "score", "fwd_close_return"]
    mom = _read_oof(ARTIFACTS["mom_oof"])[cols].dropna()
    htf = _read_oof(ARTIFACTS["htf_oof"])[cols].dropna()
    q = _read_oof(ARTIFACTS["meta_q_oof"])[cols + ["y"]].dropna(subset=cols)
    u = _read_oof(ARTIFACTS["meta_u_oof"])[cols + ["y"]].dropna(subset=cols)
    # combo = per-timestamp rank-pct mean of the two meta scores (mirrors live s_combo
    # and reproduce_results.py's meta_combo construction)
    m = q.merge(u[["timestamp", "ticker", "score"]], on=["timestamp", "ticker"], suffixes=("_q", "_u"))
    rq = m.groupby("timestamp")["score_q"].rank(pct=True)
    ru = m.groupby("timestamp")["score_u"].rank(pct=True)
    combo = m.assign(score=(rq + ru) / 2.0)[["timestamp", "ticker", "score", "fwd_close_return"]]
    return [
        ("Momentum 4H", mom, "momentum"),
        ("HTF Swing 4H", htf, "htf_swing"),
        ("Meta quality", q, "meta"),
        ("Meta upside", u, "meta"),
        ("Meta combo (live s_combo)", combo, "meta"),
    ]


def fig08_oof_lift():
    frames = _oof_frames()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    style_figure(fig, axes, THEME)
    flat = axes.ravel()
    for ax, (name, df, ckey) in zip(flat, frames):
        pct = df.groupby("timestamp")["score"].rank(pct=True)
        decile = np.clip(np.ceil(pct * 10).astype(int), 1, 10)
        mean_by_d = df.assign(d=decile).groupby("d")["fwd_close_return"].mean() * 100
        uni = df["fwd_close_return"].mean() * 100
        colors = [COLORS[ckey] if d == 10 else COLORS[ckey] + "99" for d in mean_by_d.index]
        ax.bar(mean_by_d.index, mean_by_d.values, color=colors, width=0.72)
        ax.axhline(uni, color=THEME.neutral, lw=1.1, ls="--")
        ax.text(0.7, uni, f"universe {uni:+.2f}%", fontsize=7.5, color=THEME.muted_text, va="bottom")
        ax.text(10, mean_by_d.loc[10], f" {mean_by_d.loc[10]:+.2f}%", fontsize=8.5,
                fontweight="bold", color=THEME.text, ha="center", va="bottom")
        w0, w1 = df.timestamp.min(), df.timestamp.max()
        ax.set_title(f"{name}\n{w0:%Y-%m-%d} → {w1:%Y-%m-%d}  ({df.ticker.nunique():,} tickers)", fontsize=9.5)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Cross-sectional score decile (per bar)")
        ax.set_ylabel("Mean fwd close return (%)")
    flat[-1].axis("off")
    flat[-1].text(0.02, 0.92,
                  "Walk-forward out-of-fold scores\n(21-day embargo), scored on rows the\n"
                  "model never trained on.\n\n"
                  "fwd_close_return is the fixed label-\nhorizon forward return; windows overlap\n"
                  "across bars, so this is SIGNAL quality,\nnot a tradable equity curve\n"
                  "(those are figs 01–03).",
                  fontsize=9, color=THEME.muted_text, va="top")
    fig.suptitle("Ranking power out-of-sample: forward return by score decile (walk-forward OOF)", fontsize=12, y=1.0)
    footer(fig,
           window="per panel (walk-forward OOF span)",
           universe="full ranked pool per model (ticker counts per panel)",
           split="wf-oof",
           source="expansion_v1/oof_preds.parquet, swing_htf models/oof_preds.parquet, meta {quality,upside}/oof_preds.parquet",
           note="Decile 10 = top 10% by score within each bar. Universe line = pool mean over the same window.")
    return save_figure(fig, FIG_DIR / "fig08_oof_decile_lift.png", dpi=200, close=True)


def fig09_meta_calibration():
    frames = {n: f for n, f, _ in _oof_frames() if n in ("Meta quality", "Meta upside")}
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=(3.2, 1.0), hspace=0.08)
    for col, (name, df) in enumerate(frames.items()):
        ax = fig.add_subplot(gs[0, col])
        axc = fig.add_subplot(gs[1, col], sharex=ax)
        style_figure(fig, (ax, axc), THEME)
        bins = pd.qcut(df["score"], 10, duplicates="drop")
        g = df.groupby(bins, observed=True).agg(p=("score", "mean"), y=("y", "mean"), n=("y", "size"))
        lim = max(g.p.max(), g.y.max()) * 1.15
        ax.plot([0, lim], [0, lim], color=THEME.spine, lw=1.0, ls="--")
        ax.plot(g.p, g.y, color=COLORS["meta"], lw=2.0, marker="o", ms=5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_ylabel("Realized P(y=1) in bin")
        ax.set_title(f"{name} — walk-forward OOF reliability", fontsize=10)
        ax.tick_params(labelbottom=False)
        axc.bar(g.p, g.n / 1000.0, width=lim / 45, color=COLORS["biased"])
        axc.set_ylim(0, g.n.max() / 1000.0 * 1.35)
        axc.set_ylabel("n (k)\nequal bins", fontsize=7.5)
        axc.set_xlabel("Mean predicted probability in bin (10 quantile bins)")
    fig.suptitle("Meta ranker calibration on clean walk-forward OOF predictions", fontsize=12, y=0.99)
    q = _read_oof(ARTIFACTS["meta_q_oof"])
    footer(fig,
           window=f"{q.timestamp.min():%Y-%m-%d} → {q.timestamp.max():%Y-%m-%d}",
           universe=f"{q.ticker.nunique():,} tickers (meta feed pool)",
           split="wf-oof",
           source="signals/meta_context/meta_ranker/models/{quality,upside}/oof_preds.parquet",
           note="Dashed line = perfect calibration. Points below the diagonal over-predict. y = the model's own binary target (quality: meta_good; upside: top-tier fwd return).")
    return save_figure(fig, FIG_DIR / "fig09_meta_calibration.png", dpi=200, close=True)


def fig10_feature_importance():
    winners = [
        ("Swing 30m — xgb (live model)", "swing",
         REPO / "strategies/multi_ticker_swing/models/feature_importance.csv"),
        ("Momentum 4H — xgb_classifier s45", "momentum",
         REPO / "strategies/momentum_expansion/models/expansion_v1/feature_importance_xgb_classifier_seed45.csv"),
        ("HTF Swing 4H — lgbm_classifier s46", "htf_swing",
         REPO / "strategies/multi_ticker_swing_htf/models/feature_importance_lgbm_classifier_seed46.csv"),
        ("Meta quality — xgb_classifier s46", "meta",
         REPO / "signals/meta_context/meta_ranker/models/quality/feature_importance_xgb_classifier_seed46.csv"),
        ("Meta upside — xgb_classifier s48", "meta",
         REPO / "signals/meta_context/meta_ranker/models/upside/feature_importance_xgb_classifier_seed48.csv"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 5.6))
    style_figure(fig, axes, THEME)
    for ax, (name, ckey, path) in zip(axes, winners):
        fi = pd.read_csv(path)
        fi["share"] = fi["importance"] / fi["importance"].sum() * 100
        top = fi.nlargest(15, "share").iloc[::-1]
        ax.barh(top.feature, top.share, color=COLORS[ckey], alpha=0.9)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Importance share (%)")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Top-15 feature importance — the five deployed winner models (normalized per model)", fontsize=12, y=1.02)
    footer(fig,
           window="training artifacts (see run_ids in each model's meta.json)",
           universe="per-model training universe",
           split="artifact (train-time importance — not an OOS effect size)",
           source="feature_importance*.csv beside each locked model",
           note="Winner families/seeds match results_lock.json winner_family_seed. Importance = model gain share; treat as descriptive, not causal.")
    return save_figure(fig, FIG_DIR / "fig10_feature_importance.png", dpi=200, close=True)


def fig11_meta_exit_policy(lock):
    policies = [
        ("current_live_dropout_g0", "Rank drop-out\n(live default pre-2026-07-12)"),
        ("target20_full_exit", "Target +20%\nfull exit"),
        ("scaleout50_at20_horizon25", "Scale-out 50% @ +20%\n+ ride to horizon 25"),
    ]
    metrics = [("mean", "Mean return / trade (%)", 100), ("win", "Win rate (%)", 100),
               ("ret_per_bar", "Return per bar held (%)", 100)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    style_figure(fig, axes, THEME)
    for ax, (mkey, ylab, scale) in zip(axes, metrics):
        vals = [lock[("meta_exit_policy", f"{p}_{mkey}")]["value"] * scale for p, _ in policies]
        ns = [lock[("meta_exit_policy", f"{p}_{mkey}")]["n"] for p, _ in policies]
        ax.bar(range(3), vals, 0.62,
               color=[COLORS["biased"], COLORS["meta"] + "99", COLORS["meta"]])
        for i, (v, n) in enumerate(zip(vals, ns)):
            ax.text(i, v + max(vals) * 0.02, f"{v:.2f}", ha="center", fontsize=9.5, fontweight="bold",
                    color=THEME.text)
        ax.set_xticks(range(3), [f"{lbl}\nn={ns[i]:,}" for i, (_, lbl) in enumerate(policies)], fontsize=8)
        ax.axhline(0, color=THEME.spine, lw=0.8)
        if mkey == "win":
            ax.axhline(50, color=THEME.spine, lw=1.0, ls=":")
        ax.set_ylabel(ylab)
    fig.suptitle("Meta ranker: selection has edge, the old exit destroyed it — exit policies on the same entries "
                 "(OOF-scored holdout 2025-07-01+)", fontsize=11.5, y=1.02)
    footer(fig,
           window="holdout 2025-07-01 → 2026-05 (entries), 4H bars",
           universe="meta feed pool, top-10 s_combo entries per bar",
           split="wf-oof",
           source="results_lock.json meta_exit_policy rows (OOF s_combo rescore of backtest_exits, audit §4.3)",
           note="Trade counts differ because the exit rule changes which entries coexist. Stock-price paths — option premium paths (the live 7/9 losses) are strictly worse under slow exits.")
    return save_figure(fig, FIG_DIR / "fig11_meta_exit_policy.png", dpi=200, close=True)


def fig12_paper_sessions():
    paper = pd.read_csv(ARTIFACTS["swing_paper_closed"])
    paper["closed_ts"] = pd.to_datetime(paper.closed_ts)
    paper = paper.sort_values("closed_ts").reset_index(drop=True)
    fresh_calls = paper[(paper.is_fresh) & (paper.direction == 1)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    style_figure(fig, (ax1, ax2), THEME)

    r = paper.option_ret_pct
    ax1.hist(r, bins=40, color=COLORS["swing"], alpha=0.85)
    ax1.axvline(0, color=THEME.spine, lw=0.9)
    ax1.axvline(r.mean(), color=THEME.text, lw=1.3, ls="--")
    ax1.set_title(f"(a) Option trade returns — all closed (n={len(paper)}, win {100 * (r > 0).mean():.0f}%, "
                  f"mean {r.mean():+.0f}%)", fontsize=9.5)
    ax1.set_xlabel("Option return (%)")
    ax1.set_ylabel("Trades")

    cum = paper.option_pnl_dollars.cumsum()
    ax2.plot(paper.closed_ts, cum, color=COLORS["swing"], lw=1.8)
    ax2.axhline(0, color=THEME.spine, lw=0.9)
    import matplotlib.dates as mdates

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=7))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    ax2.set_title(f"(b) Cumulative option P&L by close time — total ${cum.iloc[-1]:,.0f}", fontsize=9.5)
    ax2.set_ylabel("Cumulative P&L ($)")
    ax2.annotate(f"fresh-entry calls: mean {fresh_calls.option_ret_pct.mean():+.1f}% (n={len(fresh_calls)})\n"
                 f"stale/other entries drove the losses",
                 xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8.5, color=THEME.muted_text)
    fig.suptitle("Swing 30m PAPER trading, options ledger, sessions 2026-05-28/29 — small n, reported unfiltered",
                 fontsize=11.5, y=1.02)
    footer(fig,
           window="2026-05-28 → 2026-05-29 (2 sessions)",
           universe="live swing scanner picks routed to short-dated options",
           split="paper",
           source="Data/analysis/multi_ticker_swing_live/experiments/multiticker_20260528_20260529_closed_performance_rebuilt.csv",
           note="Paper fills, 2 sessions, n=123 — anecdotal evidence only; option leverage makes returns incomparable to the stock backtests in figs 01–06.")
    return save_figure(fig, FIG_DIR / "fig12_paper_sessions.png", dpi=200, close=True)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated figure prefixes, e.g. fig01,fig08")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    apply_mpl_defaults(THEME)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lock = load_lock()

    need_trades = only is None or any(f in only for f in
                                      ("fig01", "fig02", "fig03", "fig05", "fig06", "fig07"))
    trades = wins = universes = spy = None
    if need_trades:
        print("loading frozen-test trade sets...")
        trades, wins, universes = strategies_bundle()
        spy = load_spy()

    made = []
    jobs = [
        ("fig01", lambda: fig01_equity(trades, wins, universes, spy)),
        ("fig02", lambda: fig02_drawdown(trades, wins, universes, spy)),
        ("fig03", lambda: fig03_rolling_sharpe(trades, wins, universes, spy)),
        ("fig04", lambda: fig04_selection_bias(lock)),
        ("fig05", lambda: fig05_regime(trades, wins, universes, spy)),
        ("fig06", lambda: fig06_return_dists(trades, wins, universes)),
        ("fig07", lambda: fig07_hold_times(trades, wins, universes)),
        ("fig08", fig08_oof_lift),
        ("fig09", fig09_meta_calibration),
        ("fig10", fig10_feature_importance),
        ("fig11", lambda: fig11_meta_exit_policy(lock)),
        ("fig12", fig12_paper_sessions),
    ]
    for name, fn in jobs:
        if only and name not in only:
            continue
        print(f"building {name}...")
        made.append(fn())
    print("\nwrote:")
    for p in made:
        print(f"  {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
