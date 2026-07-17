"""
Baseline comparison strategies for the 4H modules (capstone paper).

The paper's frozen-test module results (family_compare_clean) need reference
points beyond SPY buy-and-hold. This script computes, over EACH 4H module's
frozen-test window (momentum_expansion, multi_ticker_swing_htf):

  Portfolio-convention baselines (daily close-to-close, $100k base):
    1. spy_buy_hold           — SPY, recomputed here so every row shares one code path.
    2. equal_weight_universe  — the module's own ranked pool (test-window tickers after
                                the low-price gate), equal-weight, daily-rebalanced,
                                frictionless. "What if we just bought the whole pool?"
    3. sector_neutral_etf     — equal weight across the 11 SPDR sector ETFs, daily-
                                rebalanced. Proxy for sector-neutral market exposure:
                                the repo has NO per-ticker sector metadata
                                (shared_universe.csv sector column is Unknown/NaN),
                                so a stock-level sector-neutral sleeve is not buildable
                                without introducing new (survivorship-prone) data.
    4. tbill_3m               — 3-month T-bill accrual from FRED DGS3MO
                                (prev-trading-day yield / 252 per trading day).
    5. largest_stock_buy_hold — all-in buy-and-hold of the single largest universe
                                name (highest avg_dollar_volume_20d among
                                cap_tier == "mega" in shared_universe.csv).
    6. best_hindsight_pool_stock — ORACLE, not a strategy: all-in buy-and-hold of
                                whichever pool ticker actually had the highest
                                window-start-to-end return. Answers "what if I'd
                                picked the single best-performing name in advance"
                                as an explicit upper bound — computed with perfect
                                foresight (look-ahead by construction; label it as
                                such wherever cited in the paper).

  Trade-convention baseline (same execution engine as the modules):
    7. random_top_k           — random scores pushed through the IDENTICAL
                                family_backtest select/simulate path with the module's
                                deployed val-frozen exit policy, N seeds. Isolates the
                                ranking model's contribution from policy + universe.

  Plus each module's own frozen-test result restated on the daily portfolio
  convention (booked P&L at exit, $100k base) so Sharpe/DD columns are computed
  by ONE function for every row of the output table.

Known, documented limitations (also written into baseline_summary.json):
  - price returns only (bars are split-adjusted, dividend-excluded) — same
    convention as the module backtests and the fig01 SPY benchmark;
  - the universe pool is today's survivor universe (leakage_audit.md §universe):
    equal_weight_universe is therefore upward-biased — it is a HARD baseline;
  - daily rebalancing is frictionless (no costs/slippage), matching the
    cost-free module simulations;
  - modules and random_top_k hold ~$1k notional per open trade on a $100k base
    (never fully invested); portfolio baselines are 100% invested. Return
    LEVELS are comparable only per dollar of account equity — flag this when
    citing Sharpe across conventions.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/baseline_strategies.py            # both modules
  PYTHONPATH=. .venv/bin/python scripts/capstone/baseline_strategies.py --module momentum
  PYTHONPATH=. .venv/bin/python scripts/capstone/baseline_strategies.py --seeds 10 --skip-random

Outputs (research/capstone/baselines/):
  baseline_metrics.csv        one row per (window, strategy) — the paper table
  baseline_equity_curves.csv  long-format daily $100k equity per (window, strategy)
  random_k_seeds.csv          per-seed trade metrics for the random baseline
  baseline_summary.json       config, provenance, coverage stats, caveats
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.backtest import family_backtest as fb
from strategies.momentum_expansion.backtest.run_family_compare import STRATEGIES

REPO = Path(__file__).resolve().parents[2]
BARS_1D = REPO / "Data/shared/bars/1d"
TREASURY = REPO / "signals/meta_context/data/processed/fmp_treasury_rates.parquet"
UNIVERSE_CSV = REPO / "Data/shared/universe/shared_universe.csv"
OUT_DIR = REPO / "research/capstone/baselines"

BASE_CAPITAL = 100_000.0
TRADING_DAYS = 252

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

MODULES = {
    "momentum": dict(
        clean_summary=REPO / "strategies/momentum_expansion/backtest/results/family_compare_clean/comparison_summary_clean.json",
        trades_glob="*_frozen_test_trades.parquet",
    ),
    "htf": dict(
        clean_summary=REPO / "strategies/multi_ticker_swing_htf/backtest/results/family_compare_clean/comparison_summary_clean.json",
        trades_glob="*_frozen_test_trades.parquet",
    ),
}

_CLOSE_CACHE: dict[str, pd.Series | None] = {}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def load_close(ticker: str) -> pd.Series | None:
    """Daily close series indexed by tz-naive normalized date. None if no file."""
    if ticker not in _CLOSE_CACHE:
        path = BARS_1D / f"{ticker}.parquet"
        if not path.exists():
            _CLOSE_CACHE[ticker] = None
        else:
            df = pd.read_parquet(path, columns=["timestamp", "close"])
            idx = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
            s = pd.Series(df["close"].to_numpy(float), index=idx).sort_index()
            _CLOSE_CACHE[ticker] = s[~s.index.duplicated(keep="last")]
    return _CLOSE_CACHE[ticker]


# ---------------------------------------------------------------------------
# Module window + pool (identical filters to family_backtest_clean.load_window)
# ---------------------------------------------------------------------------

def module_context(module: str) -> dict:
    spec = STRATEGIES[module]
    summ = json.loads(MODULES[module]["clean_summary"].read_text())
    dep = summ["deployed_winner_frozen_test"]
    trades_path = (MODULES[module]["clean_summary"].parent
                   / f"{dep['family']}_s{dep['seed']}_frozen_test_trades.parquet")
    trades = pd.read_parquet(trades_path).sort_values("exit_ts").reset_index(drop=True)
    if len(trades) != int(dep["trades"]):
        raise RuntimeError(f"{module}: trades parquet ({len(trades)}) != locked count ({dep['trades']})")

    test_start = pd.Timestamp(summ["test_window_start"])
    win_start = test_start.tz_localize(None).normalize()
    win_end = pd.Timestamp(trades["exit_ts"].max()).tz_localize(None).normalize()

    # Pool: distinct test-window tickers after the same low-price gate the
    # module backtest applies (low_price_flag is a matrix column in both modules).
    pool_df = fb._read_reset(
        Path(spec["matrix_path"]), ["timestamp", "ticker", "low_price_flag"],
        [("timestamp", ">=", test_start.to_pydatetime())],
    )
    pool_df["timestamp"] = pd.to_datetime(pool_df["timestamp"], utc=True)
    pool_df["ticker"] = pool_df["ticker"].astype(str)
    keep = pd.to_numeric(pool_df["low_price_flag"], errors="coerce").fillna(1.0) <= 0.0
    pool_df = pool_df.loc[keep, ["timestamp", "ticker"]].reset_index(drop=True)

    return dict(
        name=spec["name"], spec=spec, summary=summ, deployed=dep, trades=trades,
        window=(win_start, win_end), pool_df=pool_df,
        pool=sorted(pool_df["ticker"].unique()),
    )


# ---------------------------------------------------------------------------
# Daily-return construction
# ---------------------------------------------------------------------------

def trading_days_index(win: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DatetimeIndex:
    spy = load_close("SPY")
    return spy.loc[(spy.index >= win[0]) & (spy.index <= win[1])].index


def single_asset_returns(ticker: str, days: pd.DatetimeIndex) -> pd.Series:
    s = load_close(ticker)
    if s is None:
        raise FileNotFoundError(f"no daily bars for {ticker}")
    return s.reindex(days).pct_change(fill_method=None).fillna(0.0)


def equal_weight_returns(tickers: list[str], days: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """Daily-rebalanced equal weight: mean of available daily returns each day."""
    rets, missing = [], []
    for t in tickers:
        s = load_close(t)
        if s is None:
            missing.append(t)
            continue
        r = s.reindex(days).pct_change(fill_method=None)
        if r.notna().sum() >= 2:
            rets.append(r.rename(t))
    mat = pd.concat(rets, axis=1)
    port = mat.mean(axis=1, skipna=True).fillna(0.0)
    stats = dict(
        n_requested=len(tickers), n_used=mat.shape[1], n_missing_bars=len(missing),
        avg_names_per_day=round(float(mat.notna().sum(axis=1).mean()), 1),
        extreme_daily_moves_gt50pct=int((mat.abs() > 0.5).sum().sum()),
    )
    return port, stats


def tbill_returns(days: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """Accrue the 3M T-bill: previous trading day's DGS3MO / 252 per day."""
    t = pd.read_parquet(TREASURY, columns=["date", "month3"])
    y = pd.Series(t["month3"].to_numpy(float), index=pd.to_datetime(t["date"])).sort_index()
    y = y.reindex(days.union(y.index)).ffill().reindex(days)
    daily = (y.shift(1).bfill() / 100.0 / TRADING_DAYS).fillna(0.0)
    stale_days = int((days > y.dropna().index.max()).sum()) if y.notna().any() else len(days)
    return daily, dict(yield_start=float(y.iloc[0]), yield_end=float(y.dropna().iloc[-1]),
                       days_beyond_last_fred_obs=stale_days)


def pick_largest_stock() -> str:
    u = pd.read_csv(UNIVERSE_CSV)
    mega = u[(u["cap_tier"] == "mega") & (u["type"] == "Stock")]
    return str(mega.loc[mega["avg_dollar_volume_20d"].idxmax(), "ticker"])


def pick_best_hindsight_stock(pool: list[str], days: pd.DatetimeIndex) -> tuple[str | None, float]:
    """Oracle baseline: whichever pool ticker had the highest window-start-to-end
    return. NOT a tradeable strategy (requires perfect foresight) — reported as
    an upper bound so 'what if I'd just picked the winner' has a real number."""
    best_ticker, best_ret = None, -np.inf
    for t in pool:
        s = load_close(t)
        if s is None:
            continue
        sub = s.reindex(days).dropna()
        if len(sub) < 2:
            continue
        ret = float(sub.iloc[-1] / sub.iloc[0] - 1.0)
        if ret > best_ret:
            best_ticker, best_ret = t, ret
    return best_ticker, best_ret


def module_daily_returns(trades: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """Module frozen-test result on the portfolio convention: $1k-notional P&L
    booked at exit date on a $100k base (same as make_figures fig03)."""
    booked = trades.groupby(pd.to_datetime(trades["exit_ts"]).dt.tz_localize(None).dt.normalize())["pnl_dollar"].sum()
    booked = booked.reindex(days, fill_value=0.0)
    eq = BASE_CAPITAL + booked.cumsum()
    return eq.pct_change().fillna(booked.iloc[0] / BASE_CAPITAL if len(booked) else 0.0)


# ---------------------------------------------------------------------------
# Metrics (one code path for every table row)
# ---------------------------------------------------------------------------

def daily_metrics(ret: pd.Series) -> dict:
    eq = BASE_CAPITAL * (1.0 + ret).cumprod()
    total = float(eq.iloc[-1] / BASE_CAPITAL - 1.0)
    n = len(ret)
    years = n / TRADING_DAYS
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    vol = float(ret.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(ret.mean() / ret.std() * np.sqrt(TRADING_DAYS)) if ret.std() > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "n_days": n,
        "total_return_pct": round(total * 100, 3),
        "cagr_pct": round(cagr * 100, 3),
        "ann_vol_pct": round(vol * 100, 3),
        "sharpe_ann": round(sharpe, 3),
        "max_dd_pct": round(dd * 100, 3),
        "ret_over_dd": round(total / abs(dd), 3) if dd < 0 else float("inf"),
    }


def equity_frame(ret: pd.Series, window: str, strategy: str) -> pd.DataFrame:
    eq = BASE_CAPITAL * (1.0 + ret).cumprod()
    return pd.DataFrame({"window": window, "strategy": strategy, "date": eq.index, "equity": eq.values})


# ---------------------------------------------------------------------------
# Random top-k baseline through the module's own engine
# ---------------------------------------------------------------------------

def random_topk(ctx: dict, days: pd.DatetimeIndex, n_seeds: int) -> tuple[list[dict], list[pd.Series]]:
    dep = ctx["deployed"]
    pool_df = ctx["pool_df"].copy()
    allow_short = ctx["spec"]["allow_short"]
    bars = fb.BarCache()
    seed_rows, seed_rets = [], []
    for i in range(n_seeds):
        seed = 123 + i
        rng = np.random.default_rng(seed)
        pool_df["score"] = rng.random(len(pool_df))
        sig = fb.select_signals(pool_df, int(dep["top_k"]), allow_short)
        trades = fb.simulate(sig, bars, tp_mult=float(dep["tp_atr_mult"]),
                             sl_mult=float(dep["sl_atr_mult"]), max_hold=int(dep["max_hold"]))
        m = fb.metrics(trades)
        m.update(module=ctx["name"], seed=seed)
        seed_rows.append(m)
        seed_rets.append(module_daily_returns(trades, days))
    return seed_rows, seed_rets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(modules: list[str], n_seeds: int, skip_random: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    largest = pick_largest_stock()
    print(f"largest-stock baseline pick: {largest}")

    metric_rows, curves, seed_tables, summary_windows = [], [], [], {}
    for module in modules:
        ctx = module_context(module)
        win = ctx["window"]
        days = trading_days_index(win)
        wname = ctx["name"]
        print(f"[{wname}] window {win[0]:%Y-%m-%d} -> {win[1]:%Y-%m-%d} ({len(days)} trading days), "
              f"pool={len(ctx['pool'])} tickers")

        ew_ret, ew_stats = equal_weight_returns(ctx["pool"], days)
        sect_ret, _ = equal_weight_returns(SECTOR_ETFS, days)
        tb_ret, tb_stats = tbill_returns(days)
        best_ticker, best_ret_pct = pick_best_hindsight_stock(ctx["pool"], days)
        print(f"[{wname}] best-hindsight pool ticker: {best_ticker} ({best_ret_pct * 100:+.1f}%)")
        rows = {
            "spy_buy_hold": single_asset_returns("SPY", days),
            "equal_weight_universe": ew_ret,
            "sector_neutral_etf": sect_ret,
            "tbill_3m": tb_ret,
            f"largest_stock_{largest}": single_asset_returns(largest, days),
            f"module_{module}_deployed": module_daily_returns(ctx["trades"], days),
        }
        if best_ticker is not None:
            rows[f"best_hindsight_pool_stock_{best_ticker}"] = single_asset_returns(best_ticker, days)
        conventions = {k: ("trade_booked_daily" if k.startswith("module_") else "portfolio_daily")
                       for k in rows}

        if not skip_random:
            seed_rows, seed_rets = random_topk(ctx, days, n_seeds)
            seed_tables.extend(seed_rows)
            per_seed = [daily_metrics(r) for r in seed_rets]
            agg = {k: (round(float(np.mean([m[k] for m in per_seed])), 3) if k != "n_days"
                       else per_seed[0][k]) for k in per_seed[0]}
            agg_std = {f"{k}_std": round(float(np.std([m[k] for m in per_seed])), 3)
                       for k in per_seed[0] if k != "n_days"}
            metric_rows.append({"window": wname, "strategy": f"random_top{int(ctx['deployed']['top_k'])}",
                                "convention": "trade_booked_daily", **agg, **agg_std,
                                "seeds": len(per_seed)})
            med = int(np.argsort([m["total_return_pct"] for m in per_seed])[len(per_seed) // 2])
            curves.append(equity_frame(seed_rets[med], wname,
                                       f"random_top{int(ctx['deployed']['top_k'])}_median_seed"))

        for name, ret in rows.items():
            metric_rows.append({"window": wname, "strategy": name,
                                "convention": conventions[name], **daily_metrics(ret)})
            curves.append(equity_frame(ret, wname, name))

        summary_windows[wname] = dict(
            window=[str(win[0].date()), str(win[1].date())], trading_days=len(days),
            pool_tickers=len(ctx["pool"]), equal_weight_coverage=ew_stats, tbill=tb_stats,
            deployed_policy={k: ctx["deployed"][k] for k in
                             ("family", "seed", "top_k", "tp_atr_mult", "sl_atr_mult", "max_hold")},
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUT_DIR / "baseline_metrics.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT_DIR / "baseline_equity_curves.csv", index=False)
    if seed_tables:
        pd.DataFrame(seed_tables).to_csv(OUT_DIR / "random_k_seeds.csv", index=False)

    summary = dict(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), git_sha=_git_sha(),
        base_capital=BASE_CAPITAL, largest_stock_pick=largest,
        largest_stock_rule="max avg_dollar_volume_20d among cap_tier=='mega' stocks in shared_universe.csv",
        sector_etfs=SECTOR_ETFS, random_seeds=list(range(123, 123 + n_seeds)) if not skip_random else [],
        windows=summary_windows,
        caveats=[
            "Price returns only: shared 1d bars are split-adjusted but dividend-excluded "
            "(same convention as the module backtests and fig01's SPY benchmark). "
            "tbill_3m and sector/SPY ETF rows therefore understate total return by their yields.",
            "equal_weight_universe uses today's survivor universe (leakage_audit.md): "
            "delisted names are absent, so this baseline is upward-biased (a hard bar).",
            "Daily rebalancing is frictionless — no costs/slippage — matching the cost-free "
            "module simulations.",
            "Convention flag: 'portfolio_daily' rows are 100% invested; 'trade_booked_daily' "
            "rows (modules, random_top_k) book $1k-notional trade P&L on a $100k base and are "
            "never fully invested. Compare Sharpe/DD across conventions with care.",
            "sector_neutral_etf uses the 11 SPDR sector ETFs because the repo has no usable "
            "per-ticker sector metadata (shared_universe.csv sector column is Unknown/NaN).",
            "best_hindsight_pool_stock is an ORACLE upper bound, not a tradeable strategy: "
            "it is chosen by its own realized return over the window (perfect foresight, "
            "look-ahead by construction). Report it as 'best case if you'd picked right', "
            "never as a baseline a model could have matched.",
        ],
        artifacts={m: str(MODULES[m]["clean_summary"].relative_to(REPO)) for m in modules},
    )
    (OUT_DIR / "baseline_summary.json").write_text(json.dumps(summary, indent=2))

    cols = ["window", "strategy", "convention", "total_return_pct", "cagr_pct",
            "ann_vol_pct", "sharpe_ann", "max_dd_pct", "ret_over_dd"]
    print("\n" + metrics_df[cols].to_string(index=False))
    print(f"\nwrote {OUT_DIR.relative_to(REPO)}/(baseline_metrics.csv, baseline_equity_curves.csv, "
          f"random_k_seeds.csv, baseline_summary.json)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--module", choices=[*MODULES, "all"], default="all")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--skip-random", action="store_true")
    args = p.parse_args()
    run(list(MODULES) if args.module == "all" else [args.module], args.seeds, args.skip_random)
