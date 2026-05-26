"""Phase 2 descriptive stats on multi-ticker swing trades dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_pct(v):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "    n/a"
    return f"{100*v:6.2f}%"


def _basic_stats(df: pd.DataFrame, pnl_col: str) -> dict:
    s = df[pnl_col].dropna()
    if len(s) == 0:
        return {"n": 0, "win_rate": np.nan, "mean": np.nan, "median": np.nan, "sum": np.nan}
    return {
        "n": int(len(s)),
        "win_rate": float((s > 0).mean()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "sum": float(s.sum()),
        "std": float(s.std()),
    }


def _group_stats(df: pd.DataFrame, by, pnl_col: str, min_n: int = 5) -> pd.DataFrame:
    if isinstance(by, str):
        by = [by]
    rows = []
    for key, sub in df.groupby(by, dropna=False):
        st = _basic_stats(sub, pnl_col)
        if st["n"] < min_n:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({**dict(zip(by, key)), **st})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("mean", ascending=False)
    return out


def _print_section(title: str, lines):
    print(f"\n## {title}\n")
    for ln in lines:
        print(ln)


def report(df: pd.DataFrame, out_md: Path):
    closed = df[df["exit_price"].notna()].copy()
    print(f"\nTotal fresh-open trades: {len(df)}")
    print(f"Closed (full lifecycle): {len(closed)}")

    md_lines = []

    def emit(s: str):
        print(s)
        md_lines.append(s)

    emit("# Phase 2 — Descriptive Stats: Multi-Ticker Swing (5/14 → 5/22)\n")
    emit(f"- Dataset: {len(df)} fresh-open trades, {len(closed)} with a matched close")
    emit(f"- Window: {closed['entry_date_et'].min()} → {closed['entry_date_et'].max()}")
    emit("- All percentages: underlying-PnL unless noted (option PnL coverage is too low — only 23 trades have both ends)\n")

    # Overall
    o = _basic_stats(closed, "underlying_pnl_pct")
    emit("## Overall (underlying PnL)\n")
    emit(f"- n = {o['n']}")
    emit(f"- win rate = {_fmt_pct(o['win_rate'])}")
    emit(f"- mean PnL = {_fmt_pct(o['mean'])}, median = {_fmt_pct(o['median'])}")
    emit(f"- 25%–75% range: {_fmt_pct(o['p25'])} to {_fmt_pct(o['p75'])}")
    emit(f"- aggregate sum PnL: {_fmt_pct(o['sum'])}")
    emit(f"- std: {_fmt_pct(o['std'])}")
    expectancy = o["mean"]
    emit(f"- expectancy per trade: {_fmt_pct(expectancy)}\n")

    # Option PnL where we have it
    opt_closed = closed[closed["option_pnl_pct"].notna()]
    if len(opt_closed) > 0:
        oo = _basic_stats(opt_closed, "option_pnl_pct")
        emit("### Option PnL subset (only trades with both option_entry & option_last)\n")
        emit(f"- n = {oo['n']}")
        emit(f"- option win rate = {_fmt_pct(oo['win_rate'])}")
        emit(f"- mean option PnL = {_fmt_pct(oo['mean'])}, median = {_fmt_pct(oo['median'])}")
        emit(f"- aggregate option PnL: {_fmt_pct(oo['sum'])}\n")

    # By direction
    g = _group_stats(closed, "direction", "underlying_pnl_pct", min_n=10)
    emit("## By direction (1 = long, -1 = short)\n")
    emit(g.to_string(index=False) if not g.empty else "(no group meets min n)")
    emit("")

    # By tier
    g = _group_stats(closed, "tier", "underlying_pnl_pct", min_n=10)
    emit("\n## By tier\n")
    emit(g.to_string(index=False) if not g.empty else "(no group meets min n)")
    emit("")

    # By tier × direction
    g = _group_stats(closed, ["tier", "direction"], "underlying_pnl_pct", min_n=10)
    emit("\n## By tier × direction\n")
    emit(g.to_string(index=False) if not g.empty else "(no group meets min n)")
    emit("")

    # By exit reason
    g = _group_stats(closed, "exit_reason", "underlying_pnl_pct", min_n=5)
    emit("\n## By exit_reason\n")
    emit(g.to_string(index=False) if not g.empty else "(no group meets min n)")
    emit("")

    # By time-of-day bucket
    def tod_bucket(m):
        if pd.isna(m):
            return "unknown"
        m = int(m)
        if m < 10 * 60:
            return "pre-10:00"
        if m < 11 * 60:
            return "10:00-11:00"
        if m < 13 * 60:
            return "11:00-13:00"
        if m < 14 * 60 + 30:
            return "13:00-14:30"
        if m < 16 * 60:
            return "14:30-16:00"
        return "post-16:00"
    closed = closed.copy()
    closed["tod_bucket"] = closed["entry_minute_of_day"].apply(tod_bucket)
    g = _group_stats(closed, "tod_bucket", "underlying_pnl_pct", min_n=10)
    emit("\n## By time-of-day bucket\n")
    emit(g.to_string(index=False) if not g.empty else "(no group meets min n)")
    emit("")

    # By day of week
    g = _group_stats(closed, "day_of_week", "underlying_pnl_pct", min_n=10)
    emit("\n## By day_of_week (0=Mon)\n")
    emit(g.to_string(index=False) if not g.empty else "(no group meets min n)")
    emit("")

    # By entry date
    g = _group_stats(closed, "entry_date_et", "underlying_pnl_pct", min_n=5)
    emit("\n## By entry date\n")
    emit(g.to_string(index=False) if not g.empty else "(no group meets min n)")
    emit("")

    # Top/bottom tickers
    g = _group_stats(closed, "ticker", "underlying_pnl_pct", min_n=3)
    emit("\n## Top 15 tickers by mean underlying PnL (min n=3)\n")
    emit(g.head(15).to_string(index=False) if not g.empty else "(no group meets min n)")
    emit("\n## Bottom 15 tickers by mean underlying PnL (min n=3)\n")
    emit(g.tail(15).to_string(index=False) if not g.empty else "(no group meets min n)")
    emit("")

    # p_dir deciles
    if "p_dir" in closed.columns and closed["p_dir"].notna().sum() > 50:
        c2 = closed.dropna(subset=["p_dir"]).copy()
        c2["p_dir_bin"] = pd.qcut(c2["p_dir"], 5, labels=False, duplicates="drop")
        g = _group_stats(c2, "p_dir_bin", "underlying_pnl_pct", min_n=10)
        emit("\n## By p_dir quintile (0=lowest)\n")
        emit(g.to_string(index=False) if not g.empty else "(insufficient)")
        emit("")

    # ev_score deciles
    if "ev_score" in closed.columns and closed["ev_score"].notna().sum() > 50:
        c2 = closed.dropna(subset=["ev_score"]).copy()
        c2["ev_score_bin"] = pd.qcut(c2["ev_score"], 5, labels=False, duplicates="drop")
        g = _group_stats(c2, "ev_score_bin", "underlying_pnl_pct", min_n=10)
        emit("\n## By ev_score quintile (0=lowest)\n")
        emit(g.to_string(index=False) if not g.empty else "(insufficient)")
        emit("")

    # ATR% deciles
    if "atr_pct_of_entry" in closed.columns and closed["atr_pct_of_entry"].notna().sum() > 50:
        c2 = closed.dropna(subset=["atr_pct_of_entry"]).copy()
        c2["atrpct_bin"] = pd.qcut(c2["atr_pct_of_entry"], 5, labels=False, duplicates="drop")
        g = _group_stats(c2, "atrpct_bin", "underlying_pnl_pct", min_n=10)
        emit("\n## By ATR% (atr_at_entry / entry_price) quintile (0=lowest vol)\n")
        emit(g.to_string(index=False) if not g.empty else "(insufficient)")
        emit("")

    # Confirmation bars
    if "confirm_bars_watched" in closed.columns and closed["confirm_bars_watched"].notna().sum() > 30:
        c2 = closed.dropna(subset=["confirm_bars_watched"]).copy()
        g = _group_stats(c2, "confirm_bars_watched", "underlying_pnl_pct", min_n=10)
        emit("\n## By confirm_bars_watched (5m bars to confirm)\n")
        emit(g.to_string(index=False) if not g.empty else "(insufficient)")
        emit("")

    # Bars held
    if "bars_held" in closed.columns and closed["bars_held"].notna().sum() > 30:
        c2 = closed.dropna(subset=["bars_held"]).copy()
        c2["bh_bin"] = pd.cut(
            c2["bars_held"],
            bins=[-1, 1, 3, 6, 12, 24, 48, 100, 1000],
            labels=["0-1", "2-3", "4-6", "7-12", "13-24", "25-48", "49-100", ">100"],
        )
        g = _group_stats(c2, "bh_bin", "underlying_pnl_pct", min_n=10)
        emit("\n## By bars_held bucket\n")
        emit(g.to_string(index=False) if not g.empty else "(insufficient)")
        emit("")

    # Option DTE
    if "option_dte" in closed.columns and closed["option_dte"].notna().sum() > 30:
        c2 = closed.dropna(subset=["option_dte"]).copy()
        c2["dte_bin"] = pd.cut(
            c2["option_dte"],
            bins=[-1, 0, 1, 2, 4, 7, 14, 30, 1000],
            labels=["0DTE", "1DTE", "2DTE", "3-4DTE", "5-7DTE", "8-14DTE", "15-30DTE", ">30DTE"],
        )
        g = _group_stats(c2, "dte_bin", "underlying_pnl_pct", min_n=5)
        emit("\n## By option DTE bucket\n")
        emit(g.to_string(index=False) if not g.empty else "(insufficient)")
        emit("")

    # Match type (sanity)
    g = _group_stats(closed, "match_type", "underlying_pnl_pct", min_n=5)
    emit("\n## By match_type (sanity check: same-session vs cross-session matches)\n")
    emit(g.to_string(index=False) if not g.empty else "(insufficient)")
    emit("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWrote {out_md}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", default="local_artifacts/swing_analysis_20260525/trades.parquet")
    p.add_argument("--out", default="local_artifacts/swing_analysis_20260525/phase2_descriptive.md")
    args = p.parse_args()
    df = pd.read_parquet(args.trades)
    report(df, Path(args.out))


if __name__ == "__main__":
    main()
