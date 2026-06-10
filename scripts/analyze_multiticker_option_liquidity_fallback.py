#!/usr/bin/env python3
"""Option liquidity and stock-fallback experiments for swing audit trades.

This intentionally uses only causal CBOE snapshots: for each trade, the latest
snapshot_date on or before the trade date is joined by ticker. The local CBOE
summary is ticker/side aggregate liquidity, not selected-contract liquidity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TRADE_PATH = Path("UI/swing_audit/live_policy_experiments_20260608/audit_fresh_option_trades.csv")
CBOE_SUMMARY_PATH = Path("news/data/processed/cboe_options_summary.parquet")
OUT_DIR = Path("UI/swing_audit/live_policy_experiments_20260608")

OI_THRESHOLDS = (10, 100, 1000)
VOLUME_THRESHOLDS = (10, 100, 1000)
SPREAD_CAP_PCT = 18.0
STOCK_SHARES = 100


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    threshold_oi: float | None
    threshold_volume: float | None
    total_trades: int
    option_trades: int
    stock_fallback_trades: int
    unknown_liquidity_trades: int
    avg_cboe_snapshot_age_days: float
    option_pnl: float
    fallback_stock_pnl: float
    total_pnl_with_fallback: float
    baseline_option_pnl: float
    delta_vs_baseline: float
    option_win_rate: float
    stock_fallback_win_rate: float
    combined_win_rate: float


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.date


def _join_causal_cboe_summary(trades: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    summary = summary.copy()
    trades["ticker"] = trades["ticker"].astype(str).str.upper()
    summary["ticker"] = summary["ticker"].astype(str).str.upper()
    trades["entry_date"] = _to_date(trades["entry_time_et"])
    trades["entry_date_ts"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    summary["snapshot_date_only"] = pd.to_datetime(summary["snapshot_date"], errors="coerce", utc=True).dt.date
    summary["snapshot_date_ts"] = pd.to_datetime(summary["snapshot_date_only"], errors="coerce")

    joined_parts: list[pd.DataFrame] = []
    summary_cols = [
        "ticker",
        "snapshot_date_only",
        "snapshot_date_ts",
        "call_volume",
        "put_volume",
        "call_open_interest",
        "put_open_interest",
        "call_premium",
        "put_premium",
        "stock_volume",
    ]
    for ticker, tdf in trades.groupby("ticker", sort=False):
        sdf = summary.loc[summary["ticker"] == ticker, summary_cols].sort_values("snapshot_date_only")
        if sdf.empty:
            part = tdf.copy()
            for col in summary_cols:
                if col != "ticker":
                    part[f"cboe_{col}"] = np.nan
            joined_parts.append(part)
            continue

        left = tdf.sort_values("entry_date").copy()
        right = sdf.rename(columns={c: f"cboe_{c}" for c in summary_cols if c != "ticker"})
        merged = pd.merge_asof(
            left,
            right,
            left_on="entry_date_ts",
            right_on="cboe_snapshot_date_ts",
            by="ticker",
            direction="backward",
        )
        joined_parts.append(merged)

    out = pd.concat(joined_parts, ignore_index=True)
    out["cboe_snapshot_age_days"] = (
        pd.to_datetime(out["entry_date"], errors="coerce")
        - pd.to_datetime(out["cboe_snapshot_date_only"], errors="coerce")
    ).dt.days
    is_call = out["side"].astype(str).str.lower().eq("call")
    out["side_cboe_volume"] = np.where(is_call, out["cboe_call_volume"], out["cboe_put_volume"])
    out["side_cboe_open_interest"] = np.where(
        is_call,
        out["cboe_call_open_interest"],
        out["cboe_put_open_interest"],
    )
    out["side_cboe_premium"] = np.where(is_call, out["cboe_call_premium"], out["cboe_put_premium"])
    return out


def _prepare_trades() -> pd.DataFrame:
    trades = pd.read_csv(TRADE_PATH)
    summary = pd.read_parquet(CBOE_SUMMARY_PATH)
    df = _join_causal_cboe_summary(trades, summary)

    for col in [
        "entry_underlying",
        "exit_underlying_or_mark",
        "direction",
        "option_pnl_dollars",
        "entry_spread_pct_mid",
        "side_cboe_volume",
        "side_cboe_open_interest",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["stock_100sh_pnl"] = (
        (df["exit_underlying_or_mark"] - df["entry_underlying"]) * df["direction"] * STOCK_SHARES
    )
    df["spread_lt_18"] = df["entry_spread_pct_mid"] < SPREAD_CAP_PCT
    df["has_cboe_liquidity"] = df["side_cboe_volume"].notna() & df["side_cboe_open_interest"].notna()
    return df


def _win_rate(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float((values > 0).mean()) if len(values) else float("nan")


def _summarize(df: pd.DataFrame, option_mask: pd.Series, name: str, oi: float | None, vol: float | None) -> PolicyResult:
    option_mask = option_mask.fillna(False)
    stock_mask = ~option_mask
    option_pnl = float(df.loc[option_mask, "option_pnl_dollars"].fillna(0).sum())
    stock_pnl = float(df.loc[stock_mask, "stock_100sh_pnl"].fillna(0).sum())
    combined = pd.concat(
        [
            df.loc[option_mask, "option_pnl_dollars"],
            df.loc[stock_mask, "stock_100sh_pnl"],
        ],
        ignore_index=True,
    )
    baseline = float(df["option_pnl_dollars"].fillna(0).sum())
    ages = pd.to_numeric(df.loc[df["has_cboe_liquidity"], "cboe_snapshot_age_days"], errors="coerce")
    return PolicyResult(
        policy=name,
        threshold_oi=oi,
        threshold_volume=vol,
        total_trades=int(len(df)),
        option_trades=int(option_mask.sum()),
        stock_fallback_trades=int(stock_mask.sum()),
        unknown_liquidity_trades=int((~df["has_cboe_liquidity"]).sum()),
        avg_cboe_snapshot_age_days=float(ages.mean()) if len(ages) else float("nan"),
        option_pnl=option_pnl,
        fallback_stock_pnl=stock_pnl,
        total_pnl_with_fallback=float(option_pnl + stock_pnl),
        baseline_option_pnl=baseline,
        delta_vs_baseline=float(option_pnl + stock_pnl - baseline),
        option_win_rate=_win_rate(df.loc[option_mask, "option_pnl_dollars"]),
        stock_fallback_win_rate=_win_rate(df.loc[stock_mask, "stock_100sh_pnl"]),
        combined_win_rate=_win_rate(combined),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _prepare_trades()
    df.to_csv(OUT_DIR / "option_liquidity_joined_trades.csv", index=False)

    rows: list[PolicyResult] = []
    rows.append(_summarize(df, pd.Series(True, index=df.index), "all_options_baseline", None, None))
    rows.append(_summarize(df, df["spread_lt_18"], "spread_lt_18_else_stock_100sh", None, None))

    for oi in OI_THRESHOLDS:
        liq = df["has_cboe_liquidity"] & (df["side_cboe_open_interest"] >= oi)
        rows.append(_summarize(df, liq, f"side_oi_ge_{oi}_else_stock_100sh", oi, None))
    for vol in VOLUME_THRESHOLDS:
        liq = df["has_cboe_liquidity"] & (df["side_cboe_volume"] >= vol)
        rows.append(_summarize(df, liq, f"side_volume_ge_{vol}_else_stock_100sh", None, vol))
    for oi in OI_THRESHOLDS:
        for vol in VOLUME_THRESHOLDS:
            liq = (
                df["has_cboe_liquidity"]
                & (df["side_cboe_open_interest"] >= oi)
                & (df["side_cboe_volume"] >= vol)
            )
            rows.append(_summarize(df, liq, f"side_oi_ge_{oi}_vol_ge_{vol}_else_stock_100sh", oi, vol))

            spread_or_liq = df["spread_lt_18"] | liq
            rows.append(
                _summarize(
                    df,
                    spread_or_liq,
                    f"spread_lt_18_or_side_oi_ge_{oi}_vol_ge_{vol}_else_stock_100sh",
                    oi,
                    vol,
                )
            )

            spread_and_liq = df["spread_lt_18"] & liq
            rows.append(
                _summarize(
                    df,
                    spread_and_liq,
                    f"spread_lt_18_and_side_oi_ge_{oi}_vol_ge_{vol}_else_stock_100sh",
                    oi,
                    vol,
                )
            )

    result = pd.DataFrame([r.__dict__ for r in rows])
    result = result.sort_values("total_pnl_with_fallback", ascending=False)
    result.to_csv(OUT_DIR / "option_liquidity_stock_fallback_summary.csv", index=False)

    liquidity_by_ticker = (
        df.groupby(["ticker", "side"], dropna=False)
        .agg(
            trades=("symbol", "size"),
            option_pnl=("option_pnl_dollars", "sum"),
            stock_100sh_pnl=("stock_100sh_pnl", "sum"),
            avg_spread=("entry_spread_pct_mid", "mean"),
            side_cboe_volume=("side_cboe_volume", "max"),
            side_cboe_open_interest=("side_cboe_open_interest", "max"),
            avg_snapshot_age_days=("cboe_snapshot_age_days", "mean"),
        )
        .reset_index()
        .sort_values("option_pnl")
    )
    liquidity_by_ticker.to_csv(OUT_DIR / "option_liquidity_by_ticker_side.csv", index=False)

    print("Wrote:")
    print(f"  {OUT_DIR / 'option_liquidity_joined_trades.csv'}")
    print(f"  {OUT_DIR / 'option_liquidity_stock_fallback_summary.csv'}")
    print(f"  {OUT_DIR / 'option_liquidity_by_ticker_side.csv'}")
    print()
    print("Top policies by option/stock fallback PnL:")
    print(
        result[
            [
                "policy",
                "option_trades",
                "stock_fallback_trades",
                "unknown_liquidity_trades",
                "avg_cboe_snapshot_age_days",
                "total_pnl_with_fallback",
                "delta_vs_baseline",
                "combined_win_rate",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )
    print()
    print("CBOE proxy coverage:")
    print(df["has_cboe_liquidity"].value_counts(dropna=False).to_string())
    print()
    print("Worst ticker/side option PnL with CBOE proxy:")
    print(liquidity_by_ticker.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
