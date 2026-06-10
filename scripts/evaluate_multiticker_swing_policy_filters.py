from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


DEFAULT_WINDOWS = {
    "may": Path("UI/swing_audit/backtest_may_20260607/multiticker_swing_20260501_20260531_replay_trades.csv"),
    "june_1_5": Path("UI/swing_audit/backtest_june_current_20260607/multiticker_swing_20260601_20260605_replay_trades.csv"),
}
FEATURE_DIR = Path("strategies/multi_ticker_swing/data/processed/30m")
THEME_MAP = Path("theme_expansion/data/theme_map_v4.csv")
OUT_DIR = Path("UI/swing_audit/policy_filter_experiments_20260607")

FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "atr_pct_14",
    "range_pos_20",
    "dist_20bar_high",
    "dist_20bar_low",
    "dist_to_recent_swing_high",
    "dist_to_recent_swing_low",
    "breakout_pressure_score",
    "rel_str_qqq_4",
    "rel_str_spy_16",
    "qqq_ret_16",
    "spy_ret_16",
    "beta_like_spy_64",
    "market_regime_proxy",
    "zscore_close_64",
    "percentile_close_64",
    "daily_rsi_14",
    "daily_ema_dist_20",
    "daily_ema_dist_50",
    "daily_trend_state",
    "daily_range_pos_20",
    "volatility_pctile_rolling",
    "stock_beta_bucket",
]

GROWTH_THEME_RE = re.compile(
    r"(ai|semis|compute|software|cloud|cyber|quantum|crypto|fintech|space|robot|ev|solar|battery|"
    r"biotech|genomics|data|digital|networking|optical)",
    re.I,
)


def _load_theme_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "theme_1", "is_growth_theme"])
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    for col in ("theme_1", "theme_2", "theme_3"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.lower()
    joined = df[["theme_1", "theme_2", "theme_3"]].agg(" ".join, axis=1)
    df["is_growth_theme"] = joined.str.contains(GROWTH_THEME_RE)
    return df[["ticker", "theme_1", "theme_2", "theme_3", "is_growth_theme"]].drop_duplicates("ticker")


def _load_ticker_features(ticker: str) -> pd.DataFrame | None:
    path = FEATURE_DIR / f"{ticker.upper()}_features.parquet"
    if not path.exists():
        return None
    cols = [c for c in FEATURE_COLS if c]
    try:
        df = pd.read_parquet(path, columns=cols)
    except Exception:
        return None
    if "timestamp" not in df.columns:
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    df["prior_high_64"] = pd.to_numeric(df["high"], errors="coerce").rolling(64, min_periods=20).max().shift(1)
    df["prior_low_64"] = pd.to_numeric(df["low"], errors="coerce").rolling(64, min_periods=20).min().shift(1)
    close = pd.to_numeric(df["close"], errors="coerce")
    df["room_to_64_high_pct"] = (df["prior_high_64"] / close - 1.0) * 100.0
    df["room_to_64_low_pct"] = (close / df["prior_low_64"] - 1.0) * 100.0
    df["breakout_64"] = close.ge(df["prior_high_64"] * 0.995)
    df["breakdown_64"] = close.le(df["prior_low_64"] * 1.005)
    return df[
        ["timestamp"]
        + FEATURE_COLS
        + ["room_to_64_high_pct", "room_to_64_low_pct", "breakout_64", "breakdown_64"]
    ]


def enrich_trades(trades: pd.DataFrame, theme_map: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    enriched = []
    for ticker, group in out.groupby("ticker", sort=False):
        feats = _load_ticker_features(ticker)
        if feats is None or feats.empty:
            g = group.copy()
            for col in FEATURE_COLS + ["room_to_64_high_pct", "room_to_64_low_pct", "breakout_64", "breakdown_64"]:
                g[col] = float("nan")
            enriched.append(g)
            continue
        g = group.sort_values("signal_time").copy()
        merged = pd.merge_asof(
            g,
            feats,
            left_on="signal_time",
            right_on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=31),
        ).drop(columns=["timestamp"], errors="ignore")
        enriched.append(merged)
    out = pd.concat(enriched, ignore_index=True) if enriched else out
    out = out.merge(theme_map, on="ticker", how="left")
    out["is_growth_theme"] = out["is_growth_theme"].fillna(False)
    return out


def metrics(df: pd.DataFrame, name: str, window: str, baseline_trades: int) -> dict:
    pnl = pd.to_numeric(df["pnl_pct"], errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else math.inf
    sharpe = pnl.mean() / pnl.std() * math.sqrt(252) if len(pnl) > 1 and pnl.std() > 0 else 0.0
    ordered = df.assign(_pnl=pd.to_numeric(df["pnl_pct"], errors="coerce")).sort_values("exit_time")
    curve = ordered["_pnl"].fillna(0.0).cumsum() * 100.0
    dd = curve - curve.cummax()
    return {
        "window": window,
        "policy": name,
        "trades": int(len(pnl)),
        "kept_pct": float(len(pnl) / baseline_trades) if baseline_trades else 0.0,
        "longs": int((df["direction"] == 1).sum()) if len(df) else 0,
        "shorts": int((df["direction"] == -1).sum()) if len(df) else 0,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "profit_factor": float(pf),
        "sharpe": float(sharpe),
        "avg_trade_pp": float(pnl.mean() * 100.0) if len(pnl) else 0.0,
        "total_pnl_pp": float(pnl.sum() * 100.0) if len(pnl) else 0.0,
        "max_dd_pp": float(dd.min()) if len(dd) else 0.0,
    }


def _finite_bool(s: pd.Series, default: bool = False) -> pd.Series:
    return s.fillna(default).astype(bool)


def policy_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    d = pd.to_numeric(df["direction"], errors="coerce")
    long = d.eq(1)
    short = d.eq(-1)
    qqq_pullback = pd.to_numeric(df["qqq_ret_16"], errors="coerce").lt(0)
    high_beta = pd.to_numeric(df["stock_beta_bucket"], errors="coerce").ge(2) | pd.to_numeric(
        df["beta_like_spy_64"], errors="coerce"
    ).ge(1.2)
    growth = _finite_bool(df["is_growth_theme"])
    rs_qqq_pos = pd.to_numeric(df["rel_str_qqq_4"], errors="coerce").gt(0)
    rs_spy_pos = pd.to_numeric(df["rel_str_spy_16"], errors="coerce").gt(0)
    daily_up = pd.to_numeric(df["daily_trend_state"], errors="coerce").ge(0)
    daily_down = pd.to_numeric(df["daily_trend_state"], errors="coerce").le(0)
    not_extended_long = ~(
        pd.to_numeric(df["daily_range_pos_20"], errors="coerce").gt(0.92)
        & pd.to_numeric(df["zscore_close_64"], errors="coerce").gt(2.25)
        & ~_finite_bool(df["breakout_64"])
    )
    not_extended_short = ~(
        pd.to_numeric(df["daily_range_pos_20"], errors="coerce").lt(0.08)
        & pd.to_numeric(df["zscore_close_64"], errors="coerce").lt(-2.25)
        & ~_finite_bool(df["breakdown_64"])
    )
    sr_break = (long & _finite_bool(df["breakout_64"])) | (short & _finite_bool(df["breakdown_64"]))
    sr_room = (
        long
        & (
            _finite_bool(df["breakout_64"])
            | pd.to_numeric(df["room_to_64_high_pct"], errors="coerce").ge(3.0)
        )
    ) | (
        short
        & (
            _finite_bool(df["breakdown_64"])
            | pd.to_numeric(df["room_to_64_low_pct"], errors="coerce").ge(3.0)
        )
    )
    return {
        "baseline_all": pd.Series(True, index=df.index),
        "calls_only": long,
        "puts_only": short,
        "direction_requires_qqq_rs": (long & rs_qqq_pos) | (short & ~rs_qqq_pos),
        "direction_requires_spy_rs16": (long & rs_spy_pos) | (short & ~rs_spy_pos),
        "daily_trend_permission": (long & daily_up) | (short & daily_down),
        "block_high_beta_longs_when_qqq_pullback": ~(long & qqq_pullback & high_beta),
        "block_growth_longs_when_qqq_pullback": ~(long & qqq_pullback & growth),
        "chase_filter": (long & not_extended_long) | (short & not_extended_short),
        "sr_breakout_only": sr_break,
        "sr_breakout_or_3pct_room": sr_room,
        "calls_only_rs_spy16_daily_up": long & rs_spy_pos & daily_up,
        "calls_only_chase_filtered": long & not_extended_long,
    }


def apply_portfolio_cap(
    df: pd.DataFrame,
    *,
    max_open: int,
    max_growth_longs: int | None = None,
    max_abs_net_beta: float | None = None,
) -> pd.DataFrame:
    accepted = []
    active: list[dict] = []
    ordered = df.sort_values(["signal_time", "exit_time"]).copy()
    for idx, row in ordered.iterrows():
        now = row["signal_time"]
        active = [a for a in active if a["exit_time"] > now]
        if len(active) >= max_open:
            continue
        if max_growth_longs is not None and int(row["direction"]) == 1 and bool(row.get("is_growth_theme", False)):
            growth_longs = sum(1 for a in active if a["direction"] == 1 and a["is_growth_theme"])
            if growth_longs >= max_growth_longs:
                continue
        beta = float(row["beta_like_spy_64"]) if pd.notna(row.get("beta_like_spy_64")) else 1.0
        if max_abs_net_beta is not None:
            net = sum(a["direction"] * a["beta"] for a in active)
            if abs(net + int(row["direction"]) * beta) > max_abs_net_beta:
                continue
        accepted.append(idx)
        active.append(
            {
                "exit_time": row["exit_time"],
                "direction": int(row["direction"]),
                "is_growth_theme": bool(row.get("is_growth_theme", False)),
                "beta": beta,
            }
        )
    return df.loc[accepted].copy()


def evaluate_window(name: str, path: Path, theme_map: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(path)
    enriched = enrich_trades(trades, theme_map)
    enriched_out = out_dir / f"{name}_enriched_trades.csv"
    enriched.to_csv(enriched_out, index=False)

    rows = []
    masks = policy_masks(enriched)
    for policy, mask in masks.items():
        rows.append(metrics(enriched[mask.fillna(False)].copy(), policy, name, len(enriched)))
    cap_variants = {
        "cap_20_open": apply_portfolio_cap(enriched, max_open=20),
        "cap_12_open": apply_portfolio_cap(enriched, max_open=12),
        "cap_20_growth_longs_5": apply_portfolio_cap(enriched, max_open=20, max_growth_longs=5),
        "cap_20_net_beta_8": apply_portfolio_cap(enriched, max_open=20, max_abs_net_beta=8.0),
        "cap_12_growth_longs_3_net_beta_5": apply_portfolio_cap(
            enriched, max_open=12, max_growth_longs=3, max_abs_net_beta=5.0
        ),
    }
    for policy, pdf in cap_variants.items():
        rows.append(metrics(pdf, policy, name, len(enriched)))

    summary = pd.DataFrame(rows).sort_values(["profit_factor", "sharpe", "total_pnl_pp"], ascending=False)
    return summary, enriched


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate post-trade policy filters/caps for multi-ticker swing replays.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--theme-map", type=Path, default=THEME_MAP)
    parser.add_argument("--windows", nargs="*", default=[f"{k}={v}" for k, v in DEFAULT_WINDOWS.items()])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    theme_map = _load_theme_map(args.theme_map)

    all_summaries = []
    for item in args.windows:
        if "=" in item:
            name, raw_path = item.split("=", 1)
            path = Path(raw_path)
        else:
            path = Path(item)
            name = path.stem
        summary, enriched = evaluate_window(name, path, theme_map, args.out_dir)
        all_summaries.append(summary)
        summary.to_csv(args.out_dir / f"{name}_policy_summary.csv", index=False)
        apld = enriched[enriched["ticker"].eq("APLD")].copy()
        if not apld.empty:
            apld_rows = []
            for policy, mask in policy_masks(apld).items():
                apld_rows.append(metrics(apld[mask.fillna(False)].copy(), policy, f"{name}_APLD", len(apld)))
            pd.DataFrame(apld_rows).sort_values(["profit_factor", "total_pnl_pp"], ascending=False).to_csv(
                args.out_dir / f"{name}_apld_policy_summary.csv", index=False
            )
        print(f"\n{name}")
        print(summary.head(12).to_string(index=False))

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(args.out_dir / "combined_policy_summary.csv", index=False)
    print(f"\nwrote {args.out_dir / 'combined_policy_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
