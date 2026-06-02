from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


BASE = Path("Data/analysis/multi_ticker_swing_live")
EXP = BASE / "experiments"
OUT = EXP / "stock_option_mix"


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _from_paired() -> pd.DataFrame:
    path = BASE / "paired_option_trades.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = _num(
        df,
        [
            "direction",
            "entry_price_underlying",
            "exit_price",
            "exit_pnl_pct",
            "pnl_dollars",
            "pnl_pct_option",
            "entry_price_option",
            "dte_at_entry",
            "p_dir",
            "ev_score",
            "atr_at_entry",
            "qty",
        ],
    )
    df = df[df["direction"].isin([1, -1])].copy()
    if df.empty:
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    df["source"] = "paired_option_trades"
    df["is_fresh"] = True
    df["option_symbol"] = df["symbol"]
    df["entry_underlying"] = df["entry_price_underlying"]
    df["exit_underlying"] = df["exit_price"]
    df["underlying_signed_ret_pct"] = df["exit_pnl_pct"] * 100.0
    df["option_ret_pct"] = df["pnl_pct_option"] * 100.0
    df["option_pnl_dollars"] = df["pnl_dollars"]
    df["stock_100sh_pnl"] = df["direction"] * (df["exit_underlying"] - df["entry_underlying"]) * 100.0
    return df


def _from_rebuilt() -> pd.DataFrame:
    frames = []
    for path in [
        EXP / "multiticker_20260528_20260529_closed_performance_rebuilt.csv",
        EXP / "multiticker_20260529_closed_performance.csv",
    ]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = _num(
        df,
        [
            "direction",
            "entry_underlying",
            "exit_underlying",
            "underlying_signed_ret_pct",
            "option_entry_price",
            "option_exit_price",
            "qty",
            "option_pnl_dollars",
            "option_ret_pct",
            "stock_100sh_pnl",
            "atr_at_entry",
        ],
    )
    df = df.drop_duplicates(["option_symbol", "closed_ts"], keep="last")
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["closed_ts"], utc=True, errors="coerce")
    df["source"] = "audit_rebuilt_20260528_20260529"
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["option_type"] = np.where(df["direction"] == 1, "C", "P")
    df["entry_price_option"] = df["option_entry_price"]
    df["pnl_dollars"] = df["option_pnl_dollars"]
    return df


def load_trades() -> pd.DataFrame:
    df = pd.concat([_from_paired(), _from_rebuilt()], ignore_index=True, sort=False)
    df = df.dropna(
        subset=[
            "ticker",
            "direction",
            "entry_underlying",
            "exit_underlying",
            "underlying_signed_ret_pct",
            "option_ret_pct",
            "option_pnl_dollars",
            "stock_100sh_pnl",
        ]
    ).copy()
    df = df[np.isfinite(df["entry_underlying"]) & (df["entry_underlying"] > 0)].copy()
    df["atr_pct_entry"] = df["atr_at_entry"] / df["entry_underlying"]
    df["route_side"] = np.where(df["direction"] == 1, "long_call", "short_put")
    df["entry_date_et"] = df["entry_time"].dt.tz_convert("America/New_York").dt.date
    return df


def summarize(df: pd.DataFrame, policy: str, mask: pd.Series) -> dict[str, float | int | str]:
    route_option = mask.reindex(df.index).fillna(False)
    mixed_return_pct = np.where(route_option, df["option_ret_pct"], df["underlying_signed_ret_pct"])
    mixed_dollars = np.where(route_option, df["option_pnl_dollars"], df["stock_100sh_pnl"])
    stock_ret = df["underlying_signed_ret_pct"]
    opt_ret = df["option_ret_pct"]
    return {
        "policy": policy,
        "trades": int(len(df)),
        "option_trades": int(route_option.sum()),
        "stock_trades": int((~route_option).sum()),
        "mixed_avg_return_pct": float(np.nanmean(mixed_return_pct)),
        "mixed_median_return_pct": float(np.nanmedian(mixed_return_pct)),
        "mixed_win_rate": float(np.nanmean(mixed_return_pct > 0)),
        "mixed_total_dollars": float(np.nansum(mixed_dollars)),
        "all_options_total_dollars": float(df["option_pnl_dollars"].sum()),
        "all_stock_100sh_total_dollars": float(df["stock_100sh_pnl"].sum()),
        "all_options_avg_return_pct": float(opt_ret.mean()),
        "all_stock_avg_return_pct": float(stock_ret.mean()),
        "all_options_win_rate": float((opt_ret > 0).mean()),
        "all_stock_win_rate": float((stock_ret > 0).mean()),
        "option_route_avg_return_pct": float(df.loc[route_option, "option_ret_pct"].mean()) if route_option.any() else np.nan,
        "stock_route_avg_return_pct": float(df.loc[~route_option, "underlying_signed_ret_pct"].mean()) if (~route_option).any() else np.nan,
    }


def run_policy_grid(df: pd.DataFrame) -> pd.DataFrame:
    policies: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("all_options", lambda x: pd.Series(True, index=x.index)),
        ("all_stock", lambda x: pd.Series(False, index=x.index)),
        ("calls_options_puts_stock", lambda x: x["direction"].eq(1)),
        ("puts_options_calls_stock", lambda x: x["direction"].eq(-1)),
        ("fresh_options_restored_stock", lambda x: x["is_fresh"].fillna(False).astype(bool)),
        ("fresh_calls_options_else_stock", lambda x: x["is_fresh"].fillna(False).astype(bool) & x["direction"].eq(1)),
        ("atr_1pct_options_else_stock", lambda x: x["atr_pct_entry"].ge(0.010)),
        ("atr_1p5pct_options_else_stock", lambda x: x["atr_pct_entry"].ge(0.015)),
        ("atr_2pct_options_else_stock", lambda x: x["atr_pct_entry"].ge(0.020)),
        ("calls_atr_1pct_options_else_stock", lambda x: x["direction"].eq(1) & x["atr_pct_entry"].ge(0.010)),
        ("calls_atr_1p5pct_options_else_stock", lambda x: x["direction"].eq(1) & x["atr_pct_entry"].ge(0.015)),
        ("calls_atr_2pct_options_else_stock", lambda x: x["direction"].eq(1) & x["atr_pct_entry"].ge(0.020)),
        ("dte_ge_1_options_else_stock", lambda x: x["dte_at_entry"].ge(1)),
        ("dte_ge_2_options_else_stock", lambda x: x["dte_at_entry"].ge(2)),
        ("same_day_stock_else_options", lambda x: ~x["dte_at_entry"].eq(0)),
        ("pdir_85_options_else_stock", lambda x: x["p_dir"].ge(0.85)),
        ("pdir_90_options_else_stock", lambda x: x["p_dir"].ge(0.90)),
        ("pdir_90_atr_1pct_options_else_stock", lambda x: x["p_dir"].ge(0.90) & x["atr_pct_entry"].ge(0.010)),
        ("pdir_90_atr_1p5pct_options_else_stock", lambda x: x["p_dir"].ge(0.90) & x["atr_pct_entry"].ge(0.015)),
    ]
    rows = []
    for name, fn in policies:
        rows.append(summarize(df, name, fn(df)))
    return pd.DataFrame(rows).sort_values("mixed_total_dollars", ascending=False)


def side_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in [
        ("all", df),
        ("calls", df[df["direction"] == 1]),
        ("puts", df[df["direction"] == -1]),
        ("fresh", df[df["is_fresh"].fillna(False).astype(bool)]),
        ("restored", df[~df["is_fresh"].fillna(False).astype(bool)]),
        ("fresh_calls", df[df["is_fresh"].fillna(False).astype(bool) & df["direction"].eq(1)]),
        ("fresh_puts", df[df["is_fresh"].fillna(False).astype(bool) & df["direction"].eq(-1)]),
    ]:
        if sub.empty:
            continue
        rows.append(
            {
                "bucket": name,
                "trades": len(sub),
                "option_total_dollars": sub["option_pnl_dollars"].sum(),
                "stock_100sh_total_dollars": sub["stock_100sh_pnl"].sum(),
                "option_avg_return_pct": sub["option_ret_pct"].mean(),
                "option_median_return_pct": sub["option_ret_pct"].median(),
                "option_win_rate": (sub["option_ret_pct"] > 0).mean(),
                "stock_avg_return_pct": sub["underlying_signed_ret_pct"].mean(),
                "stock_median_return_pct": sub["underlying_signed_ret_pct"].median(),
                "stock_win_rate": (sub["underlying_signed_ret_pct"] > 0).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("option_total_dollars", ascending=False)


def long_only_capital_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    slices = {
        "all_longs_calls": df[df["direction"].eq(1)],
        "fresh_longs_calls": df[df["direction"].eq(1) & df["is_fresh"].fillna(False).astype(bool)],
        "recent_0528_0529_longs_calls": df[
            df["direction"].eq(1) & df["entry_date_et"].astype(str).isin(["2026-05-28", "2026-05-29"])
        ],
        "recent_fresh_longs_calls": df[
            df["direction"].eq(1)
            & df["is_fresh"].fillna(False).astype(bool)
            & df["entry_date_et"].astype(str).isin(["2026-05-28", "2026-05-29"])
        ],
    }
    for name, sub in slices.items():
        if sub.empty:
            continue
        opt_ret = sub["option_ret_pct"] / 100.0
        stock_ret = sub["underlying_signed_ret_pct"] / 100.0
        premium = sub["entry_price_option"] * 100.0
        contracts = np.floor(1000.0 / premium).clip(lower=0)
        integer_contract_ret = np.where(contracts > 0, contracts * sub["option_pnl_dollars"] / 1000.0, np.nan)
        rows.append(
            {
                "slice": name,
                "trades": len(sub),
                "option_avg_pct": opt_ret.mean() * 100.0,
                "option_median_pct": opt_ret.median() * 100.0,
                "option_win_rate": (opt_ret > 0).mean(),
                "stock_avg_pct": stock_ret.mean() * 100.0,
                "stock_median_pct": stock_ret.median() * 100.0,
                "stock_win_rate": (stock_ret > 0).mean(),
                "option_sum_if_1000_each_fractional_contracts": (1000.0 * opt_ret).sum(),
                "stock_sum_if_1000_each_fractional_shares": (1000.0 * stock_ret).sum(),
                "integer_contract_avg_pct_on_1000": np.nanmean(integer_contract_ret) * 100.0,
                "integer_contract_sum_dollars": np.nansum(1000.0 * integer_contract_ret),
                "trades_affordable_1_contract": (premium <= 1000.0).mean(),
                "avg_contract_premium": premium.mean(),
                "median_contract_premium": premium.median(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_trades()
    df.to_csv(OUT / "live_stock_option_mix_dataset.csv", index=False)
    side = side_summary(df)
    grid = run_policy_grid(df)
    capital = long_only_capital_summary(df)
    side.to_csv(OUT / "live_stock_vs_option_side_summary.csv", index=False)
    grid.to_csv(OUT / "live_stock_option_policy_grid.csv", index=False)
    capital.to_csv(OUT / "long_only_capital_normalized_summary.csv", index=False)
    print("dataset", len(df), "from", df["entry_date_et"].min(), "to", df["entry_date_et"].max())
    print("\nSide summary:")
    print(side.to_string(index=False))
    print("\nLong-only capital-normalized summary:")
    print(capital.to_string(index=False))
    print("\nTop mixed policies by actual dollars:")
    print(grid.head(12).to_string(index=False))
    print("\nTop mixed policies by avg return pct:")
    print(grid.sort_values("mixed_avg_return_pct", ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
