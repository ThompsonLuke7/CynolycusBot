from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE = Path("Data/analysis/multi_ticker_swing_live/experiments")
MIX_PATH = BASE / "stock_option_mix" / "live_stock_option_mix_dataset.csv"
RECENT_PATH = BASE / "multiticker_20260528_20260529_closed_performance_rebuilt.csv"
OUT = BASE / "real_account_policy_replay"


@dataclass(frozen=True)
class ReplayPolicy:
    name: str
    calls_only: bool = True
    fresh_only: bool = True
    max_open_positions: int = 2
    max_new_trades_per_day: int = 4
    max_contracts_per_trade: int = 1
    max_premium_per_trade: float = 250.0
    max_open_premium: float = 500.0
    account_size: float = 1000.0
    min_cash_after_entry: float = 250.0
    min_atr_pct: float = 0.010
    max_dte: int = 7
    block_after_time: str = "15:15"
    block_0dte_after_time: str = "12:30"


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _normalize_mix(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    rename = {
        "symbol": "option_symbol",
        "entry_price_underlying": "entry_underlying",
        "exit_price": "exit_underlying",
        "pnl_dollars": "option_pnl_dollars",
        "pnl_pct_option": "option_ret_fraction",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "option_ret_pct" not in df.columns and "option_ret_fraction" in df.columns:
        df["option_ret_pct"] = pd.to_numeric(df["option_ret_fraction"], errors="coerce") * 100.0
    if "entry_price_option" not in df.columns and "option_entry_price" in df.columns:
        df["entry_price_option"] = df["option_entry_price"]
    if "exit_price_option" not in df.columns and "option_exit_price" in df.columns:
        df["exit_price_option"] = df["option_exit_price"]
    if "entry_date_et" not in df.columns:
        df["entry_date_et"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date
    if "is_fresh" not in df.columns:
        df["is_fresh"] = True
    df["source_slice"] = source_name
    return df


def load_slices() -> dict[str, pd.DataFrame]:
    mix = _normalize_mix(pd.read_csv(MIX_PATH), "all_available")
    recent = _normalize_mix(pd.read_csv(RECENT_PATH), "recent_0528_0529")
    frames = {
        "recent_0528_0529": recent,
        "recent_fresh_0528_0529": recent[recent["is_fresh"].fillna(False).astype(bool)].copy(),
        "all_available": mix,
        "all_fresh_available": mix[mix["is_fresh"].fillna(False).astype(bool)].copy(),
    }
    clean: dict[str, pd.DataFrame] = {}
    for name, df in frames.items():
        df = _num(
            df,
            [
                "direction",
                "entry_underlying",
                "exit_underlying",
                "entry_price_option",
                "exit_price_option",
                "option_pnl_dollars",
                "option_ret_pct",
                "atr_at_entry",
                "dte_at_entry",
                "qty",
            ],
        )
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
        if "exit_time" not in df.columns and "closed_ts" in df.columns:
            df["exit_time"] = df["closed_ts"]
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
        df = df.dropna(
            subset=[
                "ticker",
                "direction",
                "entry_time",
                "exit_time",
                "entry_underlying",
                "entry_price_option",
                "option_pnl_dollars",
            ]
        ).copy()
        df = df[df["direction"].isin([1, -1])]
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["option_symbol"] = df["option_symbol"].astype(str).str.upper()
        df["atr_pct_entry"] = df["atr_at_entry"] / df["entry_underlying"]
        df["entry_time_et"] = df["entry_time"].dt.tz_convert("America/New_York")
        df["exit_time_et"] = df["exit_time"].dt.tz_convert("America/New_York")
        df["entry_day"] = df["entry_time_et"].dt.date.astype(str)
        clean[name] = df.sort_values(["entry_time", "option_symbol"]).reset_index(drop=True)
    return clean


def _hhmm_minutes(text: str) -> int:
    hh, mm = str(text).split(":", 1)
    return int(hh) * 60 + int(mm)


def _time_minutes(ts: pd.Timestamp) -> int:
    return int(ts.hour) * 60 + int(ts.minute)


def _skip_reason(row: pd.Series, policy: ReplayPolicy) -> str | None:
    if policy.calls_only and int(row["direction"]) < 0:
        return "calls_only"
    if policy.fresh_only and not bool(row.get("is_fresh", False)):
        return "not_fresh"
    atr_pct = row.get("atr_pct_entry")
    if pd.notna(atr_pct) and float(atr_pct) < policy.min_atr_pct:
        return "atr_pct_too_low"
    dte = row.get("dte_at_entry")
    if pd.notna(dte) and int(float(dte)) > policy.max_dte:
        return "dte_too_high"
    entry_et = row["entry_time_et"]
    if _time_minutes(entry_et) >= _hhmm_minutes(policy.block_after_time):
        return "after_entry_cutoff"
    if pd.notna(dte) and int(float(dte)) == 0 and _time_minutes(entry_et) >= _hhmm_minutes(policy.block_0dte_after_time):
        return "zero_dte_after_cutoff"
    premium = float(row["entry_price_option"]) * 100.0
    if premium > policy.max_premium_per_trade:
        return "premium_per_trade"
    return None


def replay(df: pd.DataFrame, policy: ReplayPolicy) -> tuple[pd.DataFrame, pd.DataFrame]:
    open_positions: list[dict[str, Any]] = []
    day_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        entry_time = row["entry_time"]
        open_positions = [p for p in open_positions if p["exit_time"] > entry_time]
        open_premium = sum(float(p["premium"]) for p in open_positions)
        day = str(row["entry_day"])
        premium = float(row["entry_price_option"]) * 100.0
        reason = _skip_reason(row, policy)
        if reason is None and len(open_positions) >= policy.max_open_positions:
            reason = "max_open_positions"
        if reason is None and day_counts.get(day, 0) >= policy.max_new_trades_per_day:
            reason = "max_new_trades_per_day"
        if reason is None and open_premium + premium > policy.max_open_premium:
            reason = "max_open_premium"
        if reason is None and open_premium + premium > policy.account_size - policy.min_cash_after_entry:
            reason = "cash_reserve"

        accepted = reason is None
        qty = min(int(policy.max_contracts_per_trade), int(np.floor(policy.max_premium_per_trade / premium))) if premium > 0 else 0
        if accepted and qty < 1:
            accepted = False
            reason = "qty_zero"
        pnl = float(row["option_pnl_dollars"]) * qty if accepted else 0.0
        premium_at_risk = premium * qty if accepted else 0.0
        if accepted:
            day_counts[day] = day_counts.get(day, 0) + 1
            open_positions.append(
                {
                    "option_symbol": row["option_symbol"],
                    "exit_time": row["exit_time"],
                    "premium": premium_at_risk,
                }
            )
        rows.append(
            {
                "accepted": accepted,
                "skip_reason": reason or "accepted",
                "qty": qty if accepted else 0,
                "premium_at_risk": premium_at_risk,
                "policy_pnl": pnl,
                "policy_return_on_premium_pct": (pnl / premium_at_risk * 100.0) if premium_at_risk else np.nan,
                "open_premium_before": open_premium,
                "day_entries_before": day_counts.get(day, 0) - (1 if accepted else 0),
                **row.to_dict(),
            }
        )

    out = pd.DataFrame(rows)
    summary = summarize(out, policy)
    return out, summary


def summarize(out: pd.DataFrame, policy: ReplayPolicy) -> pd.DataFrame:
    accepted = out[out["accepted"]].copy()
    skipped = out[~out["accepted"]].copy()
    gross_premium = accepted["premium_at_risk"].sum()
    total_pnl = accepted["policy_pnl"].sum()
    return pd.DataFrame(
        [
            {
                "policy": policy.name,
                "trades_seen": len(out),
                "accepted": len(accepted),
                "skipped": len(skipped),
                "total_pnl": total_pnl,
                "return_on_1000_account_pct": total_pnl / policy.account_size * 100.0,
                "return_on_deployed_premium_pct": total_pnl / gross_premium * 100.0 if gross_premium else np.nan,
                "gross_premium_deployed": gross_premium,
                "avg_premium": accepted["premium_at_risk"].mean() if len(accepted) else np.nan,
                "win_rate": (accepted["policy_pnl"] > 0).mean() if len(accepted) else np.nan,
                "avg_option_ret_pct": accepted["option_ret_pct"].mean() if len(accepted) else np.nan,
                "median_option_ret_pct": accepted["option_ret_pct"].median() if len(accepted) else np.nan,
                "calls": int((accepted["direction"] == 1).sum()) if len(accepted) else 0,
                "puts": int((accepted["direction"] == -1).sum()) if len(accepted) else 0,
                "max_open_positions": policy.max_open_positions,
                "max_new_trades_per_day": policy.max_new_trades_per_day,
                "max_premium_per_trade": policy.max_premium_per_trade,
                "max_open_premium": policy.max_open_premium,
                "min_atr_pct": policy.min_atr_pct,
                "fresh_only": policy.fresh_only,
                "calls_only": policy.calls_only,
                "top_skip_reasons": skipped["skip_reason"].value_counts().head(5).to_dict(),
            }
        ]
    )


def policy_grid() -> list[ReplayPolicy]:
    policies = [
        ReplayPolicy(name="real_defaults"),
        ReplayPolicy(name="defaults_more_orders", max_open_positions=3, max_new_trades_per_day=6, max_open_premium=750.0),
        ReplayPolicy(name="defaults_more_premium", max_premium_per_trade=350.0, max_open_premium=700.0),
        ReplayPolicy(name="strict_atr_1p5", min_atr_pct=0.015),
        ReplayPolicy(name="loose_atr_0p75", min_atr_pct=0.0075),
        ReplayPolicy(name="calls_not_fresh_limited", fresh_only=False),
        ReplayPolicy(name="calls_only_no_daily_cap", max_new_trades_per_day=99),
        ReplayPolicy(name="calls_only_one_open", max_open_positions=1, max_open_premium=250.0),
        ReplayPolicy(name="allow_puts_diagnostic", calls_only=False),
    ]
    for max_open in [1, 2, 3, 4]:
        for max_day in [2, 4, 6, 99]:
            for max_premium in [150.0, 250.0, 350.0]:
                for min_atr in [0.0075, 0.010, 0.015, 0.020]:
                    policies.append(
                        ReplayPolicy(
                            name=f"grid_open{max_open}_day{max_day}_prem{int(max_premium)}_atr{str(min_atr).replace('.', 'p')}",
                            max_open_positions=max_open,
                            max_new_trades_per_day=max_day,
                            max_premium_per_trade=max_premium,
                            max_open_premium=max_premium * max_open,
                            min_atr_pct=min_atr,
                        )
                    )
    return policies


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    slices = load_slices()
    all_summaries = []
    for slice_name, df in slices.items():
        slice_summaries = []
        for policy in policy_grid():
            replay_df, summary = replay(df, policy)
            summary.insert(0, "slice", slice_name)
            slice_summaries.append(summary)
            if policy.name in {"real_defaults", "defaults_more_orders", "strict_atr_1p5", "allow_puts_diagnostic"}:
                replay_df.to_csv(OUT / f"{slice_name}_{policy.name}_trades.csv", index=False)
        slice_summary = pd.concat(slice_summaries, ignore_index=True)
        slice_summary = slice_summary.sort_values(["total_pnl", "accepted"], ascending=[False, False])
        slice_summary.to_csv(OUT / f"{slice_name}_policy_grid_summary.csv", index=False)
        all_summaries.append(slice_summary)
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(OUT / "combined_policy_grid_summary.csv", index=False)

    for slice_name in ["recent_0528_0529", "recent_fresh_0528_0529", "all_available", "all_fresh_available"]:
        print(f"\n=== {slice_name} ===")
        view = combined[combined["slice"] == slice_name]
        important = view[view["policy"].isin(["real_defaults", "defaults_more_orders", "strict_atr_1p5", "allow_puts_diagnostic"])]
        print(important.sort_values("policy").to_string(index=False))
        print("\nTop 8:")
        print(view.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
