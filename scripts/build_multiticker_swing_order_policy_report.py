from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_ENRICHED = {
    "may": Path("UI/swing_audit/policy_filter_experiments_20260607/may_enriched_trades.csv"),
    "june_1_5": Path("UI/swing_audit/policy_filter_experiments_20260607/june_1_5_enriched_trades.csv"),
}
DEFAULT_PROFILES = Path("theme_expansion/data/ticker_profiles_new.csv")
DEFAULT_UNIVERSE = Path("Data/shared/universe/shared_universe.csv")
DEFAULT_OUT_DIR = Path("UI/swing_audit/order_policy_report_20260607")


def _metrics(df: pd.DataFrame, *, window: str, policy: str, baseline_n: int) -> dict:
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
        "policy": policy,
        "trades": int(len(pnl)),
        "kept_pct": float(len(pnl) / baseline_n) if baseline_n else 0.0,
        "longs": int((df["direction"] == 1).sum()) if len(df) else 0,
        "shorts": int((df["direction"] == -1).sum()) if len(df) else 0,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "profit_factor": float(pf),
        "sharpe": float(sharpe),
        "avg_trade_pp": float(pnl.mean() * 100.0) if len(pnl) else 0.0,
        "total_pnl_pp": float(pnl.sum() * 100.0) if len(pnl) else 0.0,
        "max_dd_pp": float(dd.min()) if len(dd) else 0.0,
    }


def _load_profiles(path: Path, universe_path: Path) -> pd.DataFrame:
    frames = []
    if path.exists():
        p = pd.read_csv(path)
        p["ticker"] = p["ticker"].astype(str).str.upper()
        frames.append(p[["ticker", "sector", "sectorKey", "industry", "industryKey"]])
    if universe_path.exists():
        u = pd.read_csv(universe_path)
        u["ticker"] = u["ticker"].astype(str).str.upper()
        keep = [c for c in ["ticker", "market_cap_bucket", "asset_type"] if c in u.columns]
        frames.append(u[keep])
    if not frames:
        return pd.DataFrame(columns=["ticker"])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="ticker", how="outer")
    return out.drop_duplicates("ticker")


def _prepare(df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out = out.merge(profiles, on="ticker", how="left")
    sector_text = (
        out.get("sectorKey", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
        + " "
        + out.get("sector", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
        + " "
        + out.get("industryKey", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
        + " "
        + out.get("industry", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    )
    out["is_healthcare"] = sector_text.str.contains("health|pharma|biotech|medical")
    out["is_defensive_sector"] = sector_text.str.contains("health|utilities|consumer-defensive|staples")
    out["is_high_beta"] = (
        pd.to_numeric(out["stock_beta_bucket"], errors="coerce").ge(2)
        | pd.to_numeric(out["beta_like_spy_64"], errors="coerce").ge(1.2)
        | pd.to_numeric(out["volatility_pctile_rolling"], errors="coerce").ge(0.75)
    )
    out["is_growth_or_high_beta"] = out["is_growth_theme"].fillna(False).astype(bool) | out["is_high_beta"]
    out["qqq_pullback"] = pd.to_numeric(out["qqq_ret_16"], errors="coerce").lt(0)
    out["qqq_hard_pullback"] = pd.to_numeric(out["qqq_ret_16"], errors="coerce").lt(-0.005)
    out["rs_qqq_positive"] = pd.to_numeric(out["rel_str_qqq_4"], errors="coerce").gt(0)
    out["rs_spy_positive"] = pd.to_numeric(out["rel_str_spy_16"], errors="coerce").gt(0)
    out["daily_up"] = pd.to_numeric(out["daily_trend_state"], errors="coerce").ge(0)
    out["near_top_extended"] = (
        pd.to_numeric(out["daily_range_pos_20"], errors="coerce").gt(0.90)
        & pd.to_numeric(out["zscore_close_64"], errors="coerce").gt(2.0)
    )
    out["near_bottom_extended"] = (
        pd.to_numeric(out["daily_range_pos_20"], errors="coerce").lt(0.10)
        & pd.to_numeric(out["zscore_close_64"], errors="coerce").lt(-2.0)
    )
    return out


def _apply_cap(df: pd.DataFrame, max_open: int, max_growth_longs: int | None, max_net_beta: float | None) -> pd.DataFrame:
    accepted = []
    active: list[dict] = []
    for idx, row in df.sort_values(["signal_time", "exit_time"]).iterrows():
        active = [a for a in active if a["exit_time"] > row["signal_time"]]
        if len(active) >= max_open:
            continue
        if max_growth_longs is not None and row["direction"] == 1 and bool(row["is_growth_or_high_beta"]):
            if sum(1 for a in active if a["direction"] == 1 and a["growth"]) >= max_growth_longs:
                continue
        beta = float(row["beta_like_spy_64"]) if pd.notna(row["beta_like_spy_64"]) else 1.0
        if max_net_beta is not None:
            net_beta = sum(a["direction"] * a["beta"] for a in active)
            if abs(net_beta + row["direction"] * beta) > max_net_beta:
                continue
        accepted.append(idx)
        active.append(
            {
                "exit_time": row["exit_time"],
                "direction": int(row["direction"]),
                "growth": bool(row["is_growth_or_high_beta"]),
                "beta": beta,
            }
        )
    return df.loc[accepted].copy()


def _policy_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    long = df["direction"].eq(1)
    short = df["direction"].eq(-1)
    pullback = df["qqq_pullback"]
    hard = df["qqq_hard_pullback"]
    growth = df["is_growth_or_high_beta"]
    defensive = df["is_defensive_sector"] | df["is_healthcare"]
    rs_ok_long = df["rs_qqq_positive"] | df["rs_spy_positive"]
    rs_ok_short = ~df["rs_qqq_positive"]
    not_chase_long = ~(df["near_top_extended"] & ~df["breakout_64"].fillna(False).astype(bool))
    not_chase_short = ~(df["near_bottom_extended"] & ~df["breakdown_64"].fillna(False).astype(bool))
    sr_room = (
        long
        & (df["breakout_64"].fillna(False).astype(bool) | pd.to_numeric(df["room_to_64_high_pct"], errors="coerce").ge(3))
    ) | (
        short
        & (df["breakdown_64"].fillna(False).astype(bool) | pd.to_numeric(df["room_to_64_low_pct"], errors="coerce").ge(3))
    )

    aggressive_mask = (
        (long & ~(hard & growth) & not_chase_long)
        | (short & (pullback | growth) & not_chase_short)
    )
    balanced_mask = (
        (long & ((~pullback) | defensive | rs_ok_long) & ~(pullback & growth) & not_chase_long)
        | (short & pullback & (growth | rs_ok_short) & not_chase_short)
    )
    defensive_mask = (
        (long & ((~pullback & rs_ok_long) | defensive) & ~growth & not_chase_long)
        | (short & pullback & growth & rs_ok_short & not_chase_short)
    )

    frames = {
        "baseline_all": df,
        "calls_only": df[long],
        "puts_only_all_tickers": df[short],
        "growth_highbeta_puts_only": df[short & growth],
        "non_growth_puts_only": df[short & ~growth],
        "healthcare_calls_only": df[long & df["is_healthcare"]],
        "aggressive_dynamic_uncapped": df[aggressive_mask],
        "balanced_dynamic_uncapped": df[balanced_mask],
        "defensive_dynamic_uncapped": df[defensive_mask],
        "sr_room_only": df[sr_room],
    }
    frames["aggressive_profile"] = _apply_cap(frames["aggressive_dynamic_uncapped"], 30, 10, 14.0)
    frames["balanced_profile"] = _apply_cap(frames["balanced_dynamic_uncapped"], 20, 5, 8.0)
    frames["defensive_profile"] = _apply_cap(frames["defensive_dynamic_uncapped"], 12, 3, 5.0)
    return frames


def _segment_summary(df: pd.DataFrame, window: str) -> pd.DataFrame:
    rows = []
    segment_defs = {
        "all": pd.Series(True, index=df.index),
        "growth_highbeta": df["is_growth_or_high_beta"],
        "non_growth": ~df["is_growth_or_high_beta"],
        "healthcare": df["is_healthcare"],
        "defensive_sector": df["is_defensive_sector"],
        "qqq_pullback": df["qqq_pullback"],
        "qqq_pullback_growth_highbeta": df["qqq_pullback"] & df["is_growth_or_high_beta"],
        "qqq_pullback_non_growth": df["qqq_pullback"] & ~df["is_growth_or_high_beta"],
    }
    for segment, smask in segment_defs.items():
        for side_name, dmask in {"calls": df["direction"].eq(1), "puts": df["direction"].eq(-1)}.items():
            rows.append(_metrics(df[smask & dmask], window=window, policy=f"{segment}_{side_name}", baseline_n=len(df)))
    return pd.DataFrame(rows)


def _theme_side_summary(df: pd.DataFrame, window: str, min_trades: int) -> pd.DataFrame:
    rows = []
    for (theme, direction), group in df.groupby(["theme_1", "direction"], dropna=False):
        if len(group) < min_trades:
            continue
        side = "calls" if int(direction) == 1 else "puts"
        rows.append(_metrics(group, window=window, policy=f"{theme}_{side}", baseline_n=len(df)))
    return pd.DataFrame(rows).sort_values(["total_pnl_pp", "profit_factor"], ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build actionable swing order-policy profile report.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--windows", nargs="*", default=[f"{k}={v}" for k, v in DEFAULT_ENRICHED.items()])
    parser.add_argument("--min-theme-trades", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    profiles = _load_profiles(args.profiles, args.universe)
    all_policy = []
    all_segments = []
    all_themes = []
    for item in args.windows:
        name, raw_path = item.split("=", 1) if "=" in item else (Path(item).stem, item)
        df = _prepare(pd.read_csv(raw_path), profiles)
        frames = _policy_frames(df)
        policy = pd.DataFrame([_metrics(frame, window=name, policy=policy_name, baseline_n=len(df)) for policy_name, frame in frames.items()])
        segments = _segment_summary(df, name)
        themes = _theme_side_summary(df, name, args.min_theme_trades)
        policy.to_csv(args.out_dir / f"{name}_order_policy_profiles.csv", index=False)
        segments.to_csv(args.out_dir / f"{name}_segment_side_summary.csv", index=False)
        themes.to_csv(args.out_dir / f"{name}_theme_side_summary.csv", index=False)
        all_policy.append(policy)
        all_segments.append(segments)
        all_themes.append(themes)
        print(f"\n{name} policies")
        print(policy.sort_values(["profit_factor", "total_pnl_pp"], ascending=False).to_string(index=False))
        print(f"\n{name} key segments")
        print(segments[segments["policy"].str.contains("growth|healthcare|pullback")].sort_values("total_pnl_pp", ascending=False).head(12).to_string(index=False))

    pd.concat(all_policy, ignore_index=True).to_csv(args.out_dir / "combined_order_policy_profiles.csv", index=False)
    pd.concat(all_segments, ignore_index=True).to_csv(args.out_dir / "combined_segment_side_summary.csv", index=False)
    pd.concat(all_themes, ignore_index=True).to_csv(args.out_dir / "combined_theme_side_summary.csv", index=False)
    print(f"\nwrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
