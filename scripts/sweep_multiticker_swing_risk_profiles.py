from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_multiticker_swing_policy_filters import FEATURE_COLS, _load_theme_map, _load_ticker_features
from scripts.plot_multiticker_swing_backtest_overview import _load_best_trades


DEFAULT_SWEEP_DIR = Path("multi_ticker_swing/backtest/results/sweep_v4_shared_20260606")
DEFAULT_RAW_30M_DIR = Path("multi_ticker_swing/data/raw/30m")
DEFAULT_THEME_DAILY = Path("theme_expansion/outputs/theme_daily.parquet")
DEFAULT_THEME_MAP = Path("theme_expansion/data/theme_map_v4.csv")
DEFAULT_PROFILES = Path("theme_expansion/data/ticker_profiles_new.csv")
DEFAULT_UNIVERSE = Path("Data/shared/universe/shared_universe.csv")
DEFAULT_OUT_DIR = Path("UI/swing_audit/risk_profile_sweep_20260607")

GROWTH_THEME_RE = re.compile(
    r"(ai|semis|compute|software|cloud|cyber|quantum|crypto|fintech|space|robot|ev|solar|battery|"
    r"biotech|genomics|data|digital|networking|optical)",
    re.I,
)


def _metrics(df: pd.DataFrame, name: str, baseline_n: int) -> dict:
    pnl = pd.to_numeric(df["pnl_pct"], errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else math.inf
    sharpe = pnl.mean() / pnl.std() * math.sqrt(252) if len(pnl) > 1 and pnl.std() > 0 else 0.0
    ordered = df.assign(_pnl=pd.to_numeric(df["pnl_pct"], errors="coerce")).sort_values("exit_time")
    curve = ordered["_pnl"].fillna(0.0).cumsum() * 100.0
    dd = curve - curve.cummax()
    return {
        "policy": name,
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


def _load_profiles(profile_path: Path, universe_path: Path) -> pd.DataFrame:
    frames = []
    if profile_path.exists():
        p = pd.read_csv(profile_path)
        p["ticker"] = p["ticker"].astype(str).str.upper()
        frames.append(p[[c for c in ["ticker", "sector", "sectorKey", "industry", "industryKey"] if c in p.columns]])
    if universe_path.exists():
        u = pd.read_csv(universe_path)
        u["ticker"] = u["ticker"].astype(str).str.upper()
        frames.append(u[[c for c in ["ticker", "market_cap_bucket", "asset_type", "theme_1", "theme_2", "theme_3"] if c in u.columns]])
    if not frames:
        return pd.DataFrame(columns=["ticker"])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="ticker", how="outer", suffixes=("", "_universe"))
    return out.drop_duplicates("ticker")


def _enrich_trades(trades: pd.DataFrame, theme_map: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    trades = trades.copy()
    trades["ticker"] = trades["ticker"].astype(str).str.upper()
    for ticker, group in trades.groupby("ticker", sort=False):
        feats = _load_ticker_features(ticker)
        if feats is None or feats.empty:
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
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(theme_map, on="ticker", how="left", suffixes=("", "_theme"))
    out = out.merge(profiles, on="ticker", how="left", suffixes=("", "_profile"))
    for col in ("theme_1", "theme_2", "theme_3"):
        alt = f"{col}_profile"
        if col not in out.columns:
            out[col] = ""
        if alt in out.columns:
            out[col] = out[col].fillna(out[alt])
        out[col] = out[col].fillna("").astype(str).str.lower()
    theme_text = out[["theme_1", "theme_2", "theme_3"]].agg(" ".join, axis=1)
    out["is_growth_theme"] = out.get("is_growth_theme", False)
    out["is_growth_theme"] = out["is_growth_theme"].fillna(theme_text.str.contains(GROWTH_THEME_RE)).astype(bool)
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
    out["is_growth_or_high_beta"] = out["is_growth_theme"] | out["is_high_beta"]
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
    out["signal_date"] = pd.to_datetime(out["signal_time"], utc=True).dt.tz_convert("America/New_York").dt.date
    return out.sort_values(["signal_time", "exit_time"]).reset_index(drop=True)


def _merge_theme_state(df: pd.DataFrame, theme_daily_path: Path) -> pd.DataFrame:
    if not theme_daily_path.exists():
        df["theme_return_5d_prev"] = 0.0
        df["theme_above_20d_pct_prev"] = 0.5
        return df
    td = pd.read_parquet(theme_daily_path)
    td["theme"] = td["theme"].fillna("").astype(str).str.lower()
    td["date"] = pd.to_datetime(td["date"], errors="coerce")
    td = td.sort_values("date")
    keep = ["date", "theme", "theme_return_5d", "theme_return_10d", "theme_above_20d_pct", "theme_breadth_20ema", "theme_rvol"]
    td = td[[c for c in keep if c in td.columns]].copy()
    rename = {c: f"{c}_prev" for c in td.columns if c not in {"date", "theme"}}
    td = td.rename(columns=rename)
    out = df.copy()
    out["signal_date_ts"] = pd.to_datetime(out["signal_date"])
    pieces = []
    for theme, group in out.groupby("theme_1", sort=False):
        theme_hist = td[td["theme"].eq(str(theme).lower())].sort_values("date")
        if theme_hist.empty:
            g = group.copy()
            for col in rename.values():
                g[col] = float("nan")
            pieces.append(g)
            continue
        merged = pd.merge_asof(
            group.sort_values("signal_date_ts"),
            theme_hist,
            left_on="signal_date_ts",
            right_on="date",
            direction="backward",
            allow_exact_matches=False,
        ).drop(columns=["date", "theme"], errors="ignore")
        pieces.append(merged)
    out = pd.concat(pieces, ignore_index=True)
    for col, default in {
        "theme_return_5d_prev": 0.0,
        "theme_return_10d_prev": 0.0,
        "theme_above_20d_pct_prev": 0.5,
        "theme_breadth_20ema_prev": 0.5,
        "theme_rvol_prev": 1.0,
    }.items():
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    return out.sort_values(["signal_time", "exit_time"]).reset_index(drop=True)


def _profile_for_row(row: pd.Series, soft: float, hard: float, theme_weak: float) -> str:
    qqq = float(row.get("qqq_ret_16", 0.0) or 0.0)
    theme5 = float(row.get("theme_return_5d_prev", 0.0) or 0.0)
    breadth = float(row.get("theme_above_20d_pct_prev", 0.5) or 0.5)
    if qqq <= hard or (theme5 <= theme_weak and breadth < 0.45):
        return "defensive"
    if qqq <= soft or theme5 <= theme_weak:
        return "balanced"
    return "aggressive"


def _permitted(row: pd.Series, profile: str, late_short_floor: float, require_rs_short: bool, use_sr_room: bool) -> bool:
    direction = int(row["direction"])
    long = direction == 1
    short = direction == -1
    qqq = float(row.get("qqq_ret_16", 0.0) or 0.0)
    growth = bool(row.get("is_growth_or_high_beta", False))
    defensive = bool(row.get("is_defensive_sector", False) or row.get("is_healthcare", False))
    rs_long = bool(row.get("rs_qqq_positive", False) or row.get("rs_spy_positive", False))
    rs_short = not bool(row.get("rs_qqq_positive", False))
    top_chase = bool(row.get("near_top_extended", False)) and not bool(row.get("breakout_64", False))
    bottom_chase = bool(row.get("near_bottom_extended", False)) and not bool(row.get("breakdown_64", False))
    if long and top_chase:
        return False
    if short and bottom_chase:
        return False
    if short and qqq <= late_short_floor:
        return False
    if use_sr_room:
        if long and not (bool(row.get("breakout_64", False)) or float(row.get("room_to_64_high_pct", 0.0) or 0.0) >= 3.0):
            return False
        if short and not (bool(row.get("breakdown_64", False)) or float(row.get("room_to_64_low_pct", 0.0) or 0.0) >= 3.0):
            return False
    if profile == "aggressive":
        if long:
            return True
        return (not require_rs_short) or rs_short or growth
    if profile == "balanced":
        if long:
            return defensive or (not growth and rs_long)
        return growth and ((not require_rs_short) or rs_short)
    if profile == "defensive":
        if long:
            return defensive and rs_long
        return growth and ((not require_rs_short) or rs_short)
    return True


def _apply_caps(df: pd.DataFrame, max_open: int, max_growth_longs: int, max_net_beta: float) -> pd.DataFrame:
    accepted = []
    active: list[dict] = []
    for idx, row in df.sort_values(["signal_time", "exit_time"]).iterrows():
        now = row["signal_time"]
        active = [a for a in active if a["exit_time"] > now]
        if len(active) >= max_open:
            continue
        if row["direction"] == 1 and bool(row.get("is_growth_or_high_beta", False)):
            if sum(1 for a in active if a["direction"] == 1 and a["growth"]) >= max_growth_longs:
                continue
        beta = float(row["beta_like_spy_64"]) if pd.notna(row.get("beta_like_spy_64")) else 1.0
        net_beta = sum(a["direction"] * a["beta"] for a in active)
        if abs(net_beta + int(row["direction"]) * beta) > max_net_beta:
            continue
        accepted.append(idx)
        active.append(
            {
                "exit_time": row["exit_time"],
                "direction": int(row["direction"]),
                "growth": bool(row.get("is_growth_or_high_beta", False)),
                "beta": beta,
            }
        )
    return df.loc[accepted].copy()


def _run_variant(
    df: pd.DataFrame,
    *,
    soft: float,
    hard: float,
    theme_weak: float,
    late_short_floor: float,
    require_rs_short: bool,
    use_sr_room: bool,
    max_open: int,
    max_growth_longs: int,
    max_net_beta: float,
) -> pd.DataFrame:
    profiles = df.apply(lambda row: _profile_for_row(row, soft, hard, theme_weak), axis=1)
    mask = [
        _permitted(row, profile, late_short_floor, require_rs_short, use_sr_room)
        for profile, (_, row) in zip(profiles, df.iterrows(), strict=False)
    ]
    picked = df[pd.Series(mask, index=df.index)].copy()
    picked["risk_profile"] = profiles.loc[picked.index].to_numpy()
    return _apply_caps(picked, max_open=max_open, max_growth_longs=max_growth_longs, max_net_beta=max_net_beta)


def _fast_variant_mask(
    df: pd.DataFrame,
    *,
    soft: float,
    hard: float,
    theme_weak: float,
    late_short_floor: float,
    require_rs_short: bool,
    use_sr_room: bool,
) -> tuple[pd.Series, pd.Series]:
    qqq = pd.to_numeric(df["qqq_ret_16"], errors="coerce").fillna(0.0)
    theme5 = pd.to_numeric(df["theme_return_5d_prev"], errors="coerce").fillna(0.0)
    breadth = pd.to_numeric(df["theme_above_20d_pct_prev"], errors="coerce").fillna(0.5)
    profile = pd.Series("aggressive", index=df.index)
    profile[(qqq <= soft) | (theme5 <= theme_weak)] = "balanced"
    profile[(qqq <= hard) | ((theme5 <= theme_weak) & (breadth < 0.45))] = "defensive"

    long = df["direction"].eq(1)
    short = df["direction"].eq(-1)
    growth = df["is_growth_or_high_beta"].fillna(False).astype(bool)
    defensive = (df["is_defensive_sector"].fillna(False) | df["is_healthcare"].fillna(False)).astype(bool)
    rs_long = df["rs_qqq_positive"].fillna(False).astype(bool) | df["rs_spy_positive"].fillna(False).astype(bool)
    rs_short = ~df["rs_qqq_positive"].fillna(False).astype(bool)
    top_chase = df["near_top_extended"].fillna(False).astype(bool) & ~df["breakout_64"].fillna(False).astype(bool)
    bottom_chase = df["near_bottom_extended"].fillna(False).astype(bool) & ~df["breakdown_64"].fillna(False).astype(bool)

    mask = pd.Series(False, index=df.index)
    aggr = profile.eq("aggressive")
    bal = profile.eq("balanced")
    deff = profile.eq("defensive")
    short_ok = pd.Series(True, index=df.index) if not require_rs_short else rs_short
    mask |= aggr & long
    mask |= aggr & short & (short_ok | growth)
    mask |= bal & long & (defensive | (~growth & rs_long))
    mask |= bal & short & growth & short_ok
    mask |= deff & long & defensive & rs_long
    mask |= deff & short & growth & short_ok

    mask &= ~(long & top_chase)
    mask &= ~(short & bottom_chase)
    mask &= ~(short & qqq.le(late_short_floor))
    if use_sr_room:
        long_room = df["breakout_64"].fillna(False).astype(bool) | pd.to_numeric(
            df["room_to_64_high_pct"], errors="coerce"
        ).ge(3.0)
        short_room = df["breakdown_64"].fillna(False).astype(bool) | pd.to_numeric(
            df["room_to_64_low_pct"], errors="coerce"
        ).ge(3.0)
        mask &= (long & long_room) | (short & short_room)
    return mask, profile


def main() -> int:
    parser = argparse.ArgumentParser(description="No-lookahead sweep of dynamic risk profile switching for swing trades.")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--raw-30m-dir", type=Path, default=DEFAULT_RAW_30M_DIR)
    parser.add_argument("--theme-daily", type=Path, default=DEFAULT_THEME_DAILY)
    parser.add_argument("--theme-map", type=Path, default=DEFAULT_THEME_MAP)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force-enrich", action="store_true")
    parser.add_argument("--grid", choices=["coarse", "full"], default="coarse")
    parser.add_argument("--skip-caps", action="store_true", help="Only run the fast uncapped switching sweep.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = args.out_dir / "tier12_full_enriched_trades.parquet"
    if enriched_path.exists() and not args.force_enrich:
        df = pd.read_parquet(enriched_path)
    else:
        trades = _load_best_trades(args.sweep_dir, args.raw_30m_dir, ["tier1", "tier2"])
        trades = trades.dropna(subset=["signal_time", "exit_time", "pnl_pct"])
        theme_map = _load_theme_map(args.theme_map)
        profiles = _load_profiles(args.profiles, args.universe)
        df = _enrich_trades(trades, theme_map, profiles)
        df = _merge_theme_state(df, args.theme_daily)
        df.to_parquet(enriched_path, index=False)

    baseline_rows = [
        _metrics(df, "baseline_all", len(df)),
        _metrics(df[df["direction"].eq(1)], "calls_only", len(df)),
        _metrics(df[df["direction"].eq(-1)], "puts_only", len(df)),
    ]
    pd.DataFrame(baseline_rows).to_csv(args.out_dir / "baseline_metrics.csv", index=False)

    rows = []
    variant_frames: dict[str, pd.DataFrame] = {}
    picked_dir = args.out_dir / "top_variant_trades"
    picked_dir.mkdir(exist_ok=True)
    variant_id = 0
    if args.grid == "full":
        soft_values = (0.0, -0.0025, -0.005)
        hard_values = (-0.005, -0.01, -0.015)
        theme_values = (-0.01, -0.02, -0.04)
        late_short_values = (-0.01, -0.02, -0.03, -0.05)
        rs_values = (False, True)
        sr_values = (False, True)
        cap_values = (
            (40, 12, 18.0, "loose"),
            (25, 7, 10.0, "balanced"),
            (15, 4, 6.0, "defensive"),
        )
    else:
        soft_values = (0.0, -0.005)
        hard_values = (-0.01,)
        theme_values = (-0.02,)
        late_short_values = (-0.02, -0.05)
        rs_values = (False, True)
        sr_values = (False, True)
        cap_values = (
            (40, 12, 18.0, "loose"),
            (25, 7, 10.0, "balanced"),
            (15, 4, 6.0, "defensive"),
        )

    uncapped_cap_values = cap_values[:1]
    total_variants = sum(
        1
        for soft in soft_values
        for hard in hard_values
        if hard <= soft
        for _theme_weak in theme_values
        for _late_short_floor in late_short_values
        for _require_rs_short in rs_values
        for _use_sr_room in sr_values
        for _cap in uncapped_cap_values
    )
    print(f"Running {total_variants} {args.grid} risk-profile variants on {len(df):,} trades")

    for soft in soft_values:
        for hard in hard_values:
            if hard > soft:
                continue
            for theme_weak in theme_values:
                for late_short_floor in late_short_values:
                    for require_rs_short in rs_values:
                        for use_sr_room in sr_values:
                            for max_open, max_growth_longs, max_net_beta, cap_name in uncapped_cap_values:
                                variant_id += 1
                                mask, profiles = _fast_variant_mask(
                                    df,
                                    soft=soft,
                                    hard=hard,
                                    theme_weak=theme_weak,
                                    late_short_floor=late_short_floor,
                                    require_rs_short=require_rs_short,
                                    use_sr_room=use_sr_room,
                                )
                                picked = df[mask].copy()
                                picked["risk_profile"] = profiles.loc[picked.index].to_numpy()
                                m = _metrics(picked, f"v{variant_id:04d}", len(df))
                                m.update(
                                    {
                                        "soft_qqq": soft,
                                        "hard_qqq": hard,
                                        "theme_weak_5d": theme_weak,
                                        "late_short_floor": late_short_floor,
                                        "require_rs_short": require_rs_short,
                                        "use_sr_room": use_sr_room,
                                        "max_open": max_open,
                                        "max_growth_longs": max_growth_longs,
                                        "max_net_beta": max_net_beta,
                                        "cap_name": cap_name,
                                        "aggressive_trades": int((picked.get("risk_profile") == "aggressive").sum()) if not picked.empty else 0,
                                        "balanced_trades": int((picked.get("risk_profile") == "balanced").sum()) if not picked.empty else 0,
                                        "defensive_trades": int((picked.get("risk_profile") == "defensive").sum()) if not picked.empty else 0,
                                    }
                                )
                                rows.append(m)
                                variant_frames[m["policy"]] = picked
                                if variant_id % 12 == 0 or variant_id == total_variants:
                                    print(f"  completed {variant_id}/{total_variants}", flush=True)
    results = pd.DataFrame(rows).sort_values(
        ["profit_factor", "sharpe", "total_pnl_pp"],
        ascending=[False, False, False],
    )
    results.to_csv(args.out_dir / "risk_profile_sweep_results.csv", index=False)
    score = results.copy()
    score["risk_score"] = score["total_pnl_pp"] / score["max_dd_pp"].abs().clip(lower=1.0)
    robust = score[(score["trades"] >= 1000) & (score["kept_pct"] >= 0.05)].sort_values(
        ["risk_score", "profit_factor", "total_pnl_pp"], ascending=False
    )
    robust.to_csv(args.out_dir / "risk_profile_sweep_robust_ranked.csv", index=False)
    capped_results = pd.DataFrame()
    if not args.skip_caps:
        cap_rows = []
        for _, row in robust.head(10).iterrows():
            base = variant_frames.get(str(row["policy"]))
            if base is None:
                continue
            for max_open, max_growth_longs, max_net_beta, cap_name in cap_values:
                capped = _apply_caps(base, max_open=max_open, max_growth_longs=max_growth_longs, max_net_beta=max_net_beta)
                m = _metrics(capped, f"{row['policy']}_{cap_name}_cap", len(df))
                m.update(row[[c for c in row.index if c not in m]].to_dict())
                m["cap_name"] = cap_name
                m["max_open"] = max_open
                m["max_growth_longs"] = max_growth_longs
                m["max_net_beta"] = max_net_beta
                cap_rows.append(m)
        capped_results = pd.DataFrame(cap_rows).sort_values(
            ["profit_factor", "sharpe", "total_pnl_pp"], ascending=[False, False, False]
        )
        capped_results.to_csv(args.out_dir / "risk_profile_top_capped_results.csv", index=False)
    print("Baseline:")
    print(pd.DataFrame(baseline_rows).to_string(index=False))
    print("\nTop by PF:")
    print(results.head(15).to_string(index=False))
    print("\nTop robust risk-score:")
    print(robust.head(15).to_string(index=False))
    print("\nTop capped variants:")
    print(capped_results.head(15).to_string(index=False) if not capped_results.empty else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
