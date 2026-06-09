from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from scripts.analyze_multiticker_swing_20260608_forensics import build_trade_table, parse_audit
from scripts.analyze_multiticker_swing_hold_and_runup import audit_runup_metrics, parse_expiry


ET = ZoneInfo("America/New_York")
DEFAULT_OUT = Path("UI/swing_audit/live_policy_experiments_20260608")


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _audit_date(path: Path) -> str:
    match = re.search(r"(\d{8})", path.stem)
    if not match:
        return path.stem
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"


def _load_audit_trades(audits: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in audits:
        if not path.exists():
            continue
        try:
            parsed = parse_audit(path)
            trades = build_trade_table(parsed)
        except Exception as exc:
            print(f"skip {path}: {exc}")
            continue
        if trades.empty:
            continue
        trades = trades.copy()
        trades["audit_path"] = str(path)
        trades["audit_date"] = _audit_date(path)
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        trades["entry_time_et"] = trades["entry_time"].dt.tz_convert(ET)
        trades["entry_hhmm_et"] = trades["entry_time_et"].dt.strftime("%H:%M")
        trades["entry_weekday"] = trades["entry_time_et"].dt.weekday
        exp = trades["symbol"].map(parse_expiry)
        trades["expiry"] = [None if x is None else x.date().isoformat() for x in exp]
        trades["expiry_weekday"] = [None if x is None else x.weekday() for x in exp]
        trades["dte_entry"] = [
            None if x is None or pd.isna(t) else (x.date() - t.date()).days
            for x, t in zip(exp, trades["entry_time_et"])
        ]
        metrics = audit_runup_metrics(path)
        if not metrics.empty:
            trades = trades.merge(
                metrics,
                on=["ticker", "direction", "symbol"],
                how="left",
                suffixes=("", "_audit_seed"),
            )
        frames.append(trades)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "option_pnl_dollars",
        "option_ret_pct",
        "underlying_ret_pct",
        "p_dir",
        "entry_spread_pct_mid",
        "audit_signed_run_from_day_open_pct",
        "audit_signed_run_30m_pct",
        "audit_signed_run_60m_pct",
        "audit_entry_vs_seen_high_pct",
        "audit_entry_vs_seen_low_pct",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _summarize(frame: pd.DataFrame, name: str, mask: pd.Series) -> dict[str, Any]:
    kept = frame[mask.fillna(False)].copy()
    blocked = frame[~mask.fillna(False)].copy()

    def stats(scope: pd.DataFrame, prefix: str) -> dict[str, Any]:
        if scope.empty:
            return {
                f"{prefix}_trades": 0,
                f"{prefix}_pnl": 0.0,
                f"{prefix}_win_rate": math.nan,
                f"{prefix}_underlying_win_rate": math.nan,
                f"{prefix}_median_option_ret": math.nan,
                f"{prefix}_median_underlying_ret": math.nan,
            }
        return {
            f"{prefix}_trades": int(len(scope)),
            f"{prefix}_pnl": float(scope["option_pnl_dollars"].sum()),
            f"{prefix}_win_rate": float((scope["option_pnl_dollars"] > 0).mean()),
            f"{prefix}_underlying_win_rate": float((scope["underlying_ret_pct"] > 0).mean()),
            f"{prefix}_median_option_ret": float(scope["option_ret_pct"].median()),
            f"{prefix}_median_underlying_ret": float(scope["underlying_ret_pct"].median()),
        }

    return {"filter": name, **stats(kept, "kept"), **stats(blocked, "blocked")}


def _max_drawdown(pnl: pd.Series) -> float:
    curve = pnl.fillna(0.0).cumsum()
    if curve.empty:
        return 0.0
    return float((curve - curve.cummax()).min())


def _group_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(cols, dropna=False)
        .agg(
            trades=("ticker", "count"),
            pnl=("option_pnl_dollars", "sum"),
            win_rate=("option_pnl_dollars", lambda s: (s > 0).mean()),
            underlying_win_rate=("underlying_ret_pct", lambda s: (s > 0).mean()),
            median_option_ret=("option_ret_pct", "median"),
            median_underlying_ret=("underlying_ret_pct", "median"),
            avg_p_dir=("p_dir", "mean"),
            avg_spread=("entry_spread_pct_mid", "mean"),
        )
        .round(4)
        .reset_index()
    )


def _audit_experiments(trades: pd.DataFrame, out_dir: Path) -> None:
    fresh = trades[
        (~trades["restored"].fillna(False))
        & trades["option_pnl_dollars"].notna()
        & trades["option_ret_pct"].notna()
    ].copy()
    fresh = fresh.sort_values("entry_time")
    fresh.to_csv(out_dir / "audit_fresh_option_trades.csv", index=False)

    rows: list[dict[str, Any]] = []
    rows.append(_summarize(fresh, "baseline_fresh_options", pd.Series(True, index=fresh.index)))
    for threshold in [10, 15, 18, 20, 25, 35]:
        rows.append(_summarize(fresh, f"spread_le_{threshold}pct", fresh["entry_spread_pct_mid"] <= threshold))
    for threshold in [0.75, 0.80, 0.85, 0.90, 0.95]:
        rows.append(_summarize(fresh, f"p_dir_ge_{threshold:.2f}", fresh["p_dir"] >= threshold))
    for cutoff in ["13:00", "14:00", "14:30", "15:00", "15:15"]:
        rows.append(_summarize(fresh, f"block_after_{cutoff.replace(':', '_')}_et", fresh["entry_hhmm_et"] < cutoff))
    rows.append(
        _summarize(
            fresh,
            "friday_after_13_no_monday_expiry",
            ~((fresh["entry_weekday"] == 4) & (fresh["entry_hhmm_et"] >= "13:00") & (fresh["expiry_weekday"] == 0)),
        )
    )
    rows.append(
        _summarize(
            fresh,
            "chase_block_near_high_1pct_lowrun_4pct",
            ~(
                (fresh["direction"] == 1)
                & (fresh["audit_entry_vs_seen_high_pct"] >= -1.0)
                & (fresh["audit_entry_vs_seen_low_pct"] >= 4.0)
            ),
        )
    )
    rows.append(
        _summarize(
            fresh,
            "chase_block_near_high_2pct_lowrun_6pct",
            ~(
                (fresh["direction"] == 1)
                & (fresh["audit_entry_vs_seen_high_pct"] >= -2.0)
                & (fresh["audit_entry_vs_seen_low_pct"] >= 6.0)
            ),
        )
    )
    rows.append(
        _summarize(
            fresh,
            "combo_spread18_pdir85_pre1500",
            (fresh["entry_spread_pct_mid"] <= 18.0)
            & (fresh["p_dir"] >= 0.85)
            & (fresh["entry_hhmm_et"] < "15:00"),
        )
    )
    rows.append(
        _summarize(
            fresh,
            "combo_spread18_pdir90_pre1430",
            (fresh["entry_spread_pct_mid"] <= 18.0)
            & (fresh["p_dir"] >= 0.90)
            & (fresh["entry_hhmm_et"] < "14:30"),
        )
    )
    pd.DataFrame(rows).round(4).to_csv(out_dir / "audit_filter_experiment_summary.csv", index=False)

    fresh["p_dir_bin"] = pd.cut(
        fresh["p_dir"],
        [0, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01],
        labels=["<.70", ".70-.75", ".75-.80", ".80-.85", ".85-.90", ".90-.95", ">=.95"],
    )
    fresh["spread_bin"] = pd.cut(
        fresh["entry_spread_pct_mid"],
        [-1, 10, 18, 25, 50, 100, 9999],
        labels=["<=10", "10-18", "18-25", "25-50", "50-100", ">100"],
    )
    _group_summary(fresh, ["p_dir_bin"]).to_csv(out_dir / "audit_p_dir_bin_summary.csv", index=False)
    _group_summary(fresh, ["spread_bin"]).to_csv(out_dir / "audit_spread_bin_summary.csv", index=False)
    _group_summary(fresh, ["audit_date"]).to_csv(out_dir / "audit_day_summary.csv", index=False)
    _group_summary(fresh, ["side"]).to_csv(out_dir / "audit_side_summary.csv", index=False)
    _group_summary(fresh, ["audit_date", "side"]).to_csv(out_dir / "audit_day_side_summary.csv", index=False)

    late_monday = fresh[
        (fresh["entry_weekday"] == 4) & (fresh["entry_hhmm_et"] >= "13:00") & (fresh["expiry_weekday"] == 0)
    ].copy()
    late_monday.to_csv(out_dir / "audit_friday_after_13_monday_expiry_blocked_trades.csv", index=False)


def _load_replay_trades(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        df["source_path"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "signal_time" in out.columns:
        out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce")
        out["signal_date"] = out["signal_time"].dt.date.astype(str)
    return out


def _replay_experiments(out_dir: Path) -> None:
    replay = _load_replay_trades(
        [
            Path("UI/swing_audit/backtest_may_20260607/multiticker_swing_20260501_20260531_replay_trades.csv"),
            Path("UI/swing_audit/backtest_june_current_20260607/multiticker_swing_20260601_20260605_replay_trades.csv"),
        ]
    )
    if not replay.empty:
        replay["side"] = replay["direction"].map({1: "call", -1: "put"}).fillna("unknown")
        replay["win"] = replay["pnl_pct"] > 0
        replay["month"] = replay["signal_time"].dt.strftime("%Y-%m")
        (
            replay.groupby(["month", "tier", "side"], dropna=False)
            .agg(
                trades=("ticker", "count"),
                pnl_pct_sum=("pnl_pct", "sum"),
                avg_pnl_pct=("pnl_pct", "mean"),
                median_pnl_pct=("pnl_pct", "median"),
                win_rate=("win", "mean"),
                avg_holding_bars=("holding_bars", "mean"),
            )
            .round(5)
            .reset_index()
            .to_csv(out_dir / "replay_month_tier_side_summary.csv", index=False)
        )
        (
            replay.groupby(["signal_date"], dropna=False)
            .agg(
                trades=("ticker", "count"),
                pnl_pct_sum=("pnl_pct", "sum"),
                win_rate=("win", "mean"),
            )
            .round(5)
            .reset_index()
            .to_csv(out_dir / "replay_daily_summary.csv", index=False)
        )

    enriched_path = Path("UI/swing_audit/risk_profile_sweep_20260607/tier12_full_enriched_trades.parquet")
    if not enriched_path.exists():
        return
    enriched = pd.read_parquet(enriched_path)
    enriched["signal_time"] = pd.to_datetime(enriched["signal_time"], utc=True, errors="coerce")
    recent = enriched[
        (enriched["signal_time"] >= pd.Timestamp("2026-05-01", tz="UTC"))
        & (enriched["signal_time"] < pd.Timestamp("2026-06-09", tz="UTC"))
    ].copy()
    recent["side"] = recent["direction"].map({1: "call", -1: "put"}).fillna("unknown")
    recent["win"] = recent["pnl_pct"] > 0
    recent["signal_hhmm_et"] = recent["signal_time"].dt.tz_convert(ET).dt.strftime("%H:%M")
    recent["chase_long_near_high_lowrun"] = (
        (recent["direction"] == 1)
        & (recent["dist_20bar_high"].fillna(999) <= 1.0)
        & (recent["dist_20bar_low"].fillna(0) >= 4.0)
    )
    recent["chase_long_near_high_looser"] = (
        (recent["direction"] == 1)
        & (recent["dist_20bar_high"].fillna(999) <= 2.0)
        & (recent["dist_20bar_low"].fillna(0) >= 6.0)
    )
    recent.to_csv(out_dir / "replay_recent_enriched_trades.csv", index=False)

    rows = []
    for name, mask in [
        ("baseline_recent_enriched", pd.Series(True, index=recent.index)),
        ("block_after_14_30_et", recent["signal_hhmm_et"] < "14:30"),
        ("block_after_15_00_et", recent["signal_hhmm_et"] < "15:00"),
        ("block_chase_near_high_1pct_lowrun_4pct", ~recent["chase_long_near_high_lowrun"]),
        ("block_chase_near_high_2pct_lowrun_6pct", ~recent["chase_long_near_high_looser"]),
        ("only_tier1", recent["tier"].astype(str).str.contains("1")),
        ("only_tier2", recent["tier"].astype(str).str.contains("2")),
        ("exclude_high_beta_growth", ~recent["is_growth_or_high_beta"].fillna(False)),
        ("require_qqq_alignment_1h", recent["rs_qqq_positive"].eq(recent["direction"].eq(1))),
    ]:
        kept = recent[mask.fillna(False)].copy()
        blocked = recent[~mask.fillna(False)].copy()
        rows.append(
            {
                "filter": name,
                "kept_trades": len(kept),
                "blocked_trades": len(blocked),
                "kept_pnl_pct_sum": kept["pnl_pct"].sum(),
                "blocked_pnl_pct_sum": blocked["pnl_pct"].sum(),
                "kept_win_rate": (kept["pnl_pct"] > 0).mean() if len(kept) else math.nan,
                "blocked_win_rate": (blocked["pnl_pct"] > 0).mean() if len(blocked) else math.nan,
                "kept_max_dd_pct_sum": _max_drawdown(kept["pnl_pct"]),
            }
        )
    pd.DataFrame(rows).round(5).to_csv(out_dir / "replay_recent_filter_experiment_summary.csv", index=False)

    for cols, name in [
        (["tier", "side"], "replay_recent_tier_side_summary.csv"),
        (["side", "chase_long_near_high_lowrun"], "replay_recent_chase_summary.csv"),
        (["side", "is_growth_or_high_beta"], "replay_recent_growth_high_beta_summary.csv"),
        (["side", "rs_qqq_positive"], "replay_recent_rs_qqq_summary.csv"),
    ]:
        (
            recent.groupby(cols, dropna=False)
            .agg(
                trades=("ticker", "count"),
                pnl_pct_sum=("pnl_pct", "sum"),
                avg_pnl_pct=("pnl_pct", "mean"),
                median_pnl_pct=("pnl_pct", "median"),
                win_rate=("win", "mean"),
            )
            .round(5)
            .reset_index()
            .to_csv(out_dir / name, index=False)
        )


def run(audits: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades = _load_audit_trades(audits)
    if not trades.empty:
        trades.to_csv(out_dir / "audit_all_trades.csv", index=False)
        _audit_experiments(trades, out_dir)
    _replay_experiments(out_dir)
    print(f"wrote {out_dir}")
    for name in [
        "audit_filter_experiment_summary.csv",
        "audit_p_dir_bin_summary.csv",
        "replay_recent_filter_experiment_summary.csv",
        "replay_month_tier_side_summary.csv",
    ]:
        path = out_dir / name
        if path.exists():
            print(f"\n{name}")
            print(pd.read_csv(path).round(4).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("audits", nargs="*", type=Path)
    args = parser.parse_args()
    audits = args.audits
    if not audits:
        audits = sorted(Path("UI/swing_audit").glob("swing_session_202605*.jsonl"))
        audits += sorted(Path("UI/swing_audit/paper").glob("swing_session_202606*.jsonl"))
    run(audits, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
