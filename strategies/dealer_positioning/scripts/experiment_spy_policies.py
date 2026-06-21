from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Policy:
    name: str
    min_target_points: float = 0.0
    stop_points: float = 1.10
    cooldown_minutes: int = 15
    skip_open_minutes: int = 0
    require_correct_side_target: bool = False
    require_stable_walls: int = 0
    max_channel_width: float | None = None
    gamma_bias: bool = False
    require_vwap_alignment: bool = False
    allow_long: bool = True
    allow_short: bool = True
    one_trade_at_a_time: bool = True
    allow_wall_trades: bool = True
    allow_magnet_trades: bool = True


def _load_inputs(levels_path: Path, bars_path: Path) -> pd.DataFrame:
    levels = pd.read_csv(levels_path, parse_dates=["timestamp"]).sort_values("timestamp")
    bars = pd.read_parquet(bars_path).sort_values("timestamp")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    merged = pd.merge_asof(
        bars,
        levels.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("95s"),
        suffixes=("", "_level"),
    )
    merged = merged.dropna(subset=["call_wall", "put_wall", "nearest_magnet"]).reset_index(drop=True)
    session_start = merged["timestamp"].min()
    merged["minutes_from_open"] = (merged["timestamp"] - session_start).dt.total_seconds() / 60.0
    merged["channel_width"] = merged["call_wall"] - merged["put_wall"]
    merged["walls_stable_3"] = (
        (merged["call_wall"].eq(merged["call_wall"].shift(1)))
        & (merged["call_wall"].eq(merged["call_wall"].shift(2)))
        & (merged["put_wall"].eq(merged["put_wall"].shift(1)))
        & (merged["put_wall"].eq(merged["put_wall"].shift(2)))
    )
    merged["walls_stable_5"] = merged["walls_stable_3"] & (
        merged["call_wall"].eq(merged["call_wall"].shift(3))
        & merged["call_wall"].eq(merged["call_wall"].shift(4))
        & merged["put_wall"].eq(merged["put_wall"].shift(3))
        & merged["put_wall"].eq(merged["put_wall"].shift(4))
    )
    return merged


def _target_for(row: pd.Series, direction: str, policy: Policy) -> float | None:
    entry = float(row["close"])
    if not policy.require_correct_side_target:
        target = row.get("nearest_magnet")
        return float(target) if pd.notna(target) else None
    if direction == "long":
        candidates = [row.get("nearest_magnet"), row.get("next_magnet_above"), row.get("call_wall")]
        valid = sorted(float(x) for x in candidates if pd.notna(x) and float(x) >= entry + policy.min_target_points)
        return valid[0] if valid else None
    candidates = [row.get("nearest_magnet"), row.get("next_magnet_below"), row.get("put_wall")]
    valid = sorted(
        (float(x) for x in candidates if pd.notna(x) and float(x) <= entry - policy.min_target_points),
        reverse=True,
    )
    return valid[0] if valid else None


def _passes_context(row: pd.Series, direction: str, policy: Policy) -> bool:
    if direction == "long" and not policy.allow_long:
        return False
    if direction == "short" and not policy.allow_short:
        return False
    if float(row["minutes_from_open"]) < policy.skip_open_minutes:
        return False
    if policy.require_stable_walls >= 5 and not bool(row["walls_stable_5"]):
        return False
    if policy.require_stable_walls >= 3 and not bool(row["walls_stable_3"]):
        return False
    if policy.max_channel_width is not None and float(row["channel_width"]) > policy.max_channel_width:
        return False
    if policy.gamma_bias and pd.notna(row.get("gamma_flip")):
        if direction == "long" and float(row["close"]) < float(row["gamma_flip"]):
            return False
        if direction == "short" and float(row["close"]) > float(row["gamma_flip"]):
            return False
    if policy.require_vwap_alignment and pd.notna(row.get("vwap")):
        if direction == "long" and float(row["close"]) < float(row["vwap"]):
            return False
        if direction == "short" and float(row["close"]) > float(row["vwap"]):
            return False
    return True


def _signals(data: pd.DataFrame, policy: Policy) -> list[dict]:
    out: list[dict] = []
    pending: dict | None = None
    last_signal: dict[str, pd.Timestamp] = {}
    touch_pct = 0.0015

    for i in range(2, len(data)):
        row = data.iloc[i]
        prev = data.iloc[i - 1]
        ts = row["timestamp"]
        emitted: list[dict] = []

        if policy.allow_magnet_trades and pd.notna(row["nearest_magnet"]):
            magnet = float(row["nearest_magnet"])
            if pending is not None:
                held = row["close"] > pending["level"] if pending["direction"] == "long" else row["close"] < pending["level"]
                if held:
                    pending["held"] += 1
                    if pending["held"] >= 2:
                        emitted.append({**pending, "timestamp": ts, "entry": float(row["close"])})
                        pending = None
                else:
                    pending = None
            elif prev["close"] <= magnet < row["close"] and pd.notna(row.get("next_magnet_above")) and float(row.get("air_gap_above_score", 0)) > 2:
                pending = {"signal_type": "bullish_magnet", "direction": "long", "level": magnet, "target": float(row["next_magnet_above"]), "stop": magnet, "held": 1}
            elif prev["close"] >= magnet > row["close"] and pd.notna(row.get("next_magnet_below")) and float(row.get("air_gap_below_score", 0)) > 2:
                pending = {"signal_type": "bearish_magnet", "direction": "short", "level": magnet, "target": float(row["next_magnet_below"]), "stop": magnet, "held": 1}

        if policy.allow_wall_trades:
            call_wall = float(row["call_wall"])
            if prev["close"] < call_wall and row["high"] >= call_wall * (1 - touch_pct):
                momentum_weak = row["close"] <= prev["close"] or row["close"] < row["open"]
                if momentum_weak and _passes_context(row, "short", policy):
                    target = _target_for(row, "short", policy)
                    if target is not None:
                        emitted.append(
                            {
                                "timestamp": ts,
                                "signal_type": "call_wall_rejection",
                                "direction": "short",
                                "entry": float(row["close"]),
                                "level": call_wall,
                                "target": target,
                                "stop": call_wall + policy.stop_points,
                            }
                        )
            put_wall = float(row["put_wall"])
            if prev["close"] > put_wall and row["low"] <= put_wall * (1 + touch_pct):
                selling_weak = row["close"] >= prev["close"] or row["close"] > row["open"]
                if selling_weak and _passes_context(row, "long", policy):
                    target = _target_for(row, "long", policy)
                    if target is not None:
                        emitted.append(
                            {
                                "timestamp": ts,
                                "signal_type": "put_wall_bounce",
                                "direction": "long",
                                "entry": float(row["close"]),
                                "level": put_wall,
                                "target": target,
                                "stop": put_wall - policy.stop_points,
                            }
                        )

        for sig in emitted:
            key = sig["signal_type"]
            last = last_signal.get(key)
            if last is not None and (ts - last).total_seconds() < policy.cooldown_minutes * 60:
                continue
            target_move = abs(float(sig["target"]) - float(sig["entry"]))
            if target_move < policy.min_target_points:
                continue
            last_signal[key] = ts
            out.append(sig)
    return out


def _simulate(data: pd.DataFrame, signals: list[dict]) -> pd.DataFrame:
    trades: list[dict] = []
    blocked_until = pd.Timestamp.min.tz_localize("UTC")
    for sig in signals:
        if bool(sig.get("one_trade_at_a_time", True)) and sig["timestamp"] <= blocked_until:
            continue
        start_idx = data.index[data["timestamp"].eq(sig["timestamp"])]
        if len(start_idx) == 0:
            continue
        entry_idx = int(start_idx[0])
        exit_price = float(data.iloc[-1]["close"])
        exit_time = data.iloc[-1]["timestamp"]
        exit_reason = "eod"
        for j in range(entry_idx + 1, len(data)):
            bar = data.iloc[j]
            if sig["direction"] == "long":
                if float(bar["low"]) <= float(sig["stop"]):
                    exit_price, exit_time, exit_reason = float(sig["stop"]), bar["timestamp"], "stop"
                    break
                if float(bar["high"]) >= float(sig["target"]):
                    exit_price, exit_time, exit_reason = float(sig["target"]), bar["timestamp"], "target"
                    break
            else:
                if float(bar["high"]) >= float(sig["stop"]):
                    exit_price, exit_time, exit_reason = float(sig["stop"]), bar["timestamp"], "stop"
                    break
                if float(bar["low"]) <= float(sig["target"]):
                    exit_price, exit_time, exit_reason = float(sig["target"]), bar["timestamp"], "target"
                    break
        mult = 1 if sig["direction"] == "long" else -1
        if bool(sig.get("one_trade_at_a_time", True)):
            blocked_until = exit_time
        trades.append(
            {
                **sig,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_points": (exit_price - float(sig["entry"])) * mult,
            }
        )
    return pd.DataFrame(trades)


def _policy_set() -> list[Policy]:
    base = [
        Policy("current_like_raw_targets", one_trade_at_a_time=False),
        Policy("current_like_one_at_a_time"),
        Policy("correct_side_target_0p40", min_target_points=0.40, require_correct_side_target=True),
        Policy("correct_side_target_0p75", min_target_points=0.75, require_correct_side_target=True),
        Policy("skip_open_30m_correct_side", min_target_points=0.40, skip_open_minutes=30, require_correct_side_target=True),
        Policy("stable_3_correct_side", min_target_points=0.40, require_correct_side_target=True, require_stable_walls=3),
        Policy("stable_5_correct_side", min_target_points=0.40, require_correct_side_target=True, require_stable_walls=5),
        Policy("tight_channel_correct_side", min_target_points=0.40, require_correct_side_target=True, max_channel_width=5.0),
        Policy("gamma_bias_correct_side", min_target_points=0.40, require_correct_side_target=True, gamma_bias=True),
        Policy("vwap_aligned_correct_side", min_target_points=0.40, require_correct_side_target=True, require_vwap_alignment=True),
        Policy("magnet_only_gap", allow_wall_trades=False, allow_magnet_trades=True, min_target_points=0.40),
        Policy("wall_only_correct_side", allow_magnet_trades=False, min_target_points=0.40, require_correct_side_target=True),
        Policy("put_bounce_only_tight", allow_magnet_trades=False, min_target_points=0.40, require_correct_side_target=True, max_channel_width=5.0, allow_short=False),
        Policy("call_reject_only_tight", allow_magnet_trades=False, min_target_points=0.40, require_correct_side_target=True, max_channel_width=5.0, allow_long=False),
    ]
    grid: list[Policy] = []
    for min_target in (0.40, 0.60, 0.75, 1.00):
        for channel in (3.0, 4.0, 5.0, 6.0, 8.0):
            for stop in (0.70, 0.90, 1.10, 1.40):
                for cooldown in (5, 10, 15, 30):
                    grid.append(
                        Policy(
                            name=f"grid_t{min_target:.2f}_ch{channel:.0f}_st{stop:.2f}_cd{cooldown}",
                            min_target_points=min_target,
                            stop_points=stop,
                            cooldown_minutes=cooldown,
                            require_correct_side_target=True,
                            max_channel_width=channel,
                            allow_magnet_trades=False,
                        )
                    )
    for min_target in (0.40, 0.60, 0.75, 1.00):
        for channel in (3.0, 4.0, 5.0, 6.0, 8.0):
            grid.append(
                Policy(
                    name=f"long_only_t{min_target:.2f}_ch{channel:.0f}",
                    min_target_points=min_target,
                    require_correct_side_target=True,
                    max_channel_width=channel,
                    allow_magnet_trades=False,
                    allow_short=False,
                )
            )
            grid.append(
                Policy(
                    name=f"short_only_t{min_target:.2f}_ch{channel:.0f}",
                    min_target_points=min_target,
                    require_correct_side_target=True,
                    max_channel_width=channel,
                    allow_magnet_trades=False,
                    allow_long=False,
                )
            )
    return base + grid


def run_experiments(levels_path: Path, bars_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _load_inputs(levels_path, bars_path)
    policies = _policy_set()
    all_trades: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for policy in policies:
        sigs = _signals(data, policy)
        for sig in sigs:
            sig["one_trade_at_a_time"] = policy.one_trade_at_a_time
        trades = _simulate(data, sigs)
        if not trades.empty:
            trades.insert(0, "policy", policy.name)
            all_trades.append(trades)
            wins = trades["pnl_points"] > 0
            summaries.append(
                {
                    "policy": policy.name,
                    "trades": int(len(trades)),
                    "win_rate": float(wins.mean()),
                    "pnl_points": float(trades["pnl_points"].sum()),
                    "avg_points": float(trades["pnl_points"].mean()),
                    "median_points": float(trades["pnl_points"].median()),
                    "gross_win_points": float(trades.loc[trades["pnl_points"] > 0, "pnl_points"].sum()),
                    "gross_loss_points": float(trades.loc[trades["pnl_points"] < 0, "pnl_points"].sum()),
                    "profit_factor": float(
                        trades.loc[trades["pnl_points"] > 0, "pnl_points"].sum()
                        / abs(trades.loc[trades["pnl_points"] < 0, "pnl_points"].sum())
                    )
                    if abs(float(trades.loc[trades["pnl_points"] < 0, "pnl_points"].sum())) > 0
                    else 99.0,
                    "targets": int(trades["exit_reason"].eq("target").sum()),
                    "stops": int(trades["exit_reason"].eq("stop").sum()),
                    "eod": int(trades["exit_reason"].eq("eod").sum()),
                    "long_trades": int(trades["direction"].eq("long").sum()),
                    "short_trades": int(trades["direction"].eq("short").sum()),
                    "min_target_points": policy.min_target_points,
                    "stop_points": policy.stop_points,
                    "cooldown_minutes": policy.cooldown_minutes,
                    "max_channel_width": policy.max_channel_width,
                    "require_stable_walls": policy.require_stable_walls,
                    "gamma_bias": policy.gamma_bias,
                    "require_vwap_alignment": policy.require_vwap_alignment,
                    "allow_long": policy.allow_long,
                    "allow_short": policy.allow_short,
                }
            )
        else:
            summaries.append(
                {
                    "policy": policy.name,
                    "trades": 0,
                    "win_rate": 0.0,
                    "pnl_points": 0.0,
                    "avg_points": 0.0,
                    "median_points": 0.0,
                    "gross_win_points": 0.0,
                    "gross_loss_points": 0.0,
                    "profit_factor": 0.0,
                    "targets": 0,
                    "stops": 0,
                    "eod": 0,
                    "long_trades": 0,
                    "short_trades": 0,
                    "min_target_points": policy.min_target_points,
                    "stop_points": policy.stop_points,
                    "cooldown_minutes": policy.cooldown_minutes,
                    "max_channel_width": policy.max_channel_width,
                    "require_stable_walls": policy.require_stable_walls,
                    "gamma_bias": policy.gamma_bias,
                    "require_vwap_alignment": policy.require_vwap_alignment,
                    "allow_long": policy.allow_long,
                    "allow_short": policy.allow_short,
                }
            )
    summary = pd.DataFrame(summaries).sort_values(["pnl_points", "avg_points"], ascending=False)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "SPY_policy_experiment_summary_2026-06-12.csv", index=False)
    trades_df.to_csv(output_dir / "SPY_policy_experiment_trades_2026-06-12.csv", index=False)
    return summary, trades_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment with non-ML SPY dealer-positioning policies.")
    parser.add_argument("--levels", default="Data/dealer_positioning/reports/SPY_dealer_levels_2026-06-12.csv")
    parser.add_argument("--bars", default="Data/raw/spy/spy_intraday_1min_2026_06_12.parquet")
    parser.add_argument("--output-dir", default="Data/dealer_positioning/reports")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    summary, _ = run_experiments(Path(args.levels), Path(args.bars), Path(args.output_dir))
    print(summary.head(int(args.top)).to_string(index=False))


if __name__ == "__main__":
    main()
