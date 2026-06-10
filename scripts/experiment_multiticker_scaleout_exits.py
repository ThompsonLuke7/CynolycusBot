from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.multi_ticker_swing.backtest.sweep_v4 import load_raw_5m
from scripts.analyze_multiticker_entry_timing_experiment import _events_to_frames, _load_events


DEFAULT_AUDITS = [
    Path("UI/swing_audit/swing_session_20260528T120501Z.jsonl"),
    Path("UI/swing_audit/swing_session_20260529T120845Z.jsonl"),
    Path("UI/swing_audit/paper/swing_session_20260601T120304Z.jsonl"),
]
OUT_DIR = Path("Data/analysis/multi_ticker_swing_live/experiments/scaleout_exits")
ET = "America/New_York"
MULT = 100.0


@dataclass(frozen=True)
class LotExit:
    lot: str
    exit_ts: pd.Timestamp
    exit_price: float
    exit_reason: str
    pnl_dollars: float
    ret_pct: float
    hold_minutes: float
    exact_target_hit: bool = False
    near_miss_exit: bool = False
    censored: bool = False


def _num(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _load_closed_and_bars(audits: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    closed_frames: list[pd.DataFrame] = []
    underlying_bar_frames: list[pd.DataFrame] = []
    option_bar_rows: list[dict[str, Any]] = []

    for audit in audits:
        if not audit.exists():
            print(f"missing_audit={audit}")
            continue
        closed, bars, _ = _events_to_frames(audit)
        if not closed.empty:
            closed_frames.append(closed)
        if not bars.empty:
            underlying_bar_frames.append(bars)

        for ts, typ, payload in _load_events(audit):
            if typ != "position_bar_5m":
                continue
            pos = payload.get("position") or {}
            bar = payload.get("bar") or {}
            option_bar_rows.append(
                {
                    "audit_date": audit.stem,
                    "audit_ts": ts,
                    "ticker": payload.get("ticker") or pos.get("ticker"),
                    "option_symbol": pos.get("option_symbol"),
                    "entry_time": pos.get("entry_time"),
                    "underlying_open": bar.get("open"),
                    "underlying_high": bar.get("high"),
                    "underlying_low": bar.get("low"),
                    "underlying_close": bar.get("close"),
                    "volume": bar.get("volume"),
                    "option_last_price": pos.get("option_last_price"),
                    "option_best_price": pos.get("option_best_price"),
                    "bars_held": pos.get("bars_held"),
                }
            )

    closed_all = pd.concat(closed_frames, ignore_index=True) if closed_frames else pd.DataFrame()
    bars_all = pd.concat(underlying_bar_frames, ignore_index=True) if underlying_bar_frames else pd.DataFrame()
    option_bars = pd.DataFrame(option_bar_rows)
    for df in (closed_all, bars_all, option_bars):
        if df.empty:
            continue
        for col in ("entry_time", "closed_ts", "audit_ts", "ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        for col in [
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
            "direction",
            "underlying_open",
            "underlying_high",
            "underlying_low",
            "underlying_close",
            "volume",
            "option_last_price",
            "option_best_price",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return closed_all, bars_all, option_bars


def _fresh_calls(closed: pd.DataFrame) -> pd.DataFrame:
    if closed.empty:
        return closed
    out = closed.copy()
    out = out[
        out["is_fresh"].astype(str).str.lower().eq("true")
        & pd.to_numeric(out["direction"], errors="coerce").eq(1)
    ].copy()
    out = out.dropna(subset=["ticker", "entry_time", "closed_ts", "entry_underlying", "option_entry_price"])
    out = out[pd.to_numeric(out["option_entry_price"], errors="coerce") > 0].copy()
    out["trade_id"] = (
        out["ticker"].astype(str).str.upper()
        + "|"
        + out["option_symbol"].astype(str)
        + "|"
        + out["entry_time"].astype(str)
    )
    return out.sort_values("entry_time").reset_index(drop=True)


def _load_underlying_path(trade: pd.Series, audit_bars: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    ticker = str(trade["ticker"]).upper()
    entry_ts = pd.Timestamp(trade["entry_time"])
    same_day_end = entry_ts.tz_convert(ET).normalize() + pd.Timedelta(hours=15, minutes=55)
    same_day_end = same_day_end.tz_convert("UTC")

    try:
        raw = load_raw_5m(ticker)
    except Exception:
        raw = None
    if raw is not None and not raw.empty:
        raw = raw.copy()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
        path = raw[(raw["timestamp"] >= entry_ts) & (raw["timestamp"] <= same_day_end)].copy()
        if not path.empty:
            path = path.rename(
                columns={
                    "timestamp": "ts",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )
            return path[["ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True), False

    if not audit_bars.empty:
        sub = audit_bars[
            (audit_bars["ticker"].astype(str).str.upper() == ticker)
            & (pd.to_datetime(audit_bars["ts"], utc=True, errors="coerce") >= entry_ts)
        ].copy()
        if not sub.empty:
            sub = sub.rename(columns={"ts": "ts"})
            sub["volume"] = np.nan
            return sub[["ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True), True
    return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"]), True


def _load_option_path(trade: pd.Series, option_bars: pd.DataFrame) -> pd.DataFrame:
    if option_bars.empty:
        return pd.DataFrame(columns=["ts", "price"])
    sym = str(trade["option_symbol"])
    entry_ts = pd.Timestamp(trade["entry_time"])
    close_ts = pd.Timestamp(trade["closed_ts"])
    sub = option_bars[
        (option_bars["option_symbol"].astype(str) == sym)
        & (option_bars["audit_ts"] >= entry_ts)
        & (option_bars["audit_ts"] <= close_ts)
    ].copy()
    rows = []
    for row in sub.sort_values("audit_ts").itertuples(index=False):
        price = _num(getattr(row, "option_last_price", math.nan))
        if math.isfinite(price) and price > 0:
            rows.append({"ts": getattr(row, "audit_ts"), "price": price})
    exit_price = _num(trade.get("option_exit_price"))
    if math.isfinite(exit_price) and exit_price > 0:
        rows.append({"ts": close_ts, "price": exit_price})
    if not rows:
        return pd.DataFrame(columns=["ts", "price"])
    out = pd.DataFrame(rows).drop_duplicates("ts", keep="last").sort_values("ts")
    return out.reset_index(drop=True)


def _fallback_exit_ts(trade: pd.Series) -> pd.Timestamp:
    return pd.Timestamp(trade["closed_ts"])


def _option_exit_fixed_or_near(
    *,
    path: pd.DataFrame,
    entry: float,
    target_ret: float,
    zone_frac: float = 0.925,
    giveback_frac: float = 0.25,
) -> tuple[pd.Timestamp | None, float | None, str, bool, bool]:
    best_ret = -math.inf
    zone_seen = False
    for row in path.itertuples(index=False):
        price = float(row.price)
        ret = price / entry - 1.0
        best_ret = max(best_ret, ret)
        if ret >= target_ret:
            return row.ts, entry * (1.0 + target_ret), f"target_{target_ret:.2f}", True, False
        if ret >= target_ret * zone_frac:
            zone_seen = True
        if zone_seen and best_ret > 0:
            floor_ret = best_ret * (1.0 - giveback_frac)
            if ret <= floor_ret:
                return row.ts, price, f"near_target_{target_ret:.2f}", False, True
    return None, None, "", False, False


def _option_exit_runner(
    *,
    path: pd.DataFrame,
    entry: float,
    arm_ret: float,
    giveback_frac: float,
    stop_ret: float = -0.50,
    no_progress_minutes: int | None = None,
    no_progress_mfe: float | None = None,
) -> tuple[pd.Timestamp | None, float | None, str]:
    if path.empty:
        return None, None, ""
    start = pd.Timestamp(path.iloc[0]["ts"])
    best = entry
    armed = False
    checked_np = False
    for row in path.itertuples(index=False):
        ts = pd.Timestamp(row.ts)
        price = float(row.price)
        best = max(best, price)
        ret = price / entry - 1.0
        mfe = best / entry - 1.0
        held = (ts - start).total_seconds() / 60.0
        if ret <= stop_ret:
            return ts, price, f"stop_{abs(stop_ret):.2f}"
        if (
            no_progress_minutes is not None
            and no_progress_mfe is not None
            and not checked_np
            and held >= no_progress_minutes
        ):
            checked_np = True
            if mfe < no_progress_mfe:
                return ts, price, f"no_progress_{no_progress_minutes}m_{no_progress_mfe:.2f}"
        if mfe >= arm_ret:
            armed = True
        if armed:
            floor_profit = (best - entry) * (1.0 - giveback_frac)
            if price - entry <= floor_profit:
                return ts, price, f"runner_giveback_{arm_ret:.2f}_{giveback_frac:.2f}"
    return None, None, ""


def _lot_from_option(
    *,
    lot: str,
    trade: pd.Series,
    exit_ts: pd.Timestamp,
    exit_price: float,
    reason: str,
    exact: bool = False,
    near: bool = False,
    censored: bool = False,
) -> LotExit:
    entry = float(trade["option_entry_price"])
    hold = (pd.Timestamp(exit_ts) - pd.Timestamp(trade["entry_time"])).total_seconds() / 60.0
    pnl = (float(exit_price) - entry) * MULT
    ret = float(exit_price) / entry - 1.0
    return LotExit(lot, pd.Timestamp(exit_ts), float(exit_price), reason, pnl, ret * 100.0, hold, exact, near, censored)


def _simulate_option_policy(trade: pd.Series, path: pd.DataFrame, policy: str) -> list[LotExit]:
    entry = float(trade["option_entry_price"])
    actual_ts = _fallback_exit_ts(trade)
    actual_price = _num(trade.get("option_exit_price"))
    if not math.isfinite(actual_price) or actual_price <= 0:
        actual_price = entry
    if path.empty:
        path = pd.DataFrame([{"ts": actual_ts, "price": actual_price}])

    if policy == "current_single_exit":
        return [_lot_from_option(lot="single", trade=trade, exit_ts=actual_ts, exit_price=actual_price, reason="current_single_exit")]
    if policy == "three_lots_same_exit":
        return [
            _lot_from_option(lot=f"lot{i}", trade=trade, exit_ts=actual_ts, exit_price=actual_price, reason="three_lots_same_exit")
            for i in range(1, 4)
        ]

    if policy == "fixed_scale_50_150_runner":
        targets = [0.50, 1.50]
        runner = (0.75, 0.35, 90, 0.15)
    elif policy == "fixed_scale_100_200_runner":
        targets = [1.00, 2.00]
        runner = (1.00, 0.35, 90, 0.15)
    elif policy == "cost_recovery_200_dynamic_runner":
        targets = [2.00]
        runner = (0.75, 0.50, 120, 0.25)
    else:
        return []

    exits: list[LotExit] = []
    for idx, target in enumerate(targets, start=1):
        ts, price, reason, exact, near = _option_exit_fixed_or_near(path=path, entry=entry, target_ret=target)
        if ts is None or price is None:
            ts, price, reason = actual_ts, actual_price, f"fallback_actual_target_{target:.2f}"
        exits.append(_lot_from_option(lot=f"lot{idx}", trade=trade, exit_ts=ts, exit_price=price, reason=reason, exact=exact, near=near))

    arm, giveback, np_min, np_mfe = runner
    while len(exits) < 3:
        ts, price, reason = _option_exit_runner(
            path=path,
            entry=entry,
            arm_ret=arm,
            giveback_frac=giveback,
            no_progress_minutes=np_min,
            no_progress_mfe=np_mfe,
        )
        if ts is None or price is None:
            ts, price, reason = actual_ts, actual_price, "fallback_actual_runner_censored"
            censored = True
        else:
            censored = False
        exits.append(_lot_from_option(lot=f"lot{len(exits)+1}", trade=trade, exit_ts=ts, exit_price=price, reason=reason, censored=censored))
    return exits


def _prepare_underlying_path(path: pd.DataFrame, entry_ts: pd.Timestamp) -> pd.DataFrame:
    if path.empty:
        return path
    out = path.copy().sort_values("ts").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["ema9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    vol = out["volume"].fillna(0.0)
    out["vwap"] = (out["close"] * vol).cumsum() / vol.replace(0, np.nan).cumsum()
    out["vwap"] = out["vwap"].fillna(out["close"].expanding().mean())
    out["recent_support"] = out["low"].rolling(5, min_periods=1).min().shift(1)
    out["held_minutes"] = (pd.to_datetime(out["ts"], utc=True) - entry_ts).dt.total_seconds() / 60.0
    return out


def _underlying_lot(
    *,
    lot: str,
    trade: pd.Series,
    exit_ts: pd.Timestamp,
    exit_price: float,
    reason: str,
    censored: bool = False,
) -> LotExit:
    entry = float(trade["entry_underlying"])
    hold = (pd.Timestamp(exit_ts) - pd.Timestamp(trade["entry_time"])).total_seconds() / 60.0
    ret = exit_price / entry - 1.0
    pnl = ret * 100.0 * entry
    return LotExit(lot, pd.Timestamp(exit_ts), float(exit_price), reason, pnl, ret * 100.0, hold, censored=censored)


def _exit_at_underlying_target(path: pd.DataFrame, entry: float, target_price: float) -> tuple[pd.Timestamp | None, float | None, str]:
    for row in path.itertuples(index=False):
        if float(row.high) >= target_price:
            return row.ts, target_price, "atr_target"
    return None, None, ""


def _exit_underlying_runner(
    path: pd.DataFrame,
    *,
    entry: float,
    atr: float,
    arm_atr: float,
    trail_atr: float | None = None,
    giveback_frac: float | None = None,
    no_progress_minutes: int | None = None,
    no_progress_atr: float | None = None,
    use_ema_vwap: bool = False,
    use_support: bool = False,
    use_breakout_failure: bool = False,
) -> tuple[pd.Timestamp | None, float | None, str]:
    best = entry
    armed = False
    checked_np = False
    for row in path.itertuples(index=False):
        high = float(row.high)
        close = float(row.close)
        low = float(row.low)
        ts = row.ts
        best = max(best, high)
        mfe_atr = (best - entry) / atr if atr > 0 else 0.0
        if no_progress_minutes is not None and no_progress_atr is not None and not checked_np and float(row.held_minutes) >= no_progress_minutes:
            checked_np = True
            if mfe_atr < no_progress_atr:
                return ts, close, f"no_progress_{no_progress_minutes}m_{no_progress_atr:.2f}atr"
        if mfe_atr >= arm_atr:
            armed = True
        if not armed:
            continue
        if use_breakout_failure and best >= entry + 0.50 * atr and close < entry:
            return ts, close, "breakout_failure"
        if use_ema_vwap and close < min(float(row.ema9), float(row.ema20), float(row.vwap)):
            return ts, close, "ema_vwap_loss"
        if use_support and math.isfinite(float(row.recent_support)) and close < float(row.recent_support):
            return ts, close, "support_loss"
        if trail_atr is not None:
            stop = best - trail_atr * atr
            if low <= stop:
                return ts, stop, f"atr_trail_{trail_atr:.2f}"
        if giveback_frac is not None:
            floor = entry + (best - entry) * (1.0 - giveback_frac)
            if close <= floor:
                return ts, close, f"mfe_giveback_{giveback_frac:.2f}"
    return None, None, ""


def _simulate_underlying_policy(trade: pd.Series, path: pd.DataFrame, policy: str) -> list[LotExit]:
    entry = float(trade["entry_underlying"])
    atr = _num(trade.get("atr_at_entry"))
    if not math.isfinite(atr) or atr <= 0:
        atr = max(entry * 0.01, 1e-6)
    actual_ts = _fallback_exit_ts(trade)
    actual_price = _num(trade.get("exit_underlying"), entry)
    path = _prepare_underlying_path(path, pd.Timestamp(trade["entry_time"]))
    if path.empty:
        path = pd.DataFrame([{"ts": actual_ts, "high": actual_price, "low": actual_price, "close": actual_price, "held_minutes": 0, "ema9": actual_price, "ema20": actual_price, "vwap": actual_price, "recent_support": actual_price}])

    if policy == "current_single_exit":
        return [_underlying_lot(lot="single", trade=trade, exit_ts=actual_ts, exit_price=actual_price, reason="current_single_exit")]
    if policy == "three_lots_same_exit":
        return [
            _underlying_lot(lot=f"lot{i}", trade=trade, exit_ts=actual_ts, exit_price=actual_price, reason="three_lots_same_exit")
            for i in range(1, 4)
        ]

    exits: list[LotExit] = []
    if policy == "atr_ladder_runner":
        targets = [0.75, 1.50]
        runner_kwargs = {"arm_atr": 1.0, "trail_atr": 1.6, "no_progress_minutes": 90, "no_progress_atr": 0.35}
    elif policy == "structure_resistance_runner":
        pre_high = max(_num(trade.get("entry_underlying")), path["high"].head(3).max())
        targets = [(pre_high - entry) / atr if pre_high > entry else 0.75, 1.50]
        runner_kwargs = {"arm_atr": 1.0, "trail_atr": 2.2, "use_support": True, "use_breakout_failure": True}
    elif policy == "ema_vwap_runner":
        targets = [0.75, 1.25]
        runner_kwargs = {"arm_atr": 0.75, "giveback_frac": 0.35, "use_ema_vwap": True, "no_progress_minutes": 90, "no_progress_atr": 0.25}
    elif policy == "support_reclaim_runner":
        targets = [0.75, 1.50]
        runner_kwargs = {"arm_atr": 1.0, "trail_atr": 2.4, "use_support": True, "use_breakout_failure": True}
    elif policy == "mfe_giveback_25":
        targets = [0.75, 1.50]
        runner_kwargs = {"arm_atr": 1.0, "giveback_frac": 0.25}
    elif policy == "mfe_giveback_35":
        targets = [0.75, 1.50]
        runner_kwargs = {"arm_atr": 1.0, "giveback_frac": 0.35}
    elif policy == "mfe_giveback_50":
        targets = [0.75, 1.50]
        runner_kwargs = {"arm_atr": 1.0, "giveback_frac": 0.50}
    elif policy == "no_progress_dynamic":
        targets = [0.75, 1.25]
        runner_kwargs = {"arm_atr": 1.0, "trail_atr": 2.0, "no_progress_minutes": 60, "no_progress_atr": 0.25}
    else:
        return []

    for idx, target_atr in enumerate(targets, start=1):
        target_price = entry + max(0.25, target_atr) * atr
        ts, price, reason = _exit_at_underlying_target(path, entry, target_price)
        if ts is None or price is None:
            ts, price, reason = actual_ts, actual_price, f"fallback_actual_atr_target_{target_atr:.2f}"
        exits.append(_underlying_lot(lot=f"lot{idx}", trade=trade, exit_ts=ts, exit_price=price, reason=reason))

    ts, price, reason = _exit_underlying_runner(path, entry=entry, atr=atr, **runner_kwargs)
    if ts is None or price is None:
        ts, price, reason = path.iloc[-1]["ts"], float(path.iloc[-1]["close"]), "time_cutoff"
        censored = False
    else:
        censored = False
    exits.append(_underlying_lot(lot="lot3", trade=trade, exit_ts=ts, exit_price=price, reason=reason, censored=censored))
    return exits


OPTION_POLICIES = [
    "current_single_exit",
    "three_lots_same_exit",
    "fixed_scale_50_150_runner",
    "fixed_scale_100_200_runner",
    "cost_recovery_200_dynamic_runner",
]

UNDERLYING_POLICIES = [
    "current_single_exit",
    "three_lots_same_exit",
    "atr_ladder_runner",
    "structure_resistance_runner",
    "ema_vwap_runner",
    "support_reclaim_runner",
    "mfe_giveback_25",
    "mfe_giveback_35",
    "mfe_giveback_50",
    "no_progress_dynamic",
]


def _simulate_all(trades: pd.DataFrame, audit_bars: pd.DataFrame, option_bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lot_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    for trade in trades.to_dict("records"):
        s = pd.Series(trade)
        option_path = _load_option_path(s, option_bars)
        underlying_path, underlying_censored = _load_underlying_path(s, audit_bars)
        contexts = [
            ("option_mark", OPTION_POLICIES, option_path, underlying_censored),
            ("underlying_proxy", UNDERLYING_POLICIES, underlying_path, underlying_censored),
        ]
        for replay_mode, policies, path, path_censored in contexts:
            for policy in policies:
                exits = (
                    _simulate_option_policy(s, path, policy)
                    if replay_mode == "option_mark"
                    else _simulate_underlying_policy(s, path, policy)
                )
                if not exits:
                    continue
                for ex in exits:
                    lot_rows.append(
                        {
                            "trade_id": s["trade_id"],
                            "ticker": s["ticker"],
                            "option_symbol": s["option_symbol"],
                            "replay_mode": replay_mode,
                            "policy": policy,
                            "lot": ex.lot,
                            "entry_time": s["entry_time"],
                            "exit_ts": ex.exit_ts,
                            "exit_price": ex.exit_price,
                            "exit_reason": ex.exit_reason,
                            "pnl_dollars": ex.pnl_dollars,
                            "ret_pct": ex.ret_pct,
                            "hold_minutes": ex.hold_minutes,
                            "exact_target_hit": ex.exact_target_hit,
                            "near_miss_exit": ex.near_miss_exit,
                            "censored": bool(ex.censored or path_censored),
                            "entry_option_price": s["option_entry_price"],
                            "entry_underlying": s["entry_underlying"],
                            "actual_option_pnl": s.get("option_pnl_dollars"),
                            "actual_underlying_ret_pct": s.get("underlying_signed_ret_pct"),
                        }
                    )
                total = sum(ex.pnl_dollars for ex in exits)
                entry_cost = (
                    float(s["option_entry_price"]) * MULT * (3 if policy != "current_single_exit" else 1)
                    if replay_mode == "option_mark"
                    else float(s["entry_underlying"]) * (3 if policy != "current_single_exit" else 1)
                )
                lot1 = exits[0]
                current_same_lot = _num(s.get("option_pnl_dollars")) if replay_mode == "option_mark" else _num(s.get("stock_100sh_pnl"))
                trade_rows.append(
                    {
                        "trade_id": s["trade_id"],
                        "ticker": s["ticker"],
                        "option_symbol": s["option_symbol"],
                        "replay_mode": replay_mode,
                        "policy": policy,
                        "lots": len(exits),
                        "entry_time": s["entry_time"],
                        "last_exit_ts": max(ex.exit_ts for ex in exits),
                        "total_pnl": total,
                        "return_on_entry_cost_pct": (total / entry_cost * 100.0) if entry_cost > 0 else np.nan,
                        "avg_lot_hold_minutes": np.mean([ex.hold_minutes for ex in exits]),
                        "win": total > 0,
                        "lot1_pnl": lot1.pnl_dollars,
                        "lot1_ret_pct": lot1.ret_pct,
                        "lot1_recovers_one_contract_cost": lot1.pnl_dollars >= float(s["option_entry_price"]) * MULT
                        if replay_mode == "option_mark"
                        else np.nan,
                        "lot1_recovers_full_3_contract_cost": lot1.pnl_dollars >= float(s["option_entry_price"]) * MULT * 3.0
                        if replay_mode == "option_mark"
                        else np.nan,
                        "runner_pnl": exits[-1].pnl_dollars,
                        "runner_pnl_minus_current_lot": exits[-1].pnl_dollars - current_same_lot
                        if math.isfinite(current_same_lot)
                        else np.nan,
                        "near_miss_exits": sum(ex.near_miss_exit for ex in exits),
                        "exact_target_hits": sum(ex.exact_target_hit for ex in exits),
                        "censored": any(ex.censored for ex in exits) or path_censored,
                        "partial_bank_before_reversal": (
                            len(exits) > 1
                            and max(ex.exit_price for ex in exits[:-1]) > exits[-1].exit_price
                            and sum(ex.pnl_dollars for ex in exits[:-1]) > 0
                        ),
                    }
                )
    return pd.DataFrame(lot_rows), pd.DataFrame(trade_rows)


def _profit_factor(values: pd.Series) -> float:
    wins = values[values > 0].sum()
    losses = values[values < 0].sum()
    return float(wins / abs(losses)) if losses < 0 else float("inf")


def _max_drawdown(values: pd.Series) -> float:
    curve = values.fillna(0).cumsum()
    peak = curve.cummax()
    dd = curve - peak
    return float(dd.min()) if len(dd) else 0.0


def _rounded_table(df: pd.DataFrame, cols: list[str]) -> str:
    table = df[cols].copy()
    numeric_cols = table.select_dtypes(include=["number"]).columns
    table[numeric_cols] = table[numeric_cols].round(3)
    return table.to_string(index=False)


def _summarize(trade_results: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, policy), group in trade_results.sort_values("last_exit_ts").groupby(["replay_mode", "policy"]):
        total = group["total_pnl"].sum()
        avg = group["total_pnl"].mean()
        median = group["total_pnl"].median()
        rows.append(
            {
                "replay_mode": mode,
                "policy": policy,
                "trades": len(group),
                "total_pnl": total,
                "avg_trade_pnl": avg,
                "median_trade_pnl": median,
                "avg_return_on_entry_cost_pct": group["return_on_entry_cost_pct"].mean(),
                "win_rate": group["win"].mean(),
                "profit_factor": _profit_factor(group["total_pnl"]),
                "max_loss": group["total_pnl"].min(),
                "max_drawdown_proxy": _max_drawdown(group["total_pnl"]),
                "avg_lot_hold_minutes": group["avg_lot_hold_minutes"].mean(),
                "lot1_one_contract_cost_recovery_rate": group["lot1_recovers_one_contract_cost"].mean(),
                "lot1_full_3_contract_cost_recovery_rate": group["lot1_recovers_full_3_contract_cost"].mean(),
                "avg_runner_pnl_minus_current_lot": group["runner_pnl_minus_current_lot"].mean(),
                "near_miss_saves": group["near_miss_exits"].sum(),
                "exact_target_hits": group["exact_target_hits"].sum(),
                "giveback_avoided_trades": group["partial_bank_before_reversal"].sum(),
                "censored_trades": group["censored"].sum(),
                "avg_3_contract_premium": (trades["option_entry_price"] * MULT * 3.0).mean(),
                "median_3_contract_premium": (trades["option_entry_price"] * MULT * 3.0).median(),
                "affordable_under_1k": ((trades["option_entry_price"] * MULT * 3.0) <= 1000).mean(),
                "affordable_under_2k": ((trades["option_entry_price"] * MULT * 3.0) <= 2000).mean(),
                "affordable_under_5k": ((trades["option_entry_price"] * MULT * 3.0) <= 5000).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["replay_mode", "total_pnl"], ascending=[True, False])


def _write_markdown(summary: pd.DataFrame, trade_results: pd.DataFrame, out_dir: Path) -> None:
    lines = ["# Multi-Ticker 3-Lot Scale-Out Exit Experiment", ""]
    for mode in ["option_mark", "underlying_proxy"]:
        sub = summary[summary["replay_mode"] == mode].copy()
        if sub.empty:
            continue
        lines.append(f"## {mode}")
        cols = [
            "policy",
            "trades",
            "total_pnl",
            "avg_trade_pnl",
            "win_rate",
            "profit_factor",
            "max_drawdown_proxy",
            "near_miss_saves",
            "giveback_avoided_trades",
            "censored_trades",
        ]
        lines.append(_rounded_table(sub, cols))
        lines.append("")
    lines.append("## Top / Bottom Trade-Level Results")
    top = trade_results.sort_values("total_pnl", ascending=False).head(10)
    bottom = trade_results.sort_values("total_pnl", ascending=True).head(10)
    cols = ["replay_mode", "policy", "ticker", "option_symbol", "entry_time", "total_pnl", "return_on_entry_cost_pct", "near_miss_exits", "censored"]
    lines.append("### Top")
    lines.append(_rounded_table(top, cols))
    lines.append("")
    lines.append("### Bottom")
    lines.append(_rounded_table(bottom, cols))
    lines.append("")
    lines.append(
        "Note: option-mark replay is censored when option marks are only available through the actual logged exit. "
        "Underlying proxy uses full raw 5m bars when available; if raw files are missing, it falls back to audit-window bars and those paths are also censored."
    )
    (out_dir / "scaleout_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(audits: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    closed, audit_bars, option_bars = _load_closed_and_bars(audits)
    trades = _fresh_calls(closed)
    trades.to_csv(out_dir / "scaleout_input_fresh_calls.csv", index=False)
    if trades.empty:
        raise RuntimeError("No fresh call trades found in audit logs.")
    lot_results, trade_results = _simulate_all(trades, audit_bars, option_bars)
    summary = _summarize(trade_results, trades)
    lot_results.to_csv(out_dir / "scaleout_lot_level_results.csv", index=False)
    trade_results.to_csv(out_dir / "scaleout_trade_level_results.csv", index=False)
    summary.to_csv(out_dir / "scaleout_policy_summary.csv", index=False)
    top_bottom = pd.concat(
        [
            trade_results.sort_values("total_pnl", ascending=False).head(20).assign(rank_bucket="top"),
            trade_results.sort_values("total_pnl", ascending=True).head(20).assign(rank_bucket="bottom"),
        ],
        ignore_index=True,
    )
    top_bottom.to_csv(out_dir / "scaleout_top_bottom_trades.csv", index=False)
    _write_markdown(summary, trade_results, out_dir)
    print(f"fresh_call_trades={len(trades)}")
    print(f"lot_rows={len(lot_results)} trade_policy_rows={len(trade_results)}")
    print(summary.round(3).to_string(index=False))
    print(f"out={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay 3-lot scale-out exits on fresh multi-ticker swing calls.")
    parser.add_argument("audits", nargs="*", type=Path, default=DEFAULT_AUDITS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    run(args.audits, args.out_dir)


if __name__ == "__main__":
    main()
