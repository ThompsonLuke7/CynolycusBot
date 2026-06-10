from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.spy_intraday.Policy.order_policy import (  # noqa: E402
    PHASE4_SWING_SETUP_BODYCLOSE_BODYCLOSE_V1,
    OptionOrderPolicy,
    OptionOrderPolicyConfig,
)
from strategies.spy_intraday.Policy.replay_option_proxy import ReplayOptionPriceProxy  # noqa: E402
from scripts.analyze_confirmed_entry_trade_quality import (  # noqa: E402
    _as_float,
    _extract_fill,
    _pair_trades,
    _quiet_logger,
    _summary_rows,
)
from scripts.compare_entry_overlay_policies import (  # noqa: E402
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_ONE_MIN,
    DEFAULT_SIGNAL_FRAME,
    _load_one_min,
    _load_signal_frame,
)


DEFAULT_SUMMARY_OUT = DEFAULT_ANALYSIS_DIR / "recent_v_exit_policy_sweep_summary.csv"
DEFAULT_TRADES_OUT = DEFAULT_ANALYSIS_DIR / "recent_v_exit_policy_sweep_trades.csv"
DEFAULT_EVENTS_OUT = DEFAULT_ANALYSIS_DIR / "recent_v_exit_policy_sweep_events.csv"
LIVE_RUNS_DIR = ROOT / "Data/inference/live_runs"


@dataclass(frozen=True)
class Variant:
    name: str
    option_exit_opposite_prob_long: float
    option_exit_opposite_profit_pct: float
    option_exit_no_progress_minutes: int
    option_exit_no_progress_mfe_pct: float
    option_exit_trailing_arm_pct: float
    option_exit_trailing_giveback_pct: float
    option_exit_time_decay_minutes: int
    option_exit_time_decay_progress_pct: float
    opposite_structure_filter: str = "none"


@dataclass
class FastPosition:
    side: str
    entry_ts: pd.Timestamp
    entry_spot: float
    entry_premium: float
    symbol: str
    best_premium: float
    best_seen_at: pd.Timestamp
    setup_ts: pd.Timestamp
    setup_ref: float
    setup_prob: float
    setup_threshold: float


@dataclass
class FastSetup:
    side: str
    ref: float
    setup_ts: pd.Timestamp
    expires_at: pd.Timestamp
    prob: float
    threshold: float


def _parse_float_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_int_grid(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _next_expiry(ts: pd.Timestamp, cutoff_hhmm: str = "13:00") -> pd.Timestamp:
    cutoff_h, cutoff_m = [int(x) for x in cutoff_hhmm.split(":")]
    cutoff = ts.replace(hour=cutoff_h, minute=cutoff_m, second=0, microsecond=0)
    expiry_day = ts.normalize()
    if ts >= cutoff:
        expiry_day = expiry_day + pd.offsets.BDay(1)
    return pd.Timestamp(expiry_day).tz_localize(None)


def _sim_symbol(ts: pd.Timestamp, right: str, strike: float) -> str:
    expiry = _next_expiry(ts)
    return f".SIM_SPY_{expiry.strftime('%y%m%d')}_{right}_{float(strike):.2f}"


def _variant_name(v: Variant) -> str:
    opp = "off" if v.option_exit_opposite_prob_long <= 0.0 else f"{v.option_exit_opposite_prob_long:g}"
    np_tag = (
        "np_off"
        if v.option_exit_no_progress_minutes <= 0 or v.option_exit_no_progress_mfe_pct <= 0.0
        else f"np{v.option_exit_no_progress_minutes}_{v.option_exit_no_progress_mfe_pct:g}"
    )
    return (
        f"opp{opp}_oppfit{v.option_exit_opposite_profit_pct:g}_"
        f"{np_tag}_trail{v.option_exit_trailing_arm_pct:g}_{v.option_exit_trailing_giveback_pct:g}_"
        f"td{v.option_exit_time_decay_minutes}_{v.option_exit_time_decay_progress_pct:g}_"
        f"struct_{v.opposite_structure_filter}"
    )


def _build_variants(args: argparse.Namespace) -> list[Variant]:
    variants: list[Variant] = [
        Variant(
            name="baseline_live",
            option_exit_opposite_prob_long=0.40,
            option_exit_opposite_profit_pct=0.0,
            option_exit_no_progress_minutes=10,
            option_exit_no_progress_mfe_pct=0.05,
            option_exit_trailing_arm_pct=1.0,
            option_exit_trailing_giveback_pct=0.20,
            option_exit_time_decay_minutes=60,
            option_exit_time_decay_progress_pct=0.5,
            opposite_structure_filter="none",
        )
    ]
    for (
        opp_long,
        opp_profit,
        no_progress_minutes,
        no_progress_mfe,
        trail_arm,
        trail_giveback,
        time_decay_minutes,
        time_decay_progress,
        structure_filter,
    ) in itertools.product(
        _parse_float_grid(args.opp_long_grid),
        _parse_float_grid(args.opp_profit_grid),
        _parse_int_grid(args.no_progress_minutes_grid),
        _parse_float_grid(args.no_progress_mfe_grid),
        _parse_float_grid(args.trail_arm_grid),
        _parse_float_grid(args.trail_giveback_grid),
        _parse_int_grid(args.time_decay_minutes_grid),
        _parse_float_grid(args.time_decay_progress_grid),
        [x.strip().lower() for x in str(args.structure_filter_grid).split(",") if x.strip()],
    ):
        v = Variant(
            name="",
            option_exit_opposite_prob_long=float(opp_long),
            option_exit_opposite_profit_pct=float(opp_profit),
            option_exit_no_progress_minutes=int(no_progress_minutes),
            option_exit_no_progress_mfe_pct=float(no_progress_mfe),
            option_exit_trailing_arm_pct=float(trail_arm),
            option_exit_trailing_giveback_pct=float(trail_giveback),
            option_exit_time_decay_minutes=int(time_decay_minutes),
            option_exit_time_decay_progress_pct=float(time_decay_progress),
            opposite_structure_filter=str(structure_filter),
        )
        variants.append(Variant(name=_variant_name(v), **{k: getattr(v, k) for k in v.__dataclass_fields__ if k != "name"}))
    seen: set[str] = set()
    deduped: list[Variant] = []
    for variant in variants:
        key = variant.name
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def _load_live_spy_decision_signal_frame(
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for decision_path in sorted(LIVE_RUNS_DIR.glob("*_live_spy/decision-10m.jsonl")):
        run_name = decision_path.parent.name
        with decision_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = record.get("payload") if isinstance(record, dict) else None
                if not isinstance(payload, dict):
                    continue
                bar = payload.get("bar")
                if not isinstance(bar, dict) or str(bar.get("symbol") or "").upper() != "SPY":
                    continue
                setup_ts = pd.to_datetime(bar.get("timestamp"), utc=True, errors="coerce")
                if pd.isna(setup_ts):
                    continue
                setup_ts = pd.Timestamp(setup_ts).tz_convert("America/New_York")
                available_ts = setup_ts + pd.Timedelta(minutes=10)
                if start is not None and available_ts < start - pd.Timedelta(days=5):
                    continue
                if end is not None and available_ts > end + pd.Timedelta(days=1):
                    continue
                rows.append(
                    {
                        "run": run_name,
                        "timestamp": setup_ts,
                        "available_ts": available_ts,
                        "open": _as_float(bar.get("open")),
                        "high": _as_float(bar.get("high")),
                        "low": _as_float(bar.get("low")),
                        "close": _as_float(bar.get("close")),
                        "ema_fast": float("nan"),
                        "ema_slow": float("nan"),
                        "atr_proxy": float("nan"),
                        "p_enter_long": _as_float(bar.get("p_enter_long")),
                        "p_enter_short": _as_float(bar.get("p_enter_short")),
                        "trend_regime": None,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["available_ts", "run"])
        .drop_duplicates(subset=["available_ts"], keep="last")
        .reset_index(drop=True)
    )


def _load_signals(*, signal_frame: Path, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    historical = (
        _load_signal_frame(signal_frame).sort_values("timestamp").reset_index(drop=True)
        if signal_frame.exists()
        else pd.DataFrame()
    )
    live = _load_live_spy_decision_signal_frame(start=start, end=end)
    all_signals = pd.concat([historical, live], ignore_index=True, sort=False)
    if all_signals.empty:
        raise SystemExit("No signals available from historical frame or live decision logs.")
    signals = all_signals.sort_values("available_ts").drop_duplicates(
        subset=["available_ts"],
        keep="last",
    )
    if start is not None:
        signals = signals[signals["available_ts"] >= start - pd.Timedelta(days=5)].copy()
    if end is not None:
        signals = signals[signals["available_ts"] <= end].copy()
    if signals.empty:
        raise SystemExit("No signal rows after filtering.")
    return signals.sort_values("available_ts").reset_index(drop=True)


def _replay_variant(
    *,
    variant: Variant,
    signals: pd.DataFrame,
    one_min: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proxy = ReplayOptionPriceProxy(
        tz_name="America/New_York",
        expiry_hhmm="15:40",
        iv_floor=0.12,
        iv_ceiling=0.90,
        iv_multiplier=1.50,
        min_dte_minutes=1.0,
    )
    policy = OptionOrderPolicy(
        OptionOrderPolicyConfig(
            underlying="SPY",
            submit_orders=False,
            meta_intrabar_entry_policy=PHASE4_SWING_SETUP_BODYCLOSE_BODYCLOSE_V1,
            meta_intrabar_execution_enabled=True,
            meta_intrabar_breakout_entry_only=True,
            meta_intrabar_setup_max_bars=3,
            meta_intrabar_setup_bar_minutes=10,
            meta_intrabar_max_confirmation_age_minutes=30,
            meta_intrabar_ref_chase_atr=0.50,
            meta_intrabar_long_setup_threshold=0.35,
            meta_intrabar_short_setup_threshold=0.65,
            meta_hard_stop_atr=0.0,
            meta_setup_failure_exit_enabled=True,
            meta_setup_failure_buffer_atr=0.10,
            meta_no_progress_exit_enabled=False,
            option_exit_policy="option_adaptive_trail_v1",
            option_exit_take_profit_pct=0.0,
            option_exit_stop_loss_pct=1.0,
            option_exit_profit_lock_arm_pct=2.0,
            option_exit_profit_lock_floor_pct=0.25,
            option_exit_trailing_arm_pct=variant.option_exit_trailing_arm_pct,
            option_exit_trailing_giveback_pct=variant.option_exit_trailing_giveback_pct,
            option_exit_no_progress_minutes=variant.option_exit_no_progress_minutes,
            option_exit_no_progress_mfe_pct=variant.option_exit_no_progress_mfe_pct,
            option_exit_time_decay_minutes=variant.option_exit_time_decay_minutes,
            option_exit_time_decay_progress_pct=variant.option_exit_time_decay_progress_pct,
            option_exit_opposite_prob=0.40,
            option_exit_opposite_prob_long=variant.option_exit_opposite_prob_long,
            option_exit_opposite_prob_short=0.75,
            option_exit_opposite_profit_pct=variant.option_exit_opposite_profit_pct,
            option_exit_quote_mode="bid",
        )
    )
    policy.set_contract_price_provider(proxy.price)

    idx_1m = 0
    fills: list[dict[str, Any]] = []
    signals = signals.sort_values("available_ts").reset_index(drop=True)
    one_min = one_min.sort_values("timestamp").reset_index(drop=True)
    first_available = pd.Timestamp(signals["available_ts"].iloc[0])
    while idx_1m < len(one_min) and pd.Timestamp(one_min.iloc[idx_1m]["timestamp"]) < first_available:
        bar = one_min.iloc[idx_1m].to_dict()
        proxy.update_bar("SPY", bar)
        policy.prefill_1m_bar(bar=bar)
        idx_1m += 1

    for idx, row in signals.iterrows():
        available_ts = pd.Timestamp(row["available_ts"])
        next_available = (
            pd.Timestamp(signals.iloc[idx + 1]["available_ts"])
            if idx + 1 < len(signals)
            else available_ts + pd.Timedelta(minutes=10)
        )
        bar_payload = {
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "ema_fast": row.get("ema_fast"),
            "ema_slow": row.get("ema_slow"),
            "p_enter_long": row["p_enter_long"],
            "p_enter_short": row["p_enter_short"],
            "thr_enter_long": 0.35,
            "thr_enter_short": 0.65,
            "thr_exit_long": 1.0,
            "thr_exit_short": 1.0,
            "p_exit_long": 0.0,
            "p_exit_short": 0.0,
            "trend_regime": row.get("trend_regime"),
            "signal_atr": row.get("atr_proxy"),
        }
        for wrapper in policy.on_decision(action=0.0, closed_bar=bar_payload, logger=_quiet_logger).get("orders") or []:
            event = _extract_fill(wrapper, ts=available_ts, spot=_as_float(row["close"]), policy=policy)
            if event:
                fills.append(event)

        while idx_1m < len(one_min):
            bar = one_min.iloc[idx_1m].to_dict()
            bar_ts = pd.Timestamp(bar["timestamp"])
            if bar_ts < available_ts:
                proxy.update_bar("SPY", bar)
                policy.prefill_1m_bar(bar=bar)
                idx_1m += 1
                continue
            if bar_ts >= next_available:
                break
            proxy.update_bar("SPY", bar)
            for wrapper in policy.on_1m_bar(bar=bar, logger=_quiet_logger).get("orders") or []:
                event = _extract_fill(wrapper, ts=bar_ts, spot=_as_float(bar.get("close")), policy=policy)
                if event:
                    fills.append(event)
            idx_1m += 1

    events = pd.DataFrame(fills)
    if not events.empty:
        events = events.sort_values("timestamp").reset_index(drop=True)
        events.insert(0, "variant", variant.name)
    trades = _pair_trades(events, one_min) if not events.empty else pd.DataFrame()
    if not trades.empty:
        trades.insert(0, "variant", variant.name)
    summary = _summary_rows(trades)
    summary.insert(0, "variant", variant.name)
    for field in (
        "option_exit_opposite_prob_long",
        "option_exit_opposite_profit_pct",
        "option_exit_no_progress_minutes",
        "option_exit_no_progress_mfe_pct",
        "option_exit_trailing_arm_pct",
        "option_exit_trailing_giveback_pct",
        "option_exit_time_decay_minutes",
        "option_exit_time_decay_progress_pct",
        "opposite_structure_filter",
    ):
        summary.insert(1, field, getattr(variant, field))
    return events, trades, summary


def _fast_close_event(pos: FastPosition, *, ts: pd.Timestamp, spot: float, premium: float, reason: str) -> dict[str, Any]:
    return {
        "symbol": pos.symbol,
        "side": pos.side,
        "entry_time": pos.entry_ts,
        "exit_time": ts,
        "hold_minutes": (ts - pos.entry_ts).total_seconds() / 60.0,
        "entry_spot": pos.entry_spot,
        "exit_spot": spot,
        "entry_premium": pos.entry_premium,
        "exit_premium": premium,
        "return_pct": (premium / pos.entry_premium - 1.0) * 100.0 if pos.entry_premium > 0 else float("nan"),
        "mfe_pct": (pos.best_premium / pos.entry_premium - 1.0) * 100.0 if pos.entry_premium > 0 else float("nan"),
        "setup_ts": pos.setup_ts,
        "setup_ref": pos.setup_ref,
        "setup_prob": pos.setup_prob,
        "setup_threshold": pos.setup_threshold,
        "exit_reason": reason,
        "entry_kind": "swing",
    }


def _fast_replay_variant(*, variant: Variant, signals: pd.DataFrame) -> pd.DataFrame:
    proxy = ReplayOptionPriceProxy(
        tz_name="America/New_York",
        expiry_hhmm="15:40",
        iv_floor=0.12,
        iv_ceiling=0.90,
        iv_multiplier=1.50,
        min_dte_minutes=1.0,
    )
    pending: dict[str, FastSetup | None] = {"long": None, "short": None}
    pos: FastPosition | None = None
    trades: list[dict[str, Any]] = []
    last_day = None
    prev_bar: dict[str, float] | None = None
    rows = signals.sort_values("available_ts").to_dict("records")
    for row in rows:
        ts = pd.Timestamp(row["available_ts"])
        if last_day is not None and ts.date() != last_day:
            pending = {"long": None, "short": None}
        last_day = ts.date()
        bar = {
            "symbol": "SPY",
            "timestamp": ts,
            "open": _as_float(row.get("open")),
            "high": _as_float(row.get("high")),
            "low": _as_float(row.get("low")),
            "close": _as_float(row.get("close")),
        }
        proxy.update_bar("SPY", bar)
        close = _as_float(row.get("close"))
        high = _as_float(row.get("high"))
        low = _as_float(row.get("low"))
        open_ = _as_float(row.get("open"))
        p_long = _as_float(row.get("p_enter_long"))
        p_short = _as_float(row.get("p_enter_short"))
        atr = _as_float(row.get("atr_proxy"))
        if not math.isfinite(atr) or atr <= 0.0:
            atr = max(0.5, high - low if math.isfinite(high - low) else 0.5)

        if pos is not None:
            premium = proxy.price(pos.symbol, mode="bid")
            if math.isfinite(premium) and premium >= 0.0:
                if premium > pos.best_premium:
                    pos.best_premium = float(premium)
                    pos.best_seen_at = ts
                ret = premium / pos.entry_premium - 1.0 if pos.entry_premium > 0 else float("nan")
                best_ret = pos.best_premium / pos.entry_premium - 1.0 if pos.entry_premium > 0 else float("nan")
                minutes_since_entry = (ts - pos.entry_ts).total_seconds() / 60.0
                minutes_since_best = (ts - pos.best_seen_at).total_seconds() / 60.0
                opp_prob = p_short if pos.side == "long" else p_long
                opp_thr = variant.option_exit_opposite_prob_long if pos.side == "long" else 0.75
                structural_ok = _opposite_structure_ok(
                    pos=pos,
                    structure_filter=variant.opposite_structure_filter,
                    open_=open_,
                    high=high,
                    low=low,
                    close=close,
                    prev_bar=prev_bar,
                )
                reason = None
                if (
                    variant.option_exit_trailing_arm_pct > 0.0
                    and best_ret >= variant.option_exit_trailing_arm_pct
                    and (pos.best_premium - premium) / pos.entry_premium >= variant.option_exit_trailing_giveback_pct
                ):
                    reason = "adaptive_trail"
                elif (
                    variant.option_exit_no_progress_minutes > 0
                    and variant.option_exit_no_progress_mfe_pct > 0.0
                    and minutes_since_entry >= variant.option_exit_no_progress_minutes
                    and best_ret < variant.option_exit_no_progress_mfe_pct
                ):
                    reason = "no_progress"
                elif (
                    minutes_since_best >= variant.option_exit_time_decay_minutes
                    and best_ret < variant.option_exit_time_decay_progress_pct
                    and ret <= 0.0
                ):
                    reason = "time_decay"
                elif (
                    opp_thr > 0.0
                    and math.isfinite(opp_prob)
                    and opp_prob >= opp_thr
                    and best_ret >= variant.option_exit_opposite_profit_pct
                    and structural_ok
                ):
                    reason = "opposite_signal"
                elif ts.hour == 15 and ts.minute >= 40 and pos.symbol.split("_")[2] == ts.strftime("%y%m%d"):
                    reason = "same_day_eod"
                if reason:
                    trades.append(_fast_close_event(pos, ts=ts, spot=close, premium=float(premium), reason=reason))
                    pos = None

        if pos is None and ts < ts.replace(hour=15, minute=0, second=0, microsecond=0):
            live_setups = [s for s in pending.values() if s is not None and ts <= s.expires_at]
            triggered: list[FastSetup] = []
            for setup in live_setups:
                if setup.side == "long" and high >= setup.ref and close > open_ and close > setup.ref:
                    triggered.append(setup)
                if setup.side == "short" and low <= setup.ref and close < open_ and close < setup.ref:
                    triggered.append(setup)
            if triggered:
                triggered.sort(key=lambda s: s.prob - s.threshold, reverse=True)
                setup = triggered[0]
                right = "C" if setup.side == "long" else "P"
                strike = round(close + atr) if setup.side == "long" else round(close - atr)
                symbol = _sim_symbol(ts, right, strike)
                premium = proxy.price(symbol, mode="mid")
                if math.isfinite(premium) and premium > 0.0:
                    pos = FastPosition(
                        side=setup.side,
                        entry_ts=ts,
                        entry_spot=float(close),
                        entry_premium=float(premium),
                        symbol=symbol,
                        best_premium=float(premium),
                        best_seen_at=ts,
                        setup_ts=setup.setup_ts,
                        setup_ref=setup.ref,
                        setup_prob=setup.prob,
                        setup_threshold=setup.threshold,
                    )
                    pending[setup.side] = None

        long_valid = math.isfinite(p_long) and p_long >= 0.35
        short_valid = math.isfinite(p_short) and p_short >= 0.65
        if long_valid and not short_valid and math.isfinite(high):
            pending["long"] = FastSetup("long", float(high), ts, ts + pd.Timedelta(minutes=40), float(p_long), 0.35)
        elif not long_valid:
            pending["long"] = None
        if short_valid and not long_valid and math.isfinite(low):
            pending["short"] = FastSetup("short", float(low), ts, ts + pd.Timedelta(minutes=40), float(p_short), 0.65)
        elif not short_valid:
            pending["short"] = None
        prev_bar = {"open": open_, "high": high, "low": low, "close": close}

    if pos is not None:
        last = rows[-1]
        ts = pd.Timestamp(last["available_ts"])
        proxy.update_bar("SPY", {"symbol": "SPY", "timestamp": ts, "close": _as_float(last.get("close")), "open": _as_float(last.get("open")), "high": _as_float(last.get("high")), "low": _as_float(last.get("low"))})
        premium = proxy.price(pos.symbol, mode="bid")
        if math.isfinite(premium):
            trades.append(_fast_close_event(pos, ts=ts, spot=_as_float(last.get("close")), premium=float(premium), reason="window_end"))
    out = pd.DataFrame(trades)
    if not out.empty:
        out.insert(0, "variant", variant.name)
    return out


def _opposite_structure_ok(
    *,
    pos: FastPosition,
    structure_filter: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    prev_bar: dict[str, float] | None,
) -> bool:
    mode = str(structure_filter or "none").strip().lower()
    if mode in {"", "none", "off"}:
        return True

    bearish_candle = math.isfinite(open_) and math.isfinite(close) and close < open_
    bullish_candle = math.isfinite(open_) and math.isfinite(close) and close > open_
    prior_low_break = (
        prev_bar is not None
        and math.isfinite(close)
        and math.isfinite(_as_float(prev_bar.get("low")))
        and close < _as_float(prev_bar.get("low"))
    )
    prior_high_break = (
        prev_bar is not None
        and math.isfinite(close)
        and math.isfinite(_as_float(prev_bar.get("high")))
        and close > _as_float(prev_bar.get("high"))
    )
    entry_break = math.isfinite(close) and close < pos.entry_spot if pos.side == "long" else math.isfinite(close) and close > pos.entry_spot
    ref_break = math.isfinite(close) and close < pos.setup_ref if pos.side == "long" else math.isfinite(close) and close > pos.setup_ref

    if pos.side == "long":
        if mode == "bearish_candle":
            return bearish_candle
        if mode == "prior_low":
            return prior_low_break
        if mode == "entry_break":
            return entry_break
        if mode == "ref_break":
            return ref_break
        if mode == "prior_low_or_entry":
            return bool(prior_low_break or entry_break)
        if mode == "prior_low_and_bearish":
            return bool(prior_low_break and bearish_candle)
        if mode == "entry_and_bearish":
            return bool(entry_break and bearish_candle)
        if mode == "ref_and_bearish":
            return bool(ref_break and bearish_candle)
    else:
        if mode == "bearish_candle":
            return bullish_candle
        if mode == "prior_low":
            return prior_high_break
        if mode == "entry_break":
            return entry_break
        if mode == "ref_break":
            return ref_break
        if mode == "prior_low_or_entry":
            return bool(prior_high_break or entry_break)
        if mode == "prior_low_and_bearish":
            return bool(prior_high_break and bullish_candle)
        if mode == "entry_and_bearish":
            return bool(entry_break and bullish_candle)
        if mode == "ref_and_bearish":
            return bool(ref_break and bullish_candle)
    raise ValueError(f"Unknown opposite structure filter: {structure_filter}")


def _fast_summary_rows(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{"bucket": "all", "trades": 0, "avg_return_pct": float("nan"), "median_return_pct": float("nan"), "win_rate": float("nan"), "avg_hold_minutes": float("nan")}])
    rows: list[dict[str, Any]] = []
    for bucket, frame in [("all", trades)] + list(trades.groupby("side")):
        rows.append(
            {
                "bucket": bucket,
                "trades": int(len(frame)),
                "avg_return_pct": float(pd.to_numeric(frame["return_pct"], errors="coerce").mean()),
                "median_return_pct": float(pd.to_numeric(frame["return_pct"], errors="coerce").median()),
                "win_rate": float((pd.to_numeric(frame["return_pct"], errors="coerce") > 0).mean()),
                "avg_hold_minutes": float(pd.to_numeric(frame["hold_minutes"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _enrich_summary(summary: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    all_rows = summary[summary["bucket"] == "all"].copy()
    if trades.empty:
        all_rows["sum_return_pct"] = 0.0
        all_rows["worst_return_pct"] = float("nan")
        all_rows["best_return_pct"] = float("nan")
        all_rows["long_trades"] = 0
        all_rows["short_trades"] = 0
        return summary.merge(all_rows[["variant", "sum_return_pct", "worst_return_pct", "best_return_pct", "long_trades", "short_trades"]], on="variant", how="left")
    grouped = trades.groupby("variant")
    extra = grouped.agg(
        sum_return_pct=("return_pct", "sum"),
        worst_return_pct=("return_pct", "min"),
        best_return_pct=("return_pct", "max"),
    ).reset_index()
    side_counts = trades.pivot_table(index="variant", columns="side", values="symbol", aggfunc="count", fill_value=0)
    side_counts = side_counts.rename(columns={"long": "long_trades", "short": "short_trades"}).reset_index()
    extra = extra.merge(side_counts, on="variant", how="left")
    for col in ("long_trades", "short_trades"):
        if col not in extra.columns:
            extra[col] = 0
    return summary.merge(extra, on="variant", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep recent SPY V-reversal option exit policy settings.")
    parser.add_argument("--signal-frame", default=str(DEFAULT_SIGNAL_FRAME))
    parser.add_argument("--one-min", default=str(DEFAULT_ONE_MIN))
    parser.add_argument("--start", default="2026-05-01")
    parser.add_argument("--end", default="2026-05-12 16:05")
    parser.add_argument("--opp-long-grid", default="0.0,0.60,0.70,0.80")
    parser.add_argument("--opp-profit-grid", default="0.0,0.25,0.50")
    parser.add_argument("--no-progress-minutes-grid", default="0,10")
    parser.add_argument("--no-progress-mfe-grid", default="0.0,0.05")
    parser.add_argument("--trail-arm-grid", default="1.0")
    parser.add_argument("--trail-giveback-grid", default="0.20,0.35")
    parser.add_argument("--time-decay-minutes-grid", default="60,90")
    parser.add_argument("--time-decay-progress-grid", default="0.5,0.75")
    parser.add_argument(
        "--structure-filter-grid",
        default="none",
        help=(
            "Comma-separated structural gates for opposite exits: none,bearish_candle,prior_low,"
            "entry_break,ref_break,prior_low_or_entry,prior_low_and_bearish,entry_and_bearish,ref_and_bearish."
        ),
    )
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--trades-out", default=str(DEFAULT_TRADES_OUT))
    parser.add_argument("--events-out", default=str(DEFAULT_EVENTS_OUT))
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="America/New_York") if args.start else None
    end = pd.Timestamp(args.end, tz="America/New_York") if args.end else None
    signals = _load_signals(signal_frame=Path(args.signal_frame), start=start, end=end)
    variants = _build_variants(args)

    event_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for idx, variant in enumerate(variants, start=1):
        trades = _fast_replay_variant(variant=variant, signals=signals)
        summary = _fast_summary_rows(trades)
        summary.insert(0, "variant", variant.name)
        for field in (
            "option_exit_opposite_prob_long",
            "option_exit_opposite_profit_pct",
            "option_exit_no_progress_minutes",
            "option_exit_no_progress_mfe_pct",
            "option_exit_trailing_arm_pct",
            "option_exit_trailing_giveback_pct",
            "option_exit_time_decay_minutes",
            "option_exit_time_decay_progress_pct",
            "opposite_structure_filter",
        ):
            summary.insert(1, field, getattr(variant, field))
        if not trades.empty:
            trade_frames.append(trades)
        summary_frames.append(summary)
        all_row = summary[summary["bucket"] == "all"].iloc[0]
        print(
            f"[v-exit-sweep] {idx}/{len(variants)} {variant.name}: "
            f"trades={int(all_row['trades'])} avg={float(all_row['avg_return_pct']):.2f}% "
            f"win={float(all_row['win_rate']):.2f}",
            flush=True,
        )

    summary_df = pd.concat(summary_frames, ignore_index=True)
    events_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary_df = _enrich_summary(summary_df, trades_df)

    summary_out = Path(args.summary_out)
    trades_out = Path(args.trades_out)
    events_out = Path(args.events_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False)
    trades_df.to_csv(trades_out, index=False)
    events_df.to_csv(events_out, index=False)

    ranked = summary_df[summary_df["bucket"] == "all"].sort_values(
        ["sum_return_pct", "avg_return_pct", "win_rate"],
        ascending=[False, False, False],
    )
    print(f"[v-exit-sweep] signals={len(signals):,} bar_source=10m_decision variants={len(variants):,}")
    print(f"[v-exit-sweep] wrote {summary_out}")
    print(f"[v-exit-sweep] wrote {trades_out}")
    print(f"[v-exit-sweep] wrote {events_out}")
    cols = [
        "variant",
        "trades",
        "sum_return_pct",
        "avg_return_pct",
        "median_return_pct",
        "win_rate",
        "avg_hold_minutes",
        "worst_return_pct",
        "best_return_pct",
    ]
    print(ranked[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
