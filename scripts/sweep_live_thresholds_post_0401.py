from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.spy_intraday.Policy.replay_option_proxy import ReplayOptionPriceProxy


DEFAULT_RUN_ROOT = Path("Data/inference/live_runs")
DEFAULT_ONE_MIN = Path("Data/raw/spy/1m_train_runtime_rth_cache.parquet")
DEFAULT_OUT = Path("Data/inference/spy/10min/meta/post_0401_threshold_sweep_summary.csv")
DEFAULT_EVENTS_OUT = Path("Data/inference/spy/10min/meta/post_0401_threshold_sweep_events.csv")


@dataclass
class PendingSetup:
    side: str
    ref: float
    setup_ts: pd.Timestamp
    start_ts: pd.Timestamp
    expires_at: pd.Timestamp
    prob: float
    threshold: float
    entry_kind: str


@dataclass
class Position:
    side: str
    entry_ts: pd.Timestamp
    entry_spot: float
    entry_premium: float
    symbol: str
    best_premium: float
    best_seen_at: pd.Timestamp
    entry_prob: float
    threshold: float
    entry_kind: str


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _to_et(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).tz_convert("America/New_York")


def _load_decisions(run_root: Path, start: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_ts = _to_et(start)
    for run_idx, run_dir in enumerate(sorted(run_root.glob("*_spy"))):
        path = run_dir / "decision-10m.jsonl"
        if not path.exists():
            continue
        for line_idx, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            rec = json.loads(line)
            payload = rec.get("payload", {})
            bar = payload.get("bar", {}) or {}
            ts = _to_et(payload.get("timestamp") or bar.get("timestamp"))
            if pd.isna(ts) or ts < start_ts:
                continue
            rows.append(
                {
                    "timestamp": ts,
                    "available_ts": ts + pd.Timedelta(minutes=10),
                    "run": run_dir.name,
                    "run_idx": run_idx,
                    "line_idx": line_idx,
                    "open": _as_float(bar.get("open")),
                    "high": _as_float(bar.get("high")),
                    "low": _as_float(bar.get("low")),
                    "close": _as_float(bar.get("close")),
                    "atr": _as_float((payload.get("policy_state") or {}).get("atr")),
                    "p_enter_long": _as_float(bar.get("p_enter_long")),
                    "p_enter_short": _as_float(bar.get("p_enter_short")),
                    "p_neutral": _as_float(bar.get("p_swing_setup_neutral")),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["timestamp", "run_idx", "line_idx"])
    # Keep the newest persisted decision if a live session was restarted.
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return df.reset_index(drop=True)


def _first_finite_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def _load_decisions_from_signal_frame(
    *,
    signal_frame: Path,
    prob_frame: Path | None,
    start: str,
    prob_source: str,
) -> pd.DataFrame:
    frame = pd.read_parquet(signal_frame)
    if not isinstance(frame.index, pd.DatetimeIndex):
        ts_col = next((c for c in ("timestamp", "date", "datetime", "index") if c in frame.columns), None)
        if ts_col is None:
            raise SystemExit(f"Signal frame has no DatetimeIndex or timestamp column: {signal_frame}")
        frame[ts_col] = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")
        frame = frame.dropna(subset=[ts_col]).set_index(ts_col)
    else:
        idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame.loc[pd.notna(idx)].copy()
        frame.index = pd.DatetimeIndex(idx[pd.notna(idx)])
    frame.index = pd.DatetimeIndex(frame.index).tz_convert("America/New_York")

    if prob_frame is not None:
        probs = pd.read_parquet(prob_frame)
        if not isinstance(probs.index, pd.DatetimeIndex):
            ts_col = next((c for c in ("timestamp", "date", "datetime", "index") if c in probs.columns), None)
            if ts_col is None:
                raise SystemExit(f"Probability frame has no DatetimeIndex or timestamp column: {prob_frame}")
            probs[ts_col] = pd.to_datetime(probs[ts_col], utc=True, errors="coerce")
            probs = probs.dropna(subset=[ts_col]).set_index(ts_col)
        else:
            idx = pd.to_datetime(probs.index, utc=True, errors="coerce")
            probs = probs.loc[pd.notna(idx)].copy()
            probs.index = pd.DatetimeIndex(idx[pd.notna(idx)])
        probs.index = pd.DatetimeIndex(probs.index).tz_convert("America/New_York")
        frame = frame.join(probs, how="left", rsuffix="_prob")

    source = str(prob_source or "blend").strip().lower()
    if source == "full":
        long_candidates = ["p_long_full", "p_swing_setup_long", "p_enter_long"]
        short_candidates = ["p_short_full", "p_swing_setup_short", "p_enter_short"]
        neutral_candidates = ["p_neutral_full", "p_swing_setup_neutral", "p_neutral"]
    elif source == "test":
        long_candidates = ["p_long_test", "p_long_full", "p_swing_setup_long", "p_enter_long"]
        short_candidates = ["p_short_test", "p_short_full", "p_swing_setup_short", "p_enter_short"]
        neutral_candidates = ["p_neutral_test", "p_neutral_full", "p_swing_setup_neutral", "p_neutral"]
    elif source == "oof":
        long_candidates = ["p_long_oof_train", "p_long_full", "p_swing_setup_long", "p_enter_long"]
        short_candidates = ["p_short_oof_train", "p_short_full", "p_swing_setup_short", "p_enter_short"]
        neutral_candidates = ["p_neutral_oof_train", "p_neutral_full", "p_swing_setup_neutral", "p_neutral"]
    else:
        long_candidates = ["p_long_oof_train", "p_long_test", "p_long_full", "p_swing_setup_long", "p_enter_long"]
        short_candidates = ["p_short_oof_train", "p_short_test", "p_short_full", "p_swing_setup_short", "p_enter_short"]
        neutral_candidates = ["p_neutral_oof_train", "p_neutral_test", "p_neutral_full", "p_swing_setup_neutral", "p_neutral"]

    out = pd.DataFrame(index=frame.index)
    for col in ("open", "high", "low", "close"):
        if col not in frame.columns:
            raise SystemExit(f"Signal frame missing required OHLC column {col!r}: {signal_frame}")
        out[col] = pd.to_numeric(frame[col], errors="coerce")
    out["timestamp"] = out.index
    out["available_ts"] = out["timestamp"] + pd.Timedelta(minutes=10)
    out["run"] = str(signal_frame)
    out["run_idx"] = 0
    out["line_idx"] = np.arange(len(out), dtype=int)
    out["p_enter_long"] = _first_finite_column(frame, long_candidates)
    out["p_enter_short"] = _first_finite_column(frame, short_candidates)
    out["p_neutral"] = _first_finite_column(frame, neutral_candidates)
    if "atr" in frame.columns:
        out["atr"] = pd.to_numeric(frame["atr"], errors="coerce")
    else:
        tr = pd.concat(
            [
                out["high"] - out["low"],
                (out["high"] - out["close"].shift(1)).abs(),
                (out["low"] - out["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(14, min_periods=1).mean()
    start_ts = _to_et(start)
    out = out[out["timestamp"] >= start_ts].copy()
    out = out.dropna(subset=["open", "high", "low", "close", "p_enter_long", "p_enter_short"])
    return out.reset_index(drop=True)


def _load_one_min(path: Path, start: str, end: pd.Timestamp | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    start_ts = _to_et(start) - pd.Timedelta(days=1)
    mask = df["timestamp"] >= start_ts
    if end is not None:
        mask &= df["timestamp"] <= end + pd.Timedelta(days=1)
    df = df.loc[mask].copy()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _hhmm_to_minutes(text: str) -> int:
    h, m = [int(x) for x in str(text).split(":")]
    return h * 60 + m


def _time_in_window(ts: pd.Timestamp, start_hhmm: str, end_hhmm: str) -> bool:
    minute = int(ts.hour) * 60 + int(ts.minute)
    return bool(_hhmm_to_minutes(start_hhmm) <= minute <= _hhmm_to_minutes(end_hhmm))


def _next_expiry(ts: pd.Timestamp, cutoff_hhmm: str) -> pd.Timestamp:
    cutoff_h, cutoff_m = [int(x) for x in cutoff_hhmm.split(":")]
    cutoff = ts.replace(hour=cutoff_h, minute=cutoff_m, second=0, microsecond=0)
    expiry_day = ts.normalize()
    if ts >= cutoff:
        expiry_day = expiry_day + pd.offsets.BDay(1)
    return pd.Timestamp(expiry_day).tz_localize(None)


def _sim_symbol(root: str, ts: pd.Timestamp, right: str, strike: float, cutoff_hhmm: str) -> str:
    expiry = _next_expiry(ts, cutoff_hhmm)
    return f".SIM_{root}_{expiry.strftime('%y%m%d')}_{right}_{float(strike):.2f}"


def _update_setups(
    row: pd.Series,
    *,
    long_thr: float,
    short_thr: float,
    setup_max_bars: int,
    scalp_enabled: bool,
    scalp_long_thr: float,
    scalp_short_thr: float,
    scalp_setup_max_bars: int,
    scalp_min_signal_range_atr: float,
    scalp_require_reversal_close: bool,
    candidate_enabled: bool,
    candidate_long_thr: float,
    candidate_short_thr: float,
    candidate_opposite_max: float,
    candidate_setup_max_bars: int,
    candidate_min_signal_range_atr: float,
    candidate_long_enabled: bool,
    candidate_short_enabled: bool,
    candidate_start_hhmm: str,
    candidate_end_hhmm: str,
    pending: dict[str, PendingSetup | None],
    above: dict[str, bool],
    peak: dict[str, float],
) -> None:
    p_long = _as_float(row.get("p_enter_long"))
    p_short = _as_float(row.get("p_enter_short"))
    long_above = math.isfinite(p_long) and p_long >= long_thr
    short_above = math.isfinite(p_short) and p_short >= short_thr
    long_margin = p_long - long_thr if long_above else float("-inf")
    short_margin = p_short - short_thr if short_above else float("-inf")
    long_valid = long_above and not (short_above and short_margin > long_margin)
    short_valid = short_above and not (long_above and long_margin > short_margin)

    signal_high = _as_float(row.get("high"))
    signal_low = _as_float(row.get("low"))
    signal_open = _as_float(row.get("open"))
    signal_close = _as_float(row.get("close"))
    signal_atr = _as_float(row.get("atr"))

    def scalp_valid(side: str, prob: float, swing_threshold: float) -> bool:
        if not scalp_enabled:
            return False
        threshold = scalp_long_thr if side == "long" else scalp_short_thr
        if not (math.isfinite(prob) and prob >= threshold):
            return False
        if math.isfinite(swing_threshold) and prob >= swing_threshold:
            return False
        if scalp_min_signal_range_atr > 0.0:
            if not (
                math.isfinite(signal_high)
                and math.isfinite(signal_low)
                and math.isfinite(signal_atr)
                and signal_atr > 0.0
            ):
                return False
            if (signal_high - signal_low) < scalp_min_signal_range_atr * signal_atr:
                return False
        if scalp_require_reversal_close:
            if not (math.isfinite(signal_high) and math.isfinite(signal_low) and math.isfinite(signal_close)):
                return False
            mid = (signal_high + signal_low) / 2.0
            if side == "long" and signal_close < mid:
                return False
            if side == "short" and signal_close > mid:
                return False
            if math.isfinite(signal_open):
                if side == "long" and signal_close <= signal_open:
                    return False
                if side == "short" and signal_close >= signal_open:
                    return False
        return True

    def candidate_valid(side: str, prob: float, opp_prob: float, swing_threshold: float) -> bool:
        if not candidate_enabled:
            return False
        if side == "long" and not candidate_long_enabled:
            return False
        if side == "short" and not candidate_short_enabled:
            return False
        if not _time_in_window(row["timestamp"], candidate_start_hhmm, candidate_end_hhmm):
            return False
        threshold = candidate_long_thr if side == "long" else candidate_short_thr
        if not (math.isfinite(prob) and prob >= threshold):
            return False
        if math.isfinite(swing_threshold) and prob >= swing_threshold:
            return False
        if math.isfinite(opp_prob) and opp_prob > candidate_opposite_max:
            return False
        if candidate_min_signal_range_atr > 0.0:
            if not (
                math.isfinite(signal_high)
                and math.isfinite(signal_low)
                and math.isfinite(signal_atr)
                and signal_atr > 0.0
            ):
                return False
            if (signal_high - signal_low) < candidate_min_signal_range_atr * signal_atr:
                return False
        if not (math.isfinite(signal_open) and math.isfinite(signal_close)):
            return False
        if side == "long":
            return bool(signal_close > signal_open)
        return bool(signal_close < signal_open)

    for side, swing_valid, scalp_side_valid, candidate_side_valid, prob, swing_threshold, scalp_threshold, candidate_threshold, ref in (
        (
            "long",
            long_valid,
            scalp_valid("long", p_long, long_thr),
            candidate_valid("long", p_long, p_short, long_thr),
            p_long,
            long_thr,
            scalp_long_thr,
            candidate_long_thr,
            signal_high,
        ),
        (
            "short",
            short_valid,
            scalp_valid("short", p_short, short_thr),
            candidate_valid("short", p_short, p_long, short_thr),
            p_short,
            short_thr,
            scalp_short_thr,
            candidate_short_thr,
            signal_low,
        ),
    ):
        if swing_valid:
            entry_kind = "swing"
            threshold = swing_threshold
            valid = True
        elif candidate_side_valid:
            entry_kind = "candidate"
            threshold = candidate_threshold
            valid = True
        elif scalp_side_valid:
            entry_kind = "scalp"
            threshold = scalp_threshold
            valid = True
        else:
            entry_kind = ""
            threshold = swing_threshold
            valid = False
        if not valid and not (math.isfinite(prob) and math.isfinite(swing_threshold) and prob >= swing_threshold):
            above[side] = False
            peak[side] = float("nan")
            pending[side] = None
            continue
        refresh = valid and (not above[side] or not math.isfinite(peak[side]) or prob > peak[side])
        if refresh and math.isfinite(ref):
            setup_ts = row["timestamp"]
            if entry_kind == "candidate":
                max_bars = candidate_setup_max_bars
            elif entry_kind == "scalp":
                max_bars = scalp_setup_max_bars
            else:
                max_bars = setup_max_bars
            pending[side] = PendingSetup(
                side=side,
                ref=float(ref),
                setup_ts=setup_ts,
                start_ts=setup_ts + pd.Timedelta(minutes=10),
                expires_at=setup_ts + pd.Timedelta(minutes=10 * (max_bars + 1)),
                prob=float(prob),
                threshold=float(threshold),
                entry_kind=entry_kind,
            )
            peak[side] = float(prob)
        above[side] = bool(valid and prob >= threshold)


def _triggered(setup: PendingSetup, row: pd.Series) -> bool:
    open_ = _as_float(row.get("open"))
    high = _as_float(row.get("high"))
    low = _as_float(row.get("low"))
    close = _as_float(row.get("close"))
    if setup.side == "long":
        return bool(high >= setup.ref and close > open_ and close > setup.ref)
    return bool(low <= setup.ref and close < open_ and close < setup.ref)


def _close_event(pos: Position, ts: pd.Timestamp, spot: float, premium: float, reason: str) -> dict[str, Any]:
    ret = (premium / pos.entry_premium) - 1.0 if pos.entry_premium > 0 else float("nan")
    mfe = (pos.best_premium / pos.entry_premium) - 1.0 if pos.entry_premium > 0 else float("nan")
    return {
        "side": pos.side,
        "entry_ts": pos.entry_ts,
        "exit_ts": ts,
        "entry_spot": pos.entry_spot,
        "exit_spot": spot,
        "entry_premium": pos.entry_premium,
        "exit_premium": premium,
        "peak_premium": pos.best_premium,
        "return": ret,
        "mfe": mfe,
        "reason": reason,
        "entry_prob": pos.entry_prob,
        "threshold": pos.threshold,
        "entry_kind": pos.entry_kind,
        "symbol": pos.symbol,
    }


def _run_one(
    *,
    decisions: pd.DataFrame,
    one_min: pd.DataFrame,
    long_thr: float,
    short_thr: float,
    exit_opp_long_thr: float,
    exit_opp_short_thr: float,
    setup_max_bars: int,
    cutoff_hhmm: str,
    new_entry_cutoff_hhmm: str,
    entry_quote_mode: str,
    exit_quote_mode: str,
    quote_spread_bps: float,
    stop_loss_pct: float,
    no_progress_minutes: int,
    no_progress_mfe_pct: float,
    trail_arm_pct: float,
    trail_giveback_pct: float,
    time_decay_minutes: int,
    time_decay_progress_pct: float,
    scalp_enabled: bool,
    scalp_long_thr: float,
    scalp_short_thr: float,
    scalp_setup_max_bars: int,
    scalp_min_signal_range_atr: float,
    scalp_require_reversal_close: bool,
    candidate_enabled: bool,
    candidate_long_thr: float,
    candidate_short_thr: float,
    candidate_opposite_max: float,
    candidate_setup_max_bars: int,
    candidate_min_signal_range_atr: float,
    candidate_long_enabled: bool,
    candidate_short_enabled: bool,
    candidate_start_hhmm: str,
    candidate_end_hhmm: str,
) -> list[dict[str, Any]]:
    proxy = ReplayOptionPriceProxy(
        tz_name="America/New_York",
        expiry_hhmm="15:40",
        iv_floor=0.12,
        iv_ceiling=0.90,
        iv_multiplier=1.50,
        min_dte_minutes=1.0,
        quote_spread_bps=quote_spread_bps,
    )
    decision_records = decisions.to_dict("records")
    decision_idx = 0
    latest_probs = {"long": float("nan"), "short": float("nan")}
    latest_atr = float("nan")
    pending: dict[str, PendingSetup | None] = {"long": None, "short": None}
    above = {"long": False, "short": False}
    peak = {"long": float("nan"), "short": float("nan")}
    pos: Position | None = None
    events: list[dict[str, Any]] = []
    last_day = None

    for row in one_min.to_dict("records"):
        ts = row["timestamp"]
        if last_day is not None and ts.date() != last_day:
            pending = {"long": None, "short": None}
            above = {"long": False, "short": False}
            peak = {"long": float("nan"), "short": float("nan")}
        last_day = ts.date()

        proxy.update_bar("SPY", row)
        while decision_idx < len(decision_records) and decision_records[decision_idx]["available_ts"] <= ts:
            drow = pd.Series(decision_records[decision_idx])
            latest_probs["long"] = _as_float(drow.get("p_enter_long"))
            latest_probs["short"] = _as_float(drow.get("p_enter_short"))
            latest_atr = _as_float(drow.get("atr"))
            if not math.isfinite(latest_atr) or latest_atr <= 0:
                latest_atr = max(0.5, _as_float(drow.get("high")) - _as_float(drow.get("low")))
            _update_setups(
                drow,
                long_thr=long_thr,
                short_thr=short_thr,
                setup_max_bars=setup_max_bars,
                scalp_enabled=scalp_enabled,
                scalp_long_thr=scalp_long_thr,
                scalp_short_thr=scalp_short_thr,
                scalp_setup_max_bars=scalp_setup_max_bars,
                scalp_min_signal_range_atr=scalp_min_signal_range_atr,
                scalp_require_reversal_close=scalp_require_reversal_close,
                candidate_enabled=candidate_enabled,
                candidate_long_thr=candidate_long_thr,
                candidate_short_thr=candidate_short_thr,
                candidate_opposite_max=candidate_opposite_max,
                candidate_setup_max_bars=candidate_setup_max_bars,
                candidate_min_signal_range_atr=candidate_min_signal_range_atr,
                candidate_long_enabled=candidate_long_enabled,
                candidate_short_enabled=candidate_short_enabled,
                candidate_start_hhmm=candidate_start_hhmm,
                candidate_end_hhmm=candidate_end_hhmm,
                pending=pending,
                above=above,
                peak=peak,
            )
            decision_idx += 1

        close = _as_float(row.get("close"))
        if pos is not None:
            premium = proxy.price(pos.symbol, mode=exit_quote_mode)
            if math.isfinite(premium):
                if premium > pos.best_premium:
                    pos.best_premium = float(premium)
                    pos.best_seen_at = ts
                ret = premium / pos.entry_premium - 1.0
                best_ret = pos.best_premium / pos.entry_premium - 1.0
                minutes_since_best = (ts - pos.best_seen_at).total_seconds() / 60.0
                minutes_since_entry = (ts - pos.entry_ts).total_seconds() / 60.0
                opp_prob = latest_probs["short"] if pos.side == "long" else latest_probs["long"]
                opp_thr = exit_opp_long_thr if pos.side == "long" else exit_opp_short_thr
                reason = None
                if 0.0 < float(stop_loss_pct) < 1.0 and ret <= -float(stop_loss_pct):
                    reason = "stop_loss"
                elif (
                    best_ret >= float(trail_arm_pct)
                    and ((pos.best_premium - premium) / pos.entry_premium) >= float(trail_giveback_pct)
                ):
                    reason = "adaptive_trail"
                elif (
                    no_progress_minutes > 0
                    and no_progress_mfe_pct > 0.0
                    and minutes_since_entry >= float(no_progress_minutes)
                    and best_ret < float(no_progress_mfe_pct)
                ):
                    reason = "no_progress"
                elif (
                    minutes_since_best >= float(time_decay_minutes)
                    and best_ret < float(time_decay_progress_pct)
                    and ret <= 0.0
                ):
                    reason = "time_decay"
                elif best_ret >= 1.0 and math.isfinite(opp_prob) and opp_prob >= opp_thr:
                    reason = "opposite_signal"
                elif ts.hour == 15 and ts.minute >= 40 and pos.symbol.split("_")[2] == ts.strftime("%y%m%d"):
                    reason = "same_day_eod"
                if reason:
                    events.append(_close_event(pos, ts, close, premium, reason))
                    pos = None

        if pos is None:
            if str(new_entry_cutoff_hhmm or "").strip():
                cutoff_h, cutoff_m = [int(x) for x in str(new_entry_cutoff_hhmm).split(":")]
                cutoff_ts = ts.replace(hour=cutoff_h, minute=cutoff_m, second=0, microsecond=0)
                if ts >= cutoff_ts:
                    continue
            live_setups = [
                setup
                for setup in (pending.get("long"), pending.get("short"))
                if setup is not None and setup.start_ts <= ts < setup.expires_at
            ]
            triggered = [setup for setup in live_setups if _triggered(setup, pd.Series(row))]
            if triggered:
                if len(triggered) > 1:
                    triggered.sort(key=lambda s: s.prob - (long_thr if s.side == "long" else short_thr), reverse=True)
                setup = triggered[0]
                right = "C" if setup.side == "long" else "P"
                strike = round(close + latest_atr) if setup.side == "long" else round(close - latest_atr)
                symbol = _sim_symbol("SPY", ts, right, strike, cutoff_hhmm)
                premium = proxy.price(symbol, mode=entry_quote_mode)
                if math.isfinite(premium) and premium > 0:
                    pos = Position(
                        side=setup.side,
                        entry_ts=ts,
                        entry_spot=float(close),
                        entry_premium=float(premium),
                        symbol=symbol,
                        best_premium=float(premium),
                        best_seen_at=ts,
                        entry_prob=setup.prob,
                        threshold=setup.threshold,
                        entry_kind=setup.entry_kind,
                    )
                    pending[setup.side] = None

    if pos is not None and len(one_min):
        last = one_min.iloc[-1]
        premium = proxy.price(pos.symbol, mode=exit_quote_mode)
        if math.isfinite(premium):
            events.append(_close_event(pos, last["timestamp"], _as_float(last["close"]), premium, "window_end"))
    return events


def _metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "trades": 0,
            "avg_return": float("nan"),
            "median_return": float("nan"),
            "win_rate": float("nan"),
            "sum_return": 0.0,
            "p25_return": float("nan"),
            "p75_return": float("nan"),
            "avg_mfe": float("nan"),
            "long_trades": 0,
            "short_trades": 0,
        }
    returns = np.array([float(e["return"]) for e in events], dtype=float)
    mfes = np.array([float(e["mfe"]) for e in events], dtype=float)
    return {
        "trades": int(len(events)),
        "avg_return": float(np.nanmean(returns)),
        "median_return": float(np.nanmedian(returns)),
        "win_rate": float(np.nanmean(returns > 0)),
        "sum_return": float(np.nansum(returns)),
        "p25_return": float(np.nanpercentile(returns, 25)),
        "p75_return": float(np.nanpercentile(returns, 75)),
        "avg_mfe": float(np.nanmean(mfes)),
        "long_trades": int(sum(1 for e in events if e["side"] == "long")),
        "short_trades": int(sum(1 for e in events if e["side"] == "short")),
        "swing_trades": int(sum(1 for e in events if e.get("entry_kind") == "swing")),
        "scalp_trades": int(sum(1 for e in events if e.get("entry_kind") == "scalp")),
        "candidate_trades": int(sum(1 for e in events if e.get("entry_kind") == "candidate")),
    }


def _parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep post-04/01 live setup probability thresholds.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--signal-frame", default="")
    parser.add_argument("--prob-frame", default="")
    parser.add_argument(
        "--prob-source",
        default="blend",
        choices=["blend", "full", "test", "oof"],
        help="Probability columns to prefer when --signal-frame/--prob-frame are used.",
    )
    parser.add_argument("--one-min", default=str(DEFAULT_ONE_MIN))
    parser.add_argument("--start", default="2026-04-01T00:00:00-04:00")
    parser.add_argument("--long-grid", default="0.35")
    parser.add_argument("--short-grid", default="0.65,0.75,0.85")
    parser.add_argument("--setup-max-bars", type=int, default=3)
    parser.add_argument("--setup-max-bars-grid", default="")
    parser.add_argument("--exit-opp-long-grid", default="0.40")
    parser.add_argument("--exit-opp-short-grid", default="0.75")
    parser.add_argument("--entry-quote-mode-grid", default="ask,mid")
    parser.add_argument("--exit-quote-mode-grid", default="bid")
    parser.add_argument("--quote-spread-bps", type=float, default=500.0)
    parser.add_argument("--cutoff-hhmm", default="13:00")
    parser.add_argument("--new-entry-cutoff-hhmm", default="15:00")
    parser.add_argument("--stop-loss-pct-grid", default="1.0,0.55")
    parser.add_argument("--no-progress-minutes-grid", default="0")
    parser.add_argument("--no-progress-mfe-grid", default="0.0")
    parser.add_argument("--trail-arm-grid", default="1.0")
    parser.add_argument("--trail-giveback-grid", default="0.20")
    parser.add_argument("--time-decay-minutes-grid", default="60")
    parser.add_argument("--time-decay-progress-grid", default="0.5")
    parser.add_argument("--scalp-enabled-grid", default="false")
    parser.add_argument("--scalp-long-grid", default="0.30")
    parser.add_argument("--scalp-short-grid", default="0.55")
    parser.add_argument("--scalp-setup-max-bars-grid", default="1")
    parser.add_argument("--scalp-min-signal-range-atr-grid", default="0.35")
    parser.add_argument("--scalp-require-reversal-close-grid", default="true")
    parser.add_argument("--candidate-enabled-grid", default="false")
    parser.add_argument("--candidate-long-grid", default="0.30")
    parser.add_argument("--candidate-short-grid", default="0.55")
    parser.add_argument("--candidate-opposite-max-grid", default="0.15")
    parser.add_argument("--candidate-setup-max-bars-grid", default="1")
    parser.add_argument("--candidate-min-signal-range-atr-grid", default="0.35")
    parser.add_argument("--candidate-long-enabled-grid", default="true")
    parser.add_argument("--candidate-short-enabled-grid", default="true")
    parser.add_argument("--candidate-start-hhmm-grid", default="09:30")
    parser.add_argument("--candidate-end-hhmm-grid", default="16:00")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--events-out", default=str(DEFAULT_EVENTS_OUT))
    args = parser.parse_args()

    if str(args.signal_frame or "").strip():
        decisions = _load_decisions_from_signal_frame(
            signal_frame=Path(args.signal_frame),
            prob_frame=Path(args.prob_frame) if str(args.prob_frame or "").strip() else None,
            start=str(args.start),
            prob_source=str(args.prob_source),
        )
    else:
        decisions = _load_decisions(Path(args.run_root), args.start)
    if decisions.empty:
        raise SystemExit("No decision rows found.")
    end = decisions["timestamp"].max()
    one_min = _load_one_min(Path(args.one_min), args.start, end)
    one_min = one_min[(one_min["timestamp"] >= decisions["timestamp"].min() - pd.Timedelta(minutes=10))]
    if one_min.empty:
        raise SystemExit("No 1m rows found.")

    summaries: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    setup_grid = (
        [int(x.strip()) for x in str(args.setup_max_bars_grid).split(",") if x.strip()]
        if str(args.setup_max_bars_grid or "").strip()
        else [int(args.setup_max_bars)]
    )
    entry_quote_modes = [x.strip().lower() for x in str(args.entry_quote_mode_grid).split(",") if x.strip()]
    exit_quote_modes = [x.strip().lower() for x in str(args.exit_quote_mode_grid).split(",") if x.strip()]
    stop_loss_grid = _parse_grid(args.stop_loss_pct_grid)
    no_progress_minutes_grid = [int(x.strip()) for x in str(args.no_progress_minutes_grid).split(",") if x.strip()]
    no_progress_mfe_grid = _parse_grid(args.no_progress_mfe_grid)
    trail_arm_grid = _parse_grid(args.trail_arm_grid)
    trail_giveback_grid = _parse_grid(args.trail_giveback_grid)
    time_decay_minutes_grid = [
        int(x.strip()) for x in str(args.time_decay_minutes_grid).split(",") if x.strip()
    ]
    time_decay_progress_grid = _parse_grid(args.time_decay_progress_grid)
    scalp_enabled_grid = [
        str(x).strip().lower() in {"1", "true", "yes", "y", "on"}
        for x in str(args.scalp_enabled_grid).split(",")
        if str(x).strip()
    ]
    scalp_setup_max_bars_grid = [
        int(x.strip()) for x in str(args.scalp_setup_max_bars_grid).split(",") if x.strip()
    ]
    scalp_require_reversal_close_grid = [
        str(x).strip().lower() in {"1", "true", "yes", "y", "on"}
        for x in str(args.scalp_require_reversal_close_grid).split(",")
        if str(x).strip()
    ]
    candidate_enabled_grid = [
        str(x).strip().lower() in {"1", "true", "yes", "y", "on"}
        for x in str(args.candidate_enabled_grid).split(",")
        if str(x).strip()
    ]
    candidate_setup_max_bars_grid = [
        int(x.strip()) for x in str(args.candidate_setup_max_bars_grid).split(",") if x.strip()
    ]
    candidate_long_enabled_grid = [
        str(x).strip().lower() in {"1", "true", "yes", "y", "on"}
        for x in str(args.candidate_long_enabled_grid).split(",")
        if str(x).strip()
    ]
    candidate_short_enabled_grid = [
        str(x).strip().lower() in {"1", "true", "yes", "y", "on"}
        for x in str(args.candidate_short_enabled_grid).split(",")
        if str(x).strip()
    ]
    candidate_start_hhmm_grid = [
        x.strip() for x in str(args.candidate_start_hhmm_grid).split(",") if x.strip()
    ]
    candidate_end_hhmm_grid = [
        x.strip() for x in str(args.candidate_end_hhmm_grid).split(",") if x.strip()
    ]
    grid = product(
        _parse_grid(args.long_grid),
        _parse_grid(args.short_grid),
        setup_grid,
        _parse_grid(args.exit_opp_long_grid),
        _parse_grid(args.exit_opp_short_grid),
        entry_quote_modes,
        exit_quote_modes,
        stop_loss_grid,
        no_progress_minutes_grid,
        no_progress_mfe_grid,
        trail_arm_grid,
        trail_giveback_grid,
        time_decay_minutes_grid,
        time_decay_progress_grid,
        scalp_enabled_grid,
        _parse_grid(args.scalp_long_grid),
        _parse_grid(args.scalp_short_grid),
        scalp_setup_max_bars_grid,
        _parse_grid(args.scalp_min_signal_range_atr_grid),
        scalp_require_reversal_close_grid,
        candidate_enabled_grid,
        _parse_grid(args.candidate_long_grid),
        _parse_grid(args.candidate_short_grid),
        _parse_grid(args.candidate_opposite_max_grid),
        candidate_setup_max_bars_grid,
        _parse_grid(args.candidate_min_signal_range_atr_grid),
        candidate_long_enabled_grid,
        candidate_short_enabled_grid,
        candidate_start_hhmm_grid,
        candidate_end_hhmm_grid,
    )
    for (
        long_thr,
        short_thr,
        setup_max_bars,
        exit_opp_long_thr,
        exit_opp_short_thr,
        entry_quote_mode,
        exit_quote_mode,
        stop_loss_pct,
        no_progress_minutes,
        no_progress_mfe_pct,
        trail_arm_pct,
        trail_giveback_pct,
        time_decay_minutes,
        time_decay_progress_pct,
        scalp_enabled,
        scalp_long_thr,
        scalp_short_thr,
        scalp_setup_max_bars,
        scalp_min_range,
        scalp_reversal,
        candidate_enabled,
        candidate_long_thr,
        candidate_short_thr,
        candidate_opp_max,
        candidate_setup_max_bars,
        candidate_min_range,
        candidate_long_enabled,
        candidate_short_enabled,
        candidate_start_hhmm,
        candidate_end_hhmm,
    ) in grid:
        events = _run_one(
            decisions=decisions,
            one_min=one_min,
            long_thr=float(long_thr),
            short_thr=float(short_thr),
            exit_opp_long_thr=float(exit_opp_long_thr),
            exit_opp_short_thr=float(exit_opp_short_thr),
            setup_max_bars=int(setup_max_bars),
            cutoff_hhmm=str(args.cutoff_hhmm),
            new_entry_cutoff_hhmm=str(args.new_entry_cutoff_hhmm),
            entry_quote_mode=str(entry_quote_mode),
            exit_quote_mode=str(exit_quote_mode),
            quote_spread_bps=float(args.quote_spread_bps),
            stop_loss_pct=float(stop_loss_pct),
            no_progress_minutes=int(no_progress_minutes),
            no_progress_mfe_pct=float(no_progress_mfe_pct),
            trail_arm_pct=float(trail_arm_pct),
            trail_giveback_pct=float(trail_giveback_pct),
            time_decay_minutes=int(time_decay_minutes),
            time_decay_progress_pct=float(time_decay_progress_pct),
            scalp_enabled=bool(scalp_enabled),
            scalp_long_thr=float(scalp_long_thr),
            scalp_short_thr=float(scalp_short_thr),
            scalp_setup_max_bars=int(scalp_setup_max_bars),
            scalp_min_signal_range_atr=float(scalp_min_range),
            scalp_require_reversal_close=bool(scalp_reversal),
            candidate_enabled=bool(candidate_enabled),
            candidate_long_thr=float(candidate_long_thr),
            candidate_short_thr=float(candidate_short_thr),
            candidate_opposite_max=float(candidate_opp_max),
            candidate_setup_max_bars=int(candidate_setup_max_bars),
            candidate_min_signal_range_atr=float(candidate_min_range),
            candidate_long_enabled=bool(candidate_long_enabled),
            candidate_short_enabled=bool(candidate_short_enabled),
            candidate_start_hhmm=str(candidate_start_hhmm),
            candidate_end_hhmm=str(candidate_end_hhmm),
        )
        summary = {
            "long_threshold": float(long_thr),
            "short_threshold": float(short_thr),
            "setup_max_bars": int(setup_max_bars),
            "exit_opp_long_threshold": float(exit_opp_long_thr),
            "exit_opp_short_threshold": float(exit_opp_short_thr),
            "entry_quote_mode": str(entry_quote_mode),
            "exit_quote_mode": str(exit_quote_mode),
            "quote_spread_bps": float(args.quote_spread_bps),
            "new_entry_cutoff_hhmm": str(args.new_entry_cutoff_hhmm),
            "stop_loss_pct": float(stop_loss_pct),
            "no_progress_minutes": int(no_progress_minutes),
            "no_progress_mfe_pct": float(no_progress_mfe_pct),
            "trail_arm_pct": float(trail_arm_pct),
            "trail_giveback_pct": float(trail_giveback_pct),
            "time_decay_minutes": int(time_decay_minutes),
            "time_decay_progress_pct": float(time_decay_progress_pct),
            "scalp_enabled": bool(scalp_enabled),
            "scalp_long_threshold": float(scalp_long_thr),
            "scalp_short_threshold": float(scalp_short_thr),
            "scalp_setup_max_bars": int(scalp_setup_max_bars),
            "scalp_min_signal_range_atr": float(scalp_min_range),
            "scalp_require_reversal_close": bool(scalp_reversal),
            "candidate_enabled": bool(candidate_enabled),
            "candidate_long_threshold": float(candidate_long_thr),
            "candidate_short_threshold": float(candidate_short_thr),
            "candidate_opposite_max": float(candidate_opp_max),
            "candidate_setup_max_bars": int(candidate_setup_max_bars),
            "candidate_min_signal_range_atr": float(candidate_min_range),
            "candidate_long_enabled": bool(candidate_long_enabled),
            "candidate_short_enabled": bool(candidate_short_enabled),
            "candidate_start_hhmm": str(candidate_start_hhmm),
            "candidate_end_hhmm": str(candidate_end_hhmm),
            "decision_rows": int(len(decisions)),
            "first_decision": decisions["timestamp"].min(),
            "last_decision": decisions["timestamp"].max(),
            **_metrics(events),
        }
        summaries.append(summary)
        for event in events:
            all_events.append({**summary, **event})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries).sort_values(
        ["sum_return", "avg_return", "trades"], ascending=[False, False, False]
    )
    summary_df.to_csv(out, index=False)

    events_out = Path(args.events_out)
    if all_events:
        pd.DataFrame(all_events).to_csv(events_out, index=False)
    else:
        pd.DataFrame().to_csv(events_out, index=False)

    print(f"[threshold-sweep] decisions={len(decisions):,} one_min={len(one_min):,}")
    print(f"[threshold-sweep] wrote {out}")
    print(summary_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
