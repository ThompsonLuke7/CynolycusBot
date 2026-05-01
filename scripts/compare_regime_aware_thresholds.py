from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Policy.replay_option_proxy import ReplayOptionPriceProxy
from Policy.regime_filter import StickyRegimeConfig, add_sticky_trend_regime
from Policy.regime_probability_filter import (
    RegimeProbabilityCalibrator,
    RegimeProbabilityThresholdConfig,
)


DEFAULT_ANALYSIS_DIR = Path(
    "Data/models/ga_xgboost/10min/analysis/"
    "phase4_1m_oof_focused_trigger_sweep_l42_s15_full_1m_train"
)
DEFAULT_SIGNAL_FRAME = DEFAULT_ANALYSIS_DIR / "phase4_signal_frame.parquet"
DEFAULT_ONE_MIN = Path("Data/raw/spy/spy_intraday_1min.parquet")
DEFAULT_OUT = DEFAULT_ANALYSIS_DIR / "phase4_regime_aware_threshold_compare_summary.csv"
DEFAULT_EVENTS_OUT = DEFAULT_ANALYSIS_DIR / "phase4_regime_aware_threshold_compare_events.csv"


@dataclass
class PendingSetup:
    side: str
    ref: float
    setup_ts: pd.Timestamp
    start_ts: pd.Timestamp
    expires_at: pd.Timestamp
    prob: float
    threshold: float
    regime: str


@dataclass
class Position:
    policy: str
    side: str
    entry_ts: pd.Timestamp
    entry_spot: float
    entry_premium: float
    symbol: str
    best_premium: float
    best_seen_at: pd.Timestamp
    entry_prob: float
    threshold: float
    regime: str


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


def _combine_prob(df: pd.DataFrame, side: str) -> pd.Series:
    test_col = f"p_{side}_test"
    oof_col = f"p_{side}_oof_train"
    if test_col in df.columns and oof_col in df.columns:
        return pd.to_numeric(df[test_col], errors="coerce").combine_first(
            pd.to_numeric(df[oof_col], errors="coerce")
        )
    if test_col in df.columns:
        return pd.to_numeric(df[test_col], errors="coerce")
    if oof_col in df.columns:
        return pd.to_numeric(df[oof_col], errors="coerce")
    raise KeyError(f"Missing probability columns for side={side}")


def _load_signal_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" not in df.columns:
            raise ValueError("Signal frame must have DatetimeIndex or timestamp column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.loc[pd.notna(idx)].copy()
    df.index = pd.DatetimeIndex(idx[pd.notna(idx)]).tz_convert("America/New_York")
    df = df.sort_index()
    df["timestamp"] = df.index
    df["available_ts"] = df.index + pd.Timedelta(minutes=10)
    df["p_enter_long"] = _combine_prob(df, "long")
    df["p_enter_short"] = _combine_prob(df, "short")
    for col in ("open", "high", "low", "close", "ema_fast", "ema_slow"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_sticky_trend_regime(df, config=StickyRegimeConfig())
    keep = [
        "timestamp",
        "available_ts",
        "open",
        "high",
        "low",
        "close",
        "p_enter_long",
        "p_enter_short",
        "trend_regime",
    ]
    return df[keep].dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _load_one_min(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"] >= start - pd.Timedelta(days=2)) & (df["timestamp"] <= end + pd.Timedelta(days=1))].copy()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "symbol" not in df.columns:
        df["symbol"] = "SPY"
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _next_expiry(ts: pd.Timestamp, cutoff_hhmm: str) -> pd.Timestamp:
    cutoff_h, cutoff_m = [int(x) for x in cutoff_hhmm.split(":")]
    cutoff = ts.replace(hour=cutoff_h, minute=cutoff_m, second=0, microsecond=0)
    expiry_day = ts.normalize()
    if ts >= cutoff:
        expiry_day = expiry_day + pd.offsets.BDay(1)
    return pd.Timestamp(expiry_day).tz_localize(None)


def _sim_symbol(ts: pd.Timestamp, right: str, strike: float, cutoff_hhmm: str) -> str:
    expiry = _next_expiry(ts, cutoff_hhmm)
    return f".SIM_SPY_{expiry.strftime('%y%m%d')}_{right}_{float(strike):.2f}"


def _daily_atr_proxy(decisions: pd.DataFrame) -> pd.Series:
    tr = (pd.to_numeric(decisions["high"], errors="coerce") - pd.to_numeric(decisions["low"], errors="coerce")).abs()
    atr = tr.rolling(14, min_periods=3).mean().bfill().fillna(0.5)
    return atr.clip(lower=0.05)


def _threshold_policy(
    name: str,
    *,
    calibrator: RegimeProbabilityCalibrator | None = None,
) -> Callable[[pd.Series], tuple[float, float]]:
    if name == "fixed_current_035_065":
        return lambda _row: (0.35, 0.65)
    if name == "fixed_conservative_035_085":
        return lambda _row: (0.35, 0.85)
    if name == "regime_simple":
        def _fn(row: pd.Series) -> tuple[float, float]:
            regime = str(row.get("trend_regime", "neutral"))
            if regime == "bullish":
                return 0.35, 0.85
            if regime == "bearish":
                return 0.65, 0.65
            return 0.50, 0.75
        return _fn
    if name == "regime_balanced":
        def _fn(row: pd.Series) -> tuple[float, float]:
            regime = str(row.get("trend_regime", "neutral"))
            if regime == "bullish":
                return 0.35, 0.80
            if regime == "bearish":
                return 0.60, 0.65
            return 0.45, 0.75
        return _fn
    if name == "regime_percentile":
        if calibrator is None:
            raise ValueError("regime_percentile policy requires a RegimeProbabilityCalibrator")

        def _fn(row: pd.Series) -> tuple[float, float]:
            regime = str(row.get("trend_regime", "neutral"))
            return (
                calibrator.threshold(regime=regime, side="long"),
                calibrator.threshold(regime=regime, side="short"),
            )

        return _fn
    raise ValueError(f"Unknown policy: {name}")


def _update_setups(
    row: pd.Series,
    *,
    long_thr: float,
    short_thr: float,
    setup_max_bars: int,
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

    for side, valid, prob, threshold, ref in (
        ("long", long_valid, p_long, long_thr, _as_float(row.get("high"))),
        ("short", short_valid, p_short, short_thr, _as_float(row.get("low"))),
    ):
        if not (math.isfinite(prob) and prob >= threshold):
            above[side] = False
            peak[side] = float("nan")
            pending[side] = None
            continue
        refresh = valid
        if refresh and math.isfinite(ref):
            setup_ts = row["timestamp"]
            pending[side] = PendingSetup(
                side=side,
                ref=float(ref),
                setup_ts=setup_ts,
                start_ts=setup_ts + pd.Timedelta(minutes=10),
                expires_at=setup_ts + pd.Timedelta(minutes=10 * (setup_max_bars + 1)),
                prob=float(prob),
                threshold=float(threshold),
                regime=str(row.get("trend_regime", "neutral")),
            )
            peak[side] = float(prob)
        above[side] = bool(prob >= threshold)


def _triggered(setup: PendingSetup, row: dict[str, Any]) -> bool:
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
        "policy": pos.policy,
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
        "entry_regime": pos.regime,
        "symbol": pos.symbol,
    }


def _run_policy(
    *,
    policy_name: str,
    decisions: pd.DataFrame,
    one_min: pd.DataFrame,
    setup_max_bars: int,
    cutoff_hhmm: str,
    calibrator: RegimeProbabilityCalibrator | None,
) -> list[dict[str, Any]]:
    threshold_fn = _threshold_policy(policy_name, calibrator=calibrator)
    proxy = ReplayOptionPriceProxy(
        tz_name="America/New_York",
        expiry_hhmm="15:40",
        iv_floor=0.12,
        iv_ceiling=0.90,
        iv_multiplier=1.50,
        min_dte_minutes=1.0,
    )
    decisions = decisions.copy().reset_index(drop=True)
    decisions["atr_proxy"] = _daily_atr_proxy(decisions)
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
            latest_atr = _as_float(drow.get("atr_proxy"))
            long_thr, short_thr = threshold_fn(drow)
            _update_setups(
                drow,
                long_thr=float(long_thr),
                short_thr=float(short_thr),
                setup_max_bars=setup_max_bars,
                pending=pending,
                above=above,
                peak=peak,
            )
            decision_idx += 1

        close = _as_float(row.get("close"))
        if pos is not None:
            premium = proxy.price(pos.symbol, mode="mid")
            if math.isfinite(premium):
                if premium > pos.best_premium:
                    pos.best_premium = float(premium)
                    pos.best_seen_at = ts
                ret = premium / pos.entry_premium - 1.0
                best_ret = pos.best_premium / pos.entry_premium - 1.0
                minutes_since_best = (ts - pos.best_seen_at).total_seconds() / 60.0
                opp_prob = latest_probs["short"] if pos.side == "long" else latest_probs["long"]
                reason = None
                if best_ret >= 2.0 and ((pos.best_premium - premium) / pos.entry_premium) >= 0.25:
                    reason = "adaptive_trail"
                elif minutes_since_best >= 80 and best_ret < 1.0 and ret <= 0.0:
                    reason = "time_decay"
                elif best_ret >= 1.0 and math.isfinite(opp_prob) and opp_prob >= 0.60:
                    reason = "opposite_signal"
                elif ts.hour == 15 and ts.minute >= 40 and pos.symbol.split("_")[2] == ts.strftime("%y%m%d"):
                    reason = "same_day_eod"
                if reason:
                    events.append(_close_event(pos, ts, close, premium, reason))
                    pos = None

        if pos is None:
            live_setups = [
                setup
                for setup in (pending.get("long"), pending.get("short"))
                if setup is not None and setup.start_ts <= ts < setup.expires_at
            ]
            triggered = [setup for setup in live_setups if _triggered(setup, row)]
            if triggered:
                if len(triggered) > 1:
                    triggered.sort(key=lambda s: s.prob - s.threshold, reverse=True)
                setup = triggered[0]
                right = "C" if setup.side == "long" else "P"
                strike = round(close + latest_atr) if setup.side == "long" else round(close - latest_atr)
                symbol = _sim_symbol(ts, right, strike, cutoff_hhmm)
                premium = proxy.price(symbol, mode="mid")
                if math.isfinite(premium) and premium > 0:
                    pos = Position(
                        policy=policy_name,
                        side=setup.side,
                        entry_ts=ts,
                        entry_spot=float(close),
                        entry_premium=float(premium),
                        symbol=symbol,
                        best_premium=float(premium),
                        best_seen_at=ts,
                        entry_prob=setup.prob,
                        threshold=setup.threshold,
                        regime=setup.regime,
                    )
                    pending[setup.side] = None

    if pos is not None and len(one_min):
        last = one_min.iloc[-1]
        premium = proxy.price(pos.symbol, mode="mid")
        if math.isfinite(premium):
            events.append(_close_event(pos, last["timestamp"], _as_float(last["close"]), premium, "window_end"))
    return events


def _metrics(events: list[dict[str, Any]], *, label: str, decisions: pd.DataFrame) -> dict[str, Any]:
    returns = np.array([float(e["return"]) for e in events], dtype=float)
    mfes = np.array([float(e["mfe"]) for e in events], dtype=float)
    out = {
        "window": label,
        "decision_rows": int(len(decisions)),
        "first_decision": decisions["timestamp"].min() if len(decisions) else pd.NaT,
        "last_decision": decisions["timestamp"].max() if len(decisions) else pd.NaT,
        "trades": int(len(events)),
        "long_trades": int(sum(1 for e in events if e["side"] == "long")),
        "short_trades": int(sum(1 for e in events if e["side"] == "short")),
        "sum_return": float(np.nansum(returns)) if returns.size else 0.0,
        "avg_return": float(np.nanmean(returns)) if returns.size else float("nan"),
        "median_return": float(np.nanmedian(returns)) if returns.size else float("nan"),
        "win_rate": float(np.nanmean(returns > 0)) if returns.size else float("nan"),
        "p25_return": float(np.nanpercentile(returns, 25)) if returns.size else float("nan"),
        "p75_return": float(np.nanpercentile(returns, 75)) if returns.size else float("nan"),
        "avg_mfe": float(np.nanmean(mfes)) if mfes.size else float("nan"),
    }
    for reason in ("adaptive_trail", "time_decay", "opposite_signal", "same_day_eod", "window_end"):
        out[f"{reason}_count"] = int(sum(1 for e in events if e["reason"] == reason))
    return out


def _filter_window(df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    out = df
    if start is not None and pd.notna(start):
        out = out[out["timestamp"] >= start]
    if end is not None and pd.notna(end):
        out = out[out["timestamp"] <= end]
    return out.copy().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed vs regime-aware setup thresholds.")
    parser.add_argument("--signal-frame", default=str(DEFAULT_SIGNAL_FRAME))
    parser.add_argument("--one-min", default=str(DEFAULT_ONE_MIN))
    parser.add_argument("--recent-months", type=int, default=3)
    parser.add_argument("--setup-max-bars", type=int, default=4)
    parser.add_argument("--cutoff-hhmm", default="13:00")
    parser.add_argument("--policies", default="fixed_current_035_065,fixed_conservative_035_085,regime_simple,regime_balanced,regime_percentile")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--events-out", default=str(DEFAULT_EVENTS_OUT))
    args = parser.parse_args()

    signal = _load_signal_frame(Path(args.signal_frame))
    if signal.empty:
        raise SystemExit("No signal rows found.")
    full_start = signal["timestamp"].min()
    full_end = signal["timestamp"].max()
    recent_start = full_end - pd.DateOffset(months=int(args.recent_months))
    windows = [
        ("recent", recent_start, full_end),
        ("full", full_start, full_end),
    ]
    policy_names = [x.strip() for x in str(args.policies).split(",") if x.strip()]
    calibrator = RegimeProbabilityCalibrator.from_signal_frame(
        Path(args.signal_frame),
        threshold_config=RegimeProbabilityThresholdConfig(),
    )

    summaries: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for window_label, start_ts, end_ts in windows:
        decisions = _filter_window(signal, start_ts, end_ts)
        if decisions.empty:
            continue
        one_min = _load_one_min(Path(args.one_min), decisions["timestamp"].min(), decisions["timestamp"].max())
        one_min = one_min[one_min["timestamp"] >= decisions["timestamp"].min() - pd.Timedelta(minutes=10)].reset_index(drop=True)
        if one_min.empty:
            continue
        for policy_name in policy_names:
            events = _run_policy(
                policy_name=policy_name,
                decisions=decisions,
                one_min=one_min,
                setup_max_bars=int(args.setup_max_bars),
                cutoff_hhmm=str(args.cutoff_hhmm),
                calibrator=calibrator,
            )
            summaries.append(
                {
                    "policy": policy_name,
                    **_metrics(events, label=window_label, decisions=decisions),
                }
            )
            for event in events:
                all_events.append({"window": window_label, **event})

    summary_df = pd.DataFrame(summaries).sort_values(
        ["window", "sum_return", "avg_return", "trades"],
        ascending=[True, False, False, False],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out, index=False)
    events_out = Path(args.events_out)
    if all_events:
        pd.DataFrame(all_events).to_csv(events_out, index=False)
    else:
        pd.DataFrame().to_csv(events_out, index=False)

    print(f"[regime-threshold-compare] wrote {out}")
    print(f"[regime-threshold-compare] wrote {events_out}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
