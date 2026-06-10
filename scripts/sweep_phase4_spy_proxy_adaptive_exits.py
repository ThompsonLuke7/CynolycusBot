from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from strategies.spy_intraday.Models.ga_xgboost.swing_label_weights import compute_wilder_atr  # noqa: E402
from scripts.sweep_phase4_spy_proxy_profit_locks import (  # noqa: E402
    DEFAULT_ANALYSIS_DIR,
    _build_entries,
    _path_after_entry,
    _load_scoreboard,
    _parse_float_list,
)
from strategies.spy_intraday.Models.ga_xgboost.analyze_phase4_triggers import _load_execution_1m  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep adaptive SPY-proxy exit policies for the phase4 entry set."
    )
    parser.add_argument(
        "--signal-frame",
        default=f"{DEFAULT_ANALYSIS_DIR}/phase4_signal_frame.parquet",
        help="Phase4 signal frame parquet.",
    )
    parser.add_argument(
        "--scoreboard",
        default=f"{DEFAULT_ANALYSIS_DIR}/phase4_trigger_scoreboard.json",
        help="Phase4 trigger scoreboard JSON used to select/rebuild entries.",
    )
    parser.add_argument("--split", default="oof", choices=["oof", "test"])
    parser.add_argument("--variant", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--long-setup-threshold", type=float, default=None)
    parser.add_argument("--short-setup-threshold", type=float, default=None)
    parser.add_argument("--cooldown-bars", type=int, default=None)
    parser.add_argument("--post-setup-max-bars", type=int, default=None)
    parser.add_argument(
        "--use-1m-execution",
        action="store_true",
        help="Use 1-minute SPY bars for live-like trigger rebuilding and exit path simulation.",
    )
    parser.add_argument(
        "--execution-1m-path",
        default="Data/raw/spy/spy_intraday_1min.parquet",
        help="1-minute SPY OHLC parquet with timestamp/open/high/low/close columns.",
    )
    parser.add_argument("--premium-atr-mult", type=float, default=2.5)
    parser.add_argument("--proxy-mode", choices=["atr", "black_scholes"], default="atr")
    parser.add_argument("--bs-iv-floor", type=float, default=0.12)
    parser.add_argument("--bs-iv-ceil", type=float, default=0.90)
    parser.add_argument("--bs-iv-mult", type=float, default=1.50)
    parser.add_argument("--bs-iv-shock", type=float, default=0.20)
    parser.add_argument("--bs-min-dte-minutes", type=float, default=1.0)
    parser.add_argument("--bs-strike-round", type=float, default=1.0)
    parser.add_argument("--stop-loss-values", default="0.8,1.0")
    parser.add_argument("--late-hybrid-stale-bars", default="5,8,12")
    parser.add_argument("--late-hybrid-progress-values", default="0.75")
    parser.add_argument("--late-hybrid-opp-thresholds", default="0.60,0.70,0.80")
    parser.add_argument("--late-hybrid-arm-values", default="2.0")
    parser.add_argument("--late-hybrid-giveback-values", default="0.25")
    parser.add_argument(
        "--strategy-styles",
        default="",
        help="Optional comma-separated filter, for example baseline,late_hybrid.",
    )
    parser.add_argument(
        "--position-modes",
        default="hedged",
        help=(
            "Comma-separated position handling modes. hedged keeps the historical "
            "independent long/short evaluation; single skips any candidate entry "
            "while a prior trade from the same regime is still open."
        ),
    )
    parser.add_argument("--exit-hhmm", default="15:40")
    parser.add_argument("--horizon-bars", type=int, default=39)
    parser.add_argument(
        "--out",
        default=f"{DEFAULT_ANALYSIS_DIR}/phase4_spy_proxy_adaptive_exit_sweep.csv",
    )
    parser.add_argument(
        "--events-out",
        default=f"{DEFAULT_ANALYSIS_DIR}/phase4_spy_proxy_adaptive_exit_events.csv",
    )
    return parser.parse_args()


def _session_cutoff(ts: pd.Timestamp, *, hhmm: str) -> pd.Timestamp:
    hour_raw, minute_raw = str(hhmm).split(":", 1)
    return pd.Timestamp(ts).normalize() + pd.Timedelta(hours=int(hour_raw), minutes=int(minute_raw))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _bs_price(
    *,
    spot: float,
    strike: float,
    tau_years: float,
    iv: float,
    kind: str,
    rate: float = 0.0,
) -> float:
    if not (np.isfinite(spot) and np.isfinite(strike) and spot > 0.0 and strike > 0.0):
        return float("nan")
    tau = max(float(tau_years), 1e-8)
    sigma = max(float(iv), 1e-4)
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    if kind == "call":
        return float(spot * _norm_cdf(d1) - strike * math.exp(-rate * tau) * _norm_cdf(d2))
    return float(strike * math.exp(-rate * tau) * _norm_cdf(-d2) - spot * _norm_cdf(-d1))


def _estimate_iv(
    close: np.ndarray,
    *,
    floor: float,
    ceil: float,
    mult: float,
) -> np.ndarray:
    ret = pd.Series(close).pct_change()
    bars_per_year = 252.0 * 39.0
    realized = ret.rolling(78, min_periods=20).std().to_numpy(dtype=float) * math.sqrt(bars_per_year)
    iv = np.nan_to_num(realized * float(mult), nan=float(floor), posinf=float(ceil), neginf=float(floor))
    return np.clip(iv, float(floor), float(ceil))


def _rounded_strike(entry: float, *, round_to: float) -> float:
    step = max(0.01, float(round_to))
    return float(round(float(entry) / step) * step)


def _bs_path_values(
    *,
    side: str,
    entry: float,
    strike: float,
    expiry_ts: pd.Timestamp,
    path_times: pd.Index,
    path_spots: np.ndarray,
    iv_base: np.ndarray,
    iv_entry: float,
    entry_atr: float,
    entry_idx: int,
    path_idx: np.ndarray,
    iv_shock: float,
    iv_floor: float,
    iv_ceil: float,
    min_dte_minutes: float,
) -> np.ndarray:
    kind = "call" if side == "long" else "put"
    out = np.full(path_idx.size, np.nan, dtype=float)
    for j, i in enumerate(path_idx):
        ts = pd.Timestamp(path_times[j])
        minutes_left = max(float(min_dte_minutes), (expiry_ts - ts).total_seconds() / 60.0)
        tau = minutes_left / (365.0 * 24.0 * 60.0)
        move_atr = abs(float(path_spots[j]) - float(entry)) / max(float(entry_atr), 1e-6)
        shocked_iv = max(float(iv_entry), float(iv_base[i])) + float(iv_shock) * min(3.0, move_atr) / 3.0
        iv = float(np.clip(shocked_iv, float(iv_floor), float(iv_ceil)))
        out[j] = _bs_price(spot=float(path_spots[j]), strike=strike, tau_years=tau, iv=iv, kind=kind)
    return out


def _build_trade_specs(
    feature_df: pd.DataFrame,
    *,
    split: str,
    long_entries: np.ndarray,
    short_entries: np.ndarray,
    long_prices: np.ndarray,
    short_prices: np.ndarray,
    long_times: np.ndarray,
    short_times: np.ndarray,
    premium_atr_mult: float,
    exit_hhmm: str,
    horizon_bars: int,
    proxy_mode: str,
    bs_iv_floor: float,
    bs_iv_ceil: float,
    bs_iv_mult: float,
    bs_iv_shock: float,
    bs_min_dte_minutes: float,
    bs_strike_round: float,
    execution_1m: pd.DataFrame | None,
) -> list[dict[str, object]]:
    high = pd.to_numeric(feature_df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(feature_df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(feature_df["close"], errors="coerce").to_numpy(dtype=float)
    atr = compute_wilder_atr(high, low, close, length=14)
    atr_med = pd.Series(atr).rolling(78, min_periods=20).median().to_numpy(dtype=float)
    iv_est = _estimate_iv(close, floor=bs_iv_floor, ceil=bs_iv_ceil, mult=bs_iv_mult)
    p_suffix = "oof_train" if split == "oof" else "test"
    p_long = pd.to_numeric(feature_df[f"p_long_{p_suffix}"], errors="coerce").to_numpy(dtype=float)
    p_short = pd.to_numeric(feature_df[f"p_short_{p_suffix}"], errors="coerce").to_numpy(dtype=float)
    ema_fast = pd.to_numeric(feature_df.get("ema_fast"), errors="coerce").to_numpy(dtype=float)
    ema_slow = pd.to_numeric(feature_df.get("ema_slow"), errors="coerce").to_numpy(dtype=float)
    idx_arr = np.arange(len(feature_df))
    specs: list[dict[str, object]] = []

    for side, entries, prices, times in (
        ("long", long_entries, long_prices, long_times),
        ("short", short_entries, short_prices, short_times),
    ):
        for idx in np.flatnonzero(entries):
            entry = float(prices[idx]) if np.isfinite(prices[idx]) else float(close[idx])
            entry_ts = pd.Timestamp(times[idx] if pd.notna(times[idx]) else feature_df.index[idx])
            cutoff = _session_cutoff(entry_ts, hhmm=exit_hhmm)
            if str(proxy_mode) == "black_scholes":
                strike = _rounded_strike(entry, round_to=bs_strike_round)
                expiry_ts = cutoff
                minutes_left = max(float(bs_min_dte_minutes), (expiry_ts - entry_ts).total_seconds() / 60.0)
                tau = minutes_left / (365.0 * 24.0 * 60.0)
                kind = "call" if side == "long" else "put"
                premium = _bs_price(
                    spot=entry,
                    strike=strike,
                    tau_years=tau,
                    iv=float(iv_est[idx]),
                    kind=kind,
                )
            else:
                strike = float("nan")
                premium = float(atr[idx]) * float(premium_atr_mult)
            if not np.isfinite(entry) or not np.isfinite(premium) or premium <= 0.0:
                continue
            path_times_raw, path_high, path_low, path_close = _path_after_entry(
                feature_df,
                entry_idx=int(idx),
                entry_ts=entry_ts,
                execution_1m=execution_1m,
                exit_hhmm=exit_hhmm,
                horizon_bars=horizon_bars,
            )
            if len(path_times_raw) == 0:
                continue
            path_times = pd.DatetimeIndex(path_times_raw)
            if execution_1m is not None:
                path_idx = feature_df.index.searchsorted(path_times, side="right") - 1
                path_idx = np.clip(path_idx, 0, len(feature_df) - 1).astype(int)
            else:
                path_idx = np.array([feature_df.index.get_loc(ts) for ts in path_times], dtype=int)
            if str(proxy_mode) == "black_scholes":
                hi_values = _bs_path_values(
                    side=side,
                    entry=entry,
                    strike=float(strike),
                    expiry_ts=cutoff,
                    path_times=path_times,
                    path_spots=path_high,
                    iv_base=iv_est,
                    iv_entry=float(iv_est[idx]),
                    entry_atr=float(atr[idx]),
                    entry_idx=int(idx),
                    path_idx=path_idx,
                    iv_shock=bs_iv_shock,
                    iv_floor=bs_iv_floor,
                    iv_ceil=bs_iv_ceil,
                    min_dte_minutes=bs_min_dte_minutes,
                )
                lo_values = _bs_path_values(
                    side=side,
                    entry=entry,
                    strike=float(strike),
                    expiry_ts=cutoff,
                    path_times=path_times,
                    path_spots=path_low,
                    iv_base=iv_est,
                    iv_entry=float(iv_est[idx]),
                    entry_atr=float(atr[idx]),
                    entry_idx=int(idx),
                    path_idx=path_idx,
                    iv_shock=bs_iv_shock,
                    iv_floor=bs_iv_floor,
                    iv_ceil=bs_iv_ceil,
                    min_dte_minutes=bs_min_dte_minutes,
                )
                close_values = _bs_path_values(
                    side=side,
                    entry=entry,
                    strike=float(strike),
                    expiry_ts=cutoff,
                    path_times=path_times,
                    path_spots=path_close,
                    iv_base=iv_est,
                    iv_entry=float(iv_est[idx]),
                    entry_atr=float(atr[idx]),
                    entry_idx=int(idx),
                    path_idx=path_idx,
                    iv_shock=bs_iv_shock,
                    iv_floor=bs_iv_floor,
                    iv_ceil=bs_iv_ceil,
                    min_dte_minutes=bs_min_dte_minutes,
                )
                if side == "long":
                    favorable_path = (hi_values - premium) / premium
                    adverse_path = (lo_values - premium) / premium
                else:
                    favorable_path = (lo_values - premium) / premium
                    adverse_path = (hi_values - premium) / premium
                close_path = (close_values - premium) / premium
            else:
                favorable_path = np.array([], dtype=float)
                adverse_path = np.array([], dtype=float)
                close_path = np.array([], dtype=float)

            entry_prob = p_long[idx] if side == "long" else p_short[idx]
            entry_opp_prob = p_short[idx] if side == "long" else p_long[idx]
            trend_spread = abs(ema_fast[idx] - ema_slow[idx]) / atr[idx] if np.isfinite(atr[idx]) and atr[idx] > 0 else 0.0
            trend_aligned = (
                bool(ema_fast[idx] > ema_slow[idx]) if side == "long" else bool(ema_fast[idx] < ema_slow[idx])
            )
            vol_ratio = atr[idx] / atr_med[idx] if np.isfinite(atr_med[idx]) and atr_med[idx] > 0 else 1.0
            specs.append(
                {
                    "side": side,
                    "entry_idx": int(idx),
                    "entry_time": entry_ts,
                    "entry": float(entry),
                    "premium": float(premium),
                    "proxy_mode": str(proxy_mode),
                    "strike": float(strike),
                    "entry_iv": float(iv_est[idx]) if np.isfinite(iv_est[idx]) else np.nan,
                    "entry_prob": float(entry_prob) if np.isfinite(entry_prob) else np.nan,
                    "entry_opp_prob": float(entry_opp_prob) if np.isfinite(entry_opp_prob) else np.nan,
                    "trend_aligned": bool(trend_aligned and trend_spread >= 0.10),
                    "trend_spread_atr": float(trend_spread),
                    "vol_ratio": float(vol_ratio) if np.isfinite(vol_ratio) else 1.0,
                    "path_idx": path_idx,
                    "path_times": path_times,
                    "path_high": path_high,
                    "path_low": path_low,
                    "path_close": path_close,
                    "path_bar_multiplier": 10 if execution_1m is not None else 1,
                    "favorable_path": favorable_path,
                    "adverse_path": adverse_path,
                    "close_path": close_path,
                }
            )
    return specs


def _strategy_specs(
    stop_losses: list[float],
    *,
    late_hybrid_stale_bars: list[int] | None = None,
    late_hybrid_progress_values: list[float] | None = None,
    late_hybrid_opp_thresholds: list[float] | None = None,
    late_hybrid_arm_values: list[float] | None = None,
    late_hybrid_giveback_values: list[float] | None = None,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    late_hybrid_stale_bars = late_hybrid_stale_bars or [5, 8, 12]
    late_hybrid_progress_values = late_hybrid_progress_values or [0.75]
    late_hybrid_opp_thresholds = late_hybrid_opp_thresholds or [0.60, 0.70, 0.80]
    late_hybrid_arm_values = late_hybrid_arm_values or [2.0]
    late_hybrid_giveback_values = late_hybrid_giveback_values or [0.25]
    for sl in stop_losses:
        specs.append({"name": f"baseline_tp2_sl{sl:.2f}", "style": "baseline", "tp": 2.0, "sl": sl})
        specs.append({"name": f"baseline_uncapped_sl{sl:.2f}", "style": "baseline", "tp": None, "sl": sl})
        for high_arm in (1.5, 2.0, 2.5):
            for mid_arm in (1.0, 1.5):
                specs.append(
                    {
                        "name": f"conftrail_sl{sl:.2f}_hi{high_arm:.1f}_mid{mid_arm:.1f}",
                        "style": "confidence_trail",
                        "sl": sl,
                        "low_arm": 0.75,
                        "mid_arm": mid_arm,
                        "high_arm": high_arm,
                        "low_gb": 0.25,
                        "mid_gb": 0.50,
                        "high_gb": 0.75,
                    }
                )
        for base_gb in (0.25, 0.35, 0.50):
            specs.append(
                {
                    "name": f"mfe_decay_vol_sl{sl:.2f}_gb{base_gb:.2f}",
                    "style": "mfe_decay_vol",
                    "sl": sl,
                    "arm": 1.0,
                    "base_gb": base_gb,
                }
            )
        for trend_gb in (0.75, 1.0):
            for chop_gb in (0.25, 0.50):
                specs.append(
                    {
                        "name": f"regime_trail_sl{sl:.2f}_trend{trend_gb:.2f}_chop{chop_gb:.2f}",
                        "style": "regime_trail",
                        "sl": sl,
                        "trend_arm": 2.0,
                        "chop_arm": 0.75,
                        "trend_gb": trend_gb,
                        "chop_gb": chop_gb,
                    }
                )
        for stale_bars in (3, 5, 8):
            for opp_thr in (0.50, 0.60, 0.70):
                specs.append(
                    {
                        "name": f"hybrid_sl{sl:.2f}_stale{stale_bars}_opp{opp_thr:.2f}",
                        "style": "hybrid",
                        "sl": sl,
                        "arm": 2.0,
                        "gb": 0.25,
                        "early_arm": 1.0,
                        "early_gb": 0.50,
                        "stale_bars": stale_bars,
                        "progress": 0.50,
                        "opp_thr": opp_thr,
                    }
                )
        for arm in (2.0, 2.5, 3.0):
            for giveback in (0.25, 0.50, 0.75):
                specs.append(
                    {
                        "name": f"late_trail_sl{sl:.2f}_arm{arm:.1f}_gb{giveback:.2f}",
                        "style": "late_trail",
                        "sl": sl,
                        "arm": arm,
                        "gb": giveback,
                    }
                )
        for high_arm in (2.5, 3.0):
            for mid_arm in (2.0, 2.5):
                specs.append(
                    {
                        "name": f"conf_late_sl{sl:.2f}_hi{high_arm:.1f}_mid{mid_arm:.1f}",
                        "style": "confidence_late_trail",
                        "sl": sl,
                        "low_arm": 1.5,
                        "mid_arm": mid_arm,
                        "high_arm": high_arm,
                        "low_gb": 0.25,
                        "mid_gb": 0.50,
                        "high_gb": 0.75,
                    }
                )
        for base_gb in (0.25, 0.50, 0.75):
            specs.append(
                {
                    "name": f"vol_late_sl{sl:.2f}_gb{base_gb:.2f}",
                    "style": "vol_late_trail",
                    "sl": sl,
                    "arm": 2.0,
                    "base_gb": base_gb,
                }
            )
        for arm in late_hybrid_arm_values:
            for giveback in late_hybrid_giveback_values:
                for stale_bars in late_hybrid_stale_bars:
                    for progress in late_hybrid_progress_values:
                        for opp_thr in late_hybrid_opp_thresholds:
                            specs.append(
                                {
                                    "name": (
                                        f"late_hybrid_sl{sl:.2f}_arm{arm:.1f}_gb{giveback:.2f}"
                                        f"_stale{stale_bars}_prog{progress:.2f}_opp{opp_thr:.2f}"
                                    ),
                                    "style": "late_hybrid",
                                    "sl": sl,
                                    "arm": arm,
                                    "gb": giveback,
                                    "stale_bars": stale_bars,
                                    "progress": progress,
                                    "opp_thr": opp_thr,
                                }
                            )
    return specs


def _trail_exit_move(best_move: float, giveback: float, premium: float) -> float:
    return max(0.0, best_move * (1.0 - min(1.0, max(0.0, float(giveback)))))


def _effective_stale_bars(spec: dict[str, object], trade: dict[str, object]) -> int:
    multiplier = max(1, int(trade.get("path_bar_multiplier", 1)))
    return int(spec["stale_bars"]) * multiplier


def _dynamic_params(spec: dict[str, object], trade: dict[str, object], best_pct: float) -> tuple[float, float]:
    style = str(spec["style"])
    if style == "confidence_trail":
        prob = float(trade["entry_prob"])
        if np.isfinite(prob) and prob >= 0.80:
            return float(spec["high_arm"]), float(spec["high_gb"])
        if np.isfinite(prob) and prob < 0.60:
            return float(spec["low_arm"]), float(spec["low_gb"])
        return float(spec["mid_arm"]), float(spec["mid_gb"])
    if style == "mfe_decay_vol":
        vol_ratio = float(trade["vol_ratio"])
        base_gb = float(spec["base_gb"])
        giveback = base_gb
        if vol_ratio >= 1.20:
            giveback = min(1.50, base_gb + 0.25)
        elif vol_ratio <= 0.80:
            giveback = max(0.15, base_gb - 0.10)
        if best_pct >= 2.0:
            giveback = max(0.25, giveback - 0.10)
        return float(spec["arm"]), giveback
    if style == "regime_trail":
        if bool(trade["trend_aligned"]):
            return float(spec["trend_arm"]), float(spec["trend_gb"])
        return float(spec["chop_arm"]), float(spec["chop_gb"])
    if style == "hybrid":
        if best_pct >= float(spec["arm"]):
            return float(spec["arm"]), float(spec["gb"])
        return float(spec["early_arm"]), float(spec["early_gb"])
    if style == "late_trail":
        return float(spec["arm"]), float(spec["gb"])
    if style == "confidence_late_trail":
        prob = float(trade["entry_prob"])
        if np.isfinite(prob) and prob >= 0.80:
            return float(spec["high_arm"]), float(spec["high_gb"])
        if np.isfinite(prob) and prob < 0.60:
            return float(spec["low_arm"]), float(spec["low_gb"])
        return float(spec["mid_arm"]), float(spec["mid_gb"])
    if style == "vol_late_trail":
        vol_ratio = float(trade["vol_ratio"])
        giveback = float(spec["base_gb"])
        if vol_ratio >= 1.20:
            giveback = min(1.50, giveback + 0.25)
        elif vol_ratio <= 0.80:
            giveback = max(0.15, giveback - 0.10)
        return float(spec["arm"]), giveback
    if style == "late_hybrid":
        return float(spec["arm"]), float(spec["gb"])
    return float("inf"), float("inf")


def _simulate_trade(
    market: dict[str, object],
    trade: dict[str, object],
    spec: dict[str, object],
) -> dict[str, object] | None:
    high = market["high"]
    low = market["low"]
    close = market["close"]
    p_long = market["p_long"]
    p_short = market["p_short"]
    index = market["index"]
    assert isinstance(high, np.ndarray)
    assert isinstance(low, np.ndarray)
    assert isinstance(close, np.ndarray)
    assert isinstance(p_long, np.ndarray)
    assert isinstance(p_short, np.ndarray)

    side = str(trade["side"])
    entry = float(trade["entry"])
    premium = float(trade["premium"])
    path_idx = trade["path_idx"]
    if not isinstance(path_idx, np.ndarray) or path_idx.size == 0:
        return None
    path_times = trade.get("path_times")
    path_high = trade.get("path_high")
    path_low = trade.get("path_low")
    path_close = trade.get("path_close")
    proxy_mode = str(trade.get("proxy_mode", "atr"))
    favorable_path = trade.get("favorable_path")
    adverse_path = trade.get("adverse_path")
    close_path = trade.get("close_path")

    best_move = 0.0
    worst_move = 0.0
    final_move = 0.0
    last_peak_bar = 0
    exit_reason = "timeout"
    exit_idx = int(path_idx[-1])
    exit_pos = path_idx.size - 1
    exit_move = 0.0
    exit_underlying_price = float("nan")
    stop_loss = float(spec["sl"])
    tp = spec.get("tp")

    for bar_n, i in enumerate(path_idx, start=1):
        pos = bar_n - 1
        if proxy_mode == "black_scholes":
            if not (
                isinstance(favorable_path, np.ndarray)
                and isinstance(adverse_path, np.ndarray)
                and isinstance(close_path, np.ndarray)
                and bar_n - 1 < favorable_path.size
            ):
                return None
            favorable_pct = float(favorable_path[bar_n - 1])
            adverse_pct = float(adverse_path[bar_n - 1])
            close_pct = float(close_path[bar_n - 1])
            favorable = favorable_pct * premium
            adverse = adverse_pct * premium
            close_move = close_pct * premium
            opp_prob = p_short[i] if side == "long" else p_long[i]
            if isinstance(path_close, np.ndarray) and pos < path_close.size:
                exit_underlying_price = float(path_close[pos])
            else:
                exit_underlying_price = float(close[i])
        else:
            if isinstance(path_high, np.ndarray) and isinstance(path_low, np.ndarray) and isinstance(path_close, np.ndarray):
                bar_high = float(path_high[pos])
                bar_low = float(path_low[pos])
                bar_close = float(path_close[pos])
            else:
                bar_high = float(high[i])
                bar_low = float(low[i])
                bar_close = float(close[i])
            exit_underlying_price = bar_close
            if side == "long":
                favorable = bar_high - entry
                adverse = bar_low - entry
                close_move = bar_close - entry
                opp_prob = p_short[i]
            else:
                favorable = entry - bar_low
                adverse = entry - bar_high
                close_move = entry - bar_close
                opp_prob = p_long[i]

        if np.isfinite(favorable) and favorable > best_move:
            best_move = float(favorable)
            last_peak_bar = bar_n
        if np.isfinite(adverse):
            worst_move = min(worst_move, float(adverse))
        final_move = float(close_move) if np.isfinite(close_move) else final_move
        best_pct = best_move / premium

        if adverse <= -stop_loss * premium:
            exit_reason = "stop_loss"
            exit_idx = int(i)
            exit_pos = pos
            exit_move = -stop_loss * premium
            break
        if tp is not None and best_pct >= float(tp):
            exit_reason = "take_profit"
            exit_idx = int(i)
            exit_pos = pos
            exit_move = float(tp) * premium
            break

        if str(spec["style"]) in {
            "confidence_trail",
            "mfe_decay_vol",
            "regime_trail",
            "hybrid",
            "late_trail",
            "confidence_late_trail",
            "vol_late_trail",
            "late_hybrid",
        }:
            arm, giveback = _dynamic_params(spec, trade, best_pct)
            if best_pct >= arm:
                trail_move = _trail_exit_move(best_move, giveback, premium)
                if adverse <= trail_move:
                    exit_reason = "adaptive_trail"
                    exit_idx = int(i)
                    exit_pos = pos
                    exit_move = trail_move
                    break

        if str(spec["style"]) == "hybrid":
            if (
                bar_n - last_peak_bar >= _effective_stale_bars(spec, trade)
                and best_pct >= float(spec["progress"])
                and final_move / premium < best_pct * 0.50
            ):
                exit_reason = "time_decay"
                exit_idx = int(i)
                exit_pos = pos
                exit_move = final_move
                break
            if np.isfinite(opp_prob) and opp_prob >= float(spec["opp_thr"]) and best_pct >= 1.0:
                exit_reason = "opposite_signal"
                exit_idx = int(i)
                exit_pos = pos
                exit_move = final_move
                break
        if str(spec["style"]) == "late_hybrid":
            if (
                bar_n - last_peak_bar >= _effective_stale_bars(spec, trade)
                and best_pct < float(spec["progress"])
                and final_move / premium <= 0.0
            ):
                exit_reason = "time_decay"
                exit_idx = int(i)
                exit_pos = pos
                exit_move = final_move
                break
            if np.isfinite(opp_prob) and opp_prob >= float(spec["opp_thr"]) and best_pct >= 1.0:
                exit_reason = "opposite_signal"
                exit_idx = int(i)
                exit_pos = pos
                exit_move = final_move
                break
    else:
        exit_move = final_move

    if isinstance(path_times, pd.DatetimeIndex) and len(path_times) > exit_pos:
        exit_time = path_times[exit_pos]
    else:
        exit_time = index[exit_idx]

    return {
        "regime": str(spec["name"]),
        "exit_style": str(spec["style"]),
        "side": side,
        "entry_time": str(trade["entry_time"]),
        "exit_time": str(exit_time),
        "entry_price": float(entry),
        "exit_price": float(exit_underlying_price),
        "entry_premium_proxy": float(premium),
        "peak_premium_proxy": float(premium + best_move),
        "exit_premium_proxy": float(premium + exit_move),
        "entry_prob": float(trade["entry_prob"]),
        "entry_opp_prob": float(trade["entry_opp_prob"]),
        "trend_aligned": bool(trade["trend_aligned"]),
        "trend_spread_atr": float(trade["trend_spread_atr"]),
        "vol_ratio": float(trade["vol_ratio"]),
        "proxy_mode": proxy_mode,
        "strike": float(trade.get("strike", np.nan)),
        "entry_iv": float(trade.get("entry_iv", np.nan)),
        "trail_giveback_basis": "peak_profit",
        "effective_stale_bars": _effective_stale_bars(spec, trade) if "stale_bars" in spec else np.nan,
        "stop_loss_pct": stop_loss,
        "outcome_pct": float(exit_move / premium),
        "mfe_pct": float(best_move / premium),
        "mae_pct": float(worst_move / premium),
        "exit_reason": exit_reason,
    }


def _summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if events.empty:
        return pd.DataFrame()
    group_cols = ["regime"]
    if "position_mode" in events.columns:
        group_cols = ["position_mode", "regime"]
    grouped = events.groupby(group_cols, sort=False)
    for group_key, group in grouped:
        if isinstance(group_key, tuple):
            position_mode, regime = group_key
        else:
            position_mode, regime = None, group_key
        outcome = pd.to_numeric(group["outcome_pct"], errors="coerce")
        row: dict[str, object] = {
            "position_mode": position_mode,
            "regime": regime,
            "exit_style": str(group["exit_style"].iloc[0]),
            "trades": int(len(group)),
            "candidate_trades": int(pd.to_numeric(group.get("candidate_trades"), errors="coerce").iloc[0])
            if "candidate_trades" in group.columns and len(group)
            else int(len(group)),
            "skipped_overlap_candidates": int(
                pd.to_numeric(group.get("skipped_overlap_candidates"), errors="coerce").iloc[0]
            )
            if "skipped_overlap_candidates" in group.columns and len(group)
            else 0,
            "mean_outcome_pct": float(outcome.mean()),
            "median_outcome_pct": float(outcome.median()),
            "total_outcome_pct": float(outcome.sum()),
            "win_rate": float((outcome > 0).mean()),
            "p90_outcome_pct": float(outcome.quantile(0.90)),
            "p95_outcome_pct": float(outcome.quantile(0.95)),
            "p99_outcome_pct": float(outcome.quantile(0.99)),
            "avg_mfe_pct": float(pd.to_numeric(group["mfe_pct"], errors="coerce").mean()),
        }
        for reason, count in group["exit_reason"].value_counts().items():
            row[f"{reason}_count"] = int(count)
        for side in ("long", "short"):
            side_group = group[group["side"].eq(side)]
            side_outcome = pd.to_numeric(side_group["outcome_pct"], errors="coerce")
            row[f"{side}_trades"] = int(len(side_group))
            row[f"{side}_mean_outcome_pct"] = float(side_outcome.mean()) if len(side_group) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).fillna(
        {
            key: 0
            for key in [
                "stop_loss_count",
                "take_profit_count",
                "adaptive_trail_count",
                "time_decay_count",
                "opposite_signal_count",
                "timeout_count",
            ]
        }
    )
    return out.sort_values(
        ["mean_outcome_pct", "p95_outcome_pct", "win_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _parse_position_modes(raw: str) -> list[str]:
    modes: list[str] = []
    aliases = {
        "hedge": "hedged",
        "hedged": "hedged",
        "current": "hedged",
        "both": "hedged",
        "single": "single",
        "single_position": "single",
        "one": "single",
        "one_position": "single",
    }
    for part in str(raw or "hedged").split(","):
        key = part.strip().lower()
        if not key:
            continue
        mode = aliases.get(key)
        if mode is None:
            raise ValueError(f"Unknown position mode {part!r}; expected hedged or single")
        if mode not in modes:
            modes.append(mode)
    return modes or ["hedged"]


def _simulate_strategy_events(
    market: dict[str, object],
    trades: list[dict[str, object]],
    spec: dict[str, object],
    *,
    position_mode: str,
) -> list[dict[str, object]]:
    ordered = sorted(trades, key=lambda trade: pd.Timestamp(trade["entry_time"]))
    rows: list[dict[str, object]] = []
    blocked = 0
    open_until: pd.Timestamp | None = None

    for trade in ordered:
        entry_time = pd.Timestamp(trade["entry_time"])
        if position_mode == "single" and open_until is not None and entry_time < open_until:
            blocked += 1
            continue

        result = _simulate_trade(market, trade, spec)
        if result is None:
            continue
        result["position_mode"] = position_mode
        rows.append(result)
        if position_mode == "single":
            open_until = pd.Timestamp(result["exit_time"])

    for result in rows:
        result["candidate_trades"] = len(ordered)
        result["skipped_overlap_candidates"] = blocked
    return rows


def main() -> None:
    args = _parse_args()
    feature_df = pd.read_parquet(REPO_ROOT / args.signal_frame).sort_index()
    try:
        selected = _load_scoreboard(REPO_ROOT / args.scoreboard, args.split, args.variant, args.mode)
    except ValueError:
        if str(args.split) == "oof":
            raise
        selected = _load_scoreboard(REPO_ROOT / args.scoreboard, "oof", args.variant, args.mode)
        selected = selected.copy()
        selected["split"] = f"oof_thresholds_for_{args.split}"

    long_threshold = float(args.long_setup_threshold if args.long_setup_threshold is not None else selected["long_setup_threshold"])
    short_threshold = float(args.short_setup_threshold if args.short_setup_threshold is not None else selected["short_setup_threshold"])
    cooldown_bars = int(args.cooldown_bars if args.cooldown_bars is not None else selected["cooldown_bars"])
    post_setup_max_bars = int(
        args.post_setup_max_bars if args.post_setup_max_bars is not None else selected.get("post_setup_max_bars", 4)
    )
    one_per_setup_cluster = bool(selected.get("one_per_setup_cluster", False))
    execution_1m = _load_execution_1m(args, feature_df.index)
    long_entries, short_entries, long_prices, short_prices, long_times, short_times = _build_entries(
        feature_df,
        selected,
        split=str(args.split),
        long_setup_threshold=long_threshold,
        short_setup_threshold=short_threshold,
        cooldown_bars=cooldown_bars,
        post_setup_max_bars=post_setup_max_bars,
        one_per_setup_cluster=one_per_setup_cluster,
        execution_1m=execution_1m,
    )
    trades = _build_trade_specs(
        feature_df,
        split=str(args.split),
        long_entries=long_entries,
        short_entries=short_entries,
        long_prices=long_prices,
        short_prices=short_prices,
        long_times=long_times,
        short_times=short_times,
        premium_atr_mult=float(args.premium_atr_mult),
        exit_hhmm=str(args.exit_hhmm),
        horizon_bars=int(args.horizon_bars),
        proxy_mode=str(args.proxy_mode),
        bs_iv_floor=float(args.bs_iv_floor),
        bs_iv_ceil=float(args.bs_iv_ceil),
        bs_iv_mult=float(args.bs_iv_mult),
        bs_iv_shock=float(args.bs_iv_shock),
        bs_min_dte_minutes=float(args.bs_min_dte_minutes),
        bs_strike_round=float(args.bs_strike_round),
        execution_1m=execution_1m,
    )
    p_suffix = "oof_train" if str(args.split) == "oof" else "test"
    market = {
        "high": pd.to_numeric(feature_df["high"], errors="coerce").to_numpy(dtype=float),
        "low": pd.to_numeric(feature_df["low"], errors="coerce").to_numpy(dtype=float),
        "close": pd.to_numeric(feature_df["close"], errors="coerce").to_numpy(dtype=float),
        "p_long": pd.to_numeric(feature_df[f"p_long_{p_suffix}"], errors="coerce").to_numpy(dtype=float),
        "p_short": pd.to_numeric(feature_df[f"p_short_{p_suffix}"], errors="coerce").to_numpy(dtype=float),
        "index": feature_df.index,
    }
    specs = _strategy_specs(
        _parse_float_list(args.stop_loss_values),
        late_hybrid_stale_bars=[int(value) for value in _parse_float_list(args.late_hybrid_stale_bars)],
        late_hybrid_progress_values=_parse_float_list(args.late_hybrid_progress_values),
        late_hybrid_opp_thresholds=_parse_float_list(args.late_hybrid_opp_thresholds),
        late_hybrid_arm_values=_parse_float_list(args.late_hybrid_arm_values),
        late_hybrid_giveback_values=_parse_float_list(args.late_hybrid_giveback_values),
    )
    if str(args.strategy_styles).strip():
        allowed_styles = {value.strip() for value in str(args.strategy_styles).split(",") if value.strip()}
        specs = [spec for spec in specs if str(spec["style"]) in allowed_styles]
    position_modes = _parse_position_modes(args.position_modes)
    event_rows: list[dict[str, object]] = []
    for position_mode in position_modes:
        for spec in specs:
            event_rows.extend(_simulate_strategy_events(market, trades, spec, position_mode=position_mode))
    events = pd.DataFrame(event_rows)
    summary = _summarize(events)

    out_path = REPO_ROOT / args.out
    events_path = REPO_ROOT / args.events_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    events.to_csv(events_path, index=False)

    print(
        "selected="
        f"{selected['split']} {selected['variant']} {selected['mode']} "
        f"proxy={args.proxy_mode} horizon_bars={int(args.horizon_bars)} "
        f"entries_long={int(long_entries.sum())} entries_short={int(short_entries.sum())} "
        f"trades={len(trades)} strategies={len(specs)}"
    )
    show_cols = [
        "position_mode",
        "regime",
        "exit_style",
        "trades",
        "candidate_trades",
        "skipped_overlap_candidates",
        "mean_outcome_pct",
        "median_outcome_pct",
        "win_rate",
        "p95_outcome_pct",
        "stop_loss_count",
        "take_profit_count",
        "adaptive_trail_count",
        "time_decay_count",
        "opposite_signal_count",
        "timeout_count",
        "long_mean_outcome_pct",
        "short_mean_outcome_pct",
    ]
    print(summary[[col for col in show_cols if col in summary.columns]].head(25).to_string(index=False))
    print(f"summary_csv={out_path}")
    print(f"events_csv={events_path}")


if __name__ == "__main__":
    main()
