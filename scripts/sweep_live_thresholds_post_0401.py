from __future__ import annotations

import argparse
import json
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

from Policy.replay_option_proxy import ReplayOptionPriceProxy


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
        refresh = valid and (not above[side] or not math.isfinite(peak[side]) or prob > peak[side])
        if refresh and math.isfinite(ref):
            setup_ts = row["timestamp"]
            pending[side] = PendingSetup(
                side=side,
                ref=float(ref),
                setup_ts=setup_ts,
                start_ts=setup_ts + pd.Timedelta(minutes=10),
                expires_at=setup_ts + pd.Timedelta(minutes=10 * (setup_max_bars + 1)),
                prob=float(prob),
            )
            peak[side] = float(prob)
        above[side] = bool(prob >= threshold)


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
        "symbol": pos.symbol,
    }


def _run_one(
    *,
    decisions: pd.DataFrame,
    one_min: pd.DataFrame,
    long_thr: float,
    short_thr: float,
    setup_max_bars: int,
    cutoff_hhmm: str,
) -> list[dict[str, Any]]:
    proxy = ReplayOptionPriceProxy(
        tz_name="America/New_York",
        expiry_hhmm="15:40",
        iv_floor=0.12,
        iv_ceiling=0.90,
        iv_multiplier=1.50,
        min_dte_minutes=1.0,
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
            triggered = [setup for setup in live_setups if _triggered(setup, pd.Series(row))]
            if triggered:
                if len(triggered) > 1:
                    triggered.sort(key=lambda s: s.prob - (long_thr if s.side == "long" else short_thr), reverse=True)
                setup = triggered[0]
                right = "C" if setup.side == "long" else "P"
                strike = round(close + latest_atr) if setup.side == "long" else round(close - latest_atr)
                symbol = _sim_symbol("SPY", ts, right, strike, cutoff_hhmm)
                premium = proxy.price(symbol, mode="mid")
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
                        threshold=long_thr if setup.side == "long" else short_thr,
                    )
                    pending[setup.side] = None

    if pos is not None and len(one_min):
        last = one_min.iloc[-1]
        premium = proxy.price(pos.symbol, mode="mid")
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
    }


def _parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep post-04/01 live setup probability thresholds.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--one-min", default=str(DEFAULT_ONE_MIN))
    parser.add_argument("--start", default="2026-04-01T00:00:00-04:00")
    parser.add_argument("--long-grid", default="0.30,0.35,0.40,0.42,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--short-grid", default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--setup-max-bars", type=int, default=4)
    parser.add_argument("--cutoff-hhmm", default="13:00")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--events-out", default=str(DEFAULT_EVENTS_OUT))
    args = parser.parse_args()

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
    for long_thr in _parse_grid(args.long_grid):
        for short_thr in _parse_grid(args.short_grid):
            events = _run_one(
                decisions=decisions,
                one_min=one_min,
                long_thr=long_thr,
                short_thr=short_thr,
                setup_max_bars=int(args.setup_max_bars),
                cutoff_hhmm=str(args.cutoff_hhmm),
            )
            summary = {
                "long_threshold": long_thr,
                "short_threshold": short_thr,
                "decision_rows": int(len(decisions)),
                "first_decision": decisions["timestamp"].min(),
                "last_decision": decisions["timestamp"].max(),
                **_metrics(events),
            }
            summaries.append(summary)
            for event in events:
                all_events.append({"long_threshold": long_thr, "short_threshold": short_thr, **event})

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
