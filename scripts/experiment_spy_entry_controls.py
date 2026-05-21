from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NY_TZ = "America/New_York"


@dataclass(frozen=True)
class Signal:
    run: str
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    atr: float
    p_enter_long: float
    selected_action_class: int


@dataclass(frozen=True)
class Entry:
    variant: str
    signal: Signal
    entry_ts: pd.Timestamp | None
    entry_price: float | None
    status: str
    reason: str
    delay_min: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay SPY 1m entry controls against live decision logs.")
    parser.add_argument("--live-root", default="Data/inference/live_runs")
    parser.add_argument("--one-min", default="Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet")
    parser.add_argument("--out-dir", default="Data/inference/spy/entry_control_experiment_20260520")
    parser.add_argument("--max-confirm-min", type=int, default=40)
    parser.add_argument("--forward-min", type=int, default=60)
    parser.add_argument("--min-prob", type=float, default=0.35)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _to_utc(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def _load_signals(live_root: Path, *, min_prob: float) -> list[Signal]:
    signals: list[Signal] = []
    for decision_path in sorted(live_root.glob("*_live_spy/decision-10m.jsonl")):
        run = decision_path.parent.name
        for row in _read_jsonl(decision_path):
            payload = row.get("payload") or {}
            bar = payload.get("bar") or {}
            action = payload.get("action") or {}
            try:
                p_enter_long = float(bar.get("p_enter_long", np.nan))
                selected = int(action.get("selected_action_class", 0))
                if selected != 1 or not np.isfinite(p_enter_long) or p_enter_long < min_prob:
                    continue
                ts = _to_utc(bar["timestamp"])
                policy_result = payload.get("policy_result") or {}
                signals.append(
                    Signal(
                        run=run,
                        ts=ts,
                        open=float(bar.get("open", np.nan)),
                        high=float(bar.get("high", np.nan)),
                        low=float(bar.get("low", np.nan)),
                        close=float(bar.get("close", np.nan)),
                        atr=float(policy_result.get("atr", np.nan)),
                        p_enter_long=p_enter_long,
                        selected_action_class=selected,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    dedup: dict[tuple[str, pd.Timestamp], Signal] = {}
    for signal in signals:
        dedup[(signal.run, signal.ts)] = signal
    return sorted(dedup.values(), key=lambda s: (s.ts, s.run))


def _load_one_min(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    local = df["timestamp"].dt.tz_convert(NY_TZ)
    df["session"] = local.dt.date.astype(str)
    df["local_time"] = local.dt.strftime("%H:%M")
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"].fillna(0.0)
    df["vwap"] = pv.groupby(df["session"]).cumsum() / df["volume"].fillna(0.0).groupby(df["session"]).cumsum().replace(0, np.nan)
    df["ema8"] = df.groupby("session", group_keys=False)["close"].apply(lambda s: s.ewm(span=8, adjust=False, min_periods=3).mean())
    df["session_high_so_far"] = df.groupby("session")["high"].cummax()
    df["session_low_so_far"] = df.groupby("session")["low"].cummin()
    first30 = df[df["local_time"].between("09:30", "09:59")].groupby("session").agg(or_high=("high", "max"), or_low=("low", "min"))
    first30["or_mid"] = (first30["or_high"] + first30["or_low"]) / 2.0
    df = df.merge(first30[["or_mid"]], left_on="session", right_index=True, how="left")
    return df.set_index("timestamp", drop=False)


def _window(one_min: pd.DataFrame, signal: Signal, *, minutes: int) -> pd.DataFrame:
    start = signal.ts + pd.Timedelta(minutes=1)
    end = signal.ts + pd.Timedelta(minutes=minutes)
    return one_min.loc[start:end].copy()


def _is_baseline_confirm(row: pd.Series, ref: float) -> bool:
    return bool(row["high"] >= ref and row["close"] > ref and row["close"] > row["open"])


def _first_baseline(signal: Signal, bars: pd.DataFrame) -> tuple[pd.Series | None, str]:
    ref = signal.high
    for _, row in bars.iterrows():
        if _is_baseline_confirm(row, ref):
            return row, "bodyclose_breakout"
    return None, "no_1m_confirmation"


def _entry_from_row(variant: str, signal: Signal, row: pd.Series | None, status: str, reason: str) -> Entry:
    if row is None:
        return Entry(variant, signal, None, None, status, reason, None)
    ts = pd.Timestamp(row["timestamp"])
    return Entry(
        variant=variant,
        signal=signal,
        entry_ts=ts,
        entry_price=float(row["close"]),
        status=status,
        reason=reason,
        delay_min=(ts - signal.ts).total_seconds() / 60.0,
    )


def _no_chase(signal: Signal, row: pd.Series, *, limit_atr: float, ref_kind: str) -> bool:
    atr = signal.atr
    if not np.isfinite(atr) or atr <= 0:
        return True
    ref = signal.high if ref_kind == "breakout_ref" else signal.close
    return float(row["close"]) - ref <= limit_atr * atr


def _anti_fomo(signal: Signal, row: pd.Series, *, high_buffer_atr: float) -> bool:
    local = pd.Timestamp(row["timestamp"]).tz_convert(NY_TZ)
    if local.time() >= pd.Timestamp("10:00", tz=NY_TZ).time():
        return True
    atr = signal.atr
    if not np.isfinite(atr) or atr <= 0:
        return True
    distance_from_high = float(row["session_high_so_far"]) - float(row["close"])
    return distance_from_high >= high_buffer_atr * atr


def _opening_pullback(signal: Signal, bars: pd.DataFrame, *, target_kind: str, stretched_atr: float) -> tuple[pd.Series | None, str]:
    base_row, _ = _first_baseline(signal, bars)
    if base_row is None:
        return None, "no_initial_breakout"
    atr = signal.atr
    stretched = np.isfinite(atr) and atr > 0 and float(base_row["close"]) - signal.high > stretched_atr * atr
    local_signal = signal.ts.tz_convert(NY_TZ)
    if local_signal.strftime("%H:%M") != "09:30" or not stretched:
        return base_row, "baseline_not_stretched_open"

    pulled_back = False
    for _, row in bars[bars["timestamp"] > base_row["timestamp"]].iterrows():
        if target_kind == "vwap":
            target = float(row["vwap"])
        elif target_kind == "ema8":
            target = float(row["ema8"])
        else:
            target = float(row["or_mid"])
        if not np.isfinite(target):
            continue
        if not pulled_back and float(row["low"]) <= target:
            pulled_back = True
            continue
        if pulled_back and float(row["close"]) > target and float(row["close"]) > float(row["open"]):
            return row, f"pullback_reclaim_{target_kind}"
    return None, f"no_pullback_reclaim_{target_kind}"


def _evaluate_signal(signal: Signal, bars: pd.DataFrame) -> list[Entry]:
    entries: list[Entry] = []
    base_row, base_reason = _first_baseline(signal, bars)
    entries.append(_entry_from_row("baseline_bodyclose", signal, base_row, "entered" if base_row is not None else "skipped", base_reason))
    if base_row is None:
        for name in [
            "no_chase_ref_035",
            "no_chase_ref_050",
            "no_chase_setupclose_035",
            "no_chase_setupclose_050",
            "anti_fomo_010",
            "anti_fomo_020",
            "combined_ref035_fomo010",
            "opening_pullback_vwap",
            "opening_pullback_ema8",
            "opening_pullback_ormid",
        ]:
            entries.append(_entry_from_row(name, signal, None, "skipped", base_reason))
        return entries

    variants = [
        ("no_chase_ref_035", _no_chase(signal, base_row, limit_atr=0.35, ref_kind="breakout_ref"), "too_far_above_breakout_ref"),
        ("no_chase_ref_050", _no_chase(signal, base_row, limit_atr=0.50, ref_kind="breakout_ref"), "too_far_above_breakout_ref"),
        ("no_chase_setupclose_035", _no_chase(signal, base_row, limit_atr=0.35, ref_kind="setup_close"), "too_far_above_setup_close"),
        ("no_chase_setupclose_050", _no_chase(signal, base_row, limit_atr=0.50, ref_kind="setup_close"), "too_far_above_setup_close"),
        ("anti_fomo_010", _anti_fomo(signal, base_row, high_buffer_atr=0.10), "near_opening_move_high"),
        ("anti_fomo_020", _anti_fomo(signal, base_row, high_buffer_atr=0.20), "near_opening_move_high"),
    ]
    for name, allowed, veto_reason in variants:
        entries.append(_entry_from_row(name, signal, base_row if allowed else None, "entered" if allowed else "vetoed", base_reason if allowed else veto_reason))
    combined_allowed = _no_chase(signal, base_row, limit_atr=0.35, ref_kind="breakout_ref") and _anti_fomo(signal, base_row, high_buffer_atr=0.10)
    entries.append(_entry_from_row("combined_ref035_fomo010", signal, base_row if combined_allowed else None, "entered" if combined_allowed else "vetoed", base_reason if combined_allowed else "combined_veto"))

    for target in ["vwap", "ema8", "ormid"]:
        row, reason = _opening_pullback(signal, bars, target_kind=target, stretched_atr=0.35)
        entries.append(_entry_from_row(f"opening_pullback_{target}", signal, row, "entered" if row is not None else "skipped", reason))
    return entries


def _forward_metrics(one_min: pd.DataFrame, entry: Entry, *, forward_min: int) -> dict[str, float | None]:
    if entry.entry_ts is None or entry.entry_price is None:
        return {"ret_10m": None, "ret_30m": None, "ret_60m": None, "mfe_30m": None, "mae_30m": None}
    start = entry.entry_ts
    end = start + pd.Timedelta(minutes=forward_min)
    rows = one_min.loc[start + pd.Timedelta(minutes=1) : end]
    out: dict[str, float | None] = {}
    for minutes in [10, 30, 60]:
        target_ts = start + pd.Timedelta(minutes=minutes)
        sub = one_min.loc[start + pd.Timedelta(minutes=1) : target_ts]
        out[f"ret_{minutes}m"] = None if sub.empty else (float(sub.iloc[-1]["close"]) - entry.entry_price) / entry.entry_price
    first30 = rows[rows["timestamp"] <= start + pd.Timedelta(minutes=30)]
    if first30.empty:
        out["mfe_30m"] = None
        out["mae_30m"] = None
    else:
        out["mfe_30m"] = (float(first30["high"].max()) - entry.entry_price) / entry.entry_price
        out["mae_30m"] = (float(first30["low"].min()) - entry.entry_price) / entry.entry_price
    return out


def _event_rows(signals: list[Signal], one_min: pd.DataFrame, *, max_confirm_min: int, forward_min: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal in signals:
        bars = _window(one_min, signal, minutes=max_confirm_min)
        if bars.empty:
            continue
        for entry in _evaluate_signal(signal, bars):
            metric = _forward_metrics(one_min, entry, forward_min=forward_min)
            rows.append(
                {
                    "variant": entry.variant,
                    "run": signal.run,
                    "signal_ts": signal.ts.isoformat(),
                    "signal_local": signal.ts.tz_convert(NY_TZ).isoformat(),
                    "signal_close": signal.close,
                    "signal_high": signal.high,
                    "signal_atr": signal.atr,
                    "p_enter_long": signal.p_enter_long,
                    "entry_ts": entry.entry_ts.isoformat() if entry.entry_ts is not None else None,
                    "entry_local": entry.entry_ts.tz_convert(NY_TZ).isoformat() if entry.entry_ts is not None else None,
                    "entry_price": entry.entry_price,
                    "status": entry.status,
                    "reason": entry.reason,
                    "delay_min": entry.delay_min,
                    **metric,
                }
            )
    return pd.DataFrame(rows)


def _summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in events.groupby("variant"):
        entered = group[group["status"] == "entered"]
        rows.append(
            {
                "variant": variant,
                "signals": int(group["signal_ts"].nunique()),
                "entered": int(len(entered)),
                "vetoed": int((group["status"] == "vetoed").sum()),
                "skipped": int((group["status"] == "skipped").sum()),
                "entry_rate": float(len(entered) / max(1, group["signal_ts"].nunique())),
                "avg_delay_min": float(entered["delay_min"].mean()) if not entered.empty else np.nan,
                "avg_ret_10m_bp": float(entered["ret_10m"].mean() * 10000) if not entered.empty else np.nan,
                "avg_ret_30m_bp": float(entered["ret_30m"].mean() * 10000) if not entered.empty else np.nan,
                "avg_ret_60m_bp": float(entered["ret_60m"].mean() * 10000) if not entered.empty else np.nan,
                "win_30m": float((entered["ret_30m"] > 0).mean()) if not entered.empty else np.nan,
                "avg_mfe_30m_bp": float(entered["mfe_30m"].mean() * 10000) if not entered.empty else np.nan,
                "avg_mae_30m_bp": float(entered["mae_30m"].mean() * 10000) if not entered.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_ret_30m_bp", "avg_mae_30m_bp"], ascending=[False, False])


def _execution_summary(live_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    submit_re = re.compile(r"ORDER SUBMITTED intent=open .*?symbol=(?P<symbol>\S+) .*?limit_price=(?P<limit>[0-9.]+) attempt=(?P<attempt>\d+)/(?P<max_attempts>\d+)")
    retry_re = re.compile(r"ORDER RETRY intent=open .*?symbol=(?P<symbol>\S+) .*?last_limit=(?P<last>[0-9.]+) next_limit=(?P<next>[0-9.]+)")
    for run_dir in sorted(live_root.glob("*_live_spy")):
        run = run_dir.name
        policy_path = run_dir / "order-policy.jsonl"
        log_path = run_dir / "logs.jsonl"
        entry_limits: list[float] = []
        retries = 0
        symbols: set[str] = set()
        pending_first: pd.Timestamp | None = None
        pending_last: pd.Timestamp | None = None
        pending_reconcile_count = 0
        for row in _read_jsonl(log_path):
            msg = ((row.get("payload") or {}).get("message") or "")
            submit_match = submit_re.search(msg)
            if submit_match:
                symbols.add(submit_match.group("symbol"))
                entry_limits.append(float(submit_match.group("limit")))
                continue
            retry_match = retry_re.search(msg)
            if retry_match:
                symbols.add(retry_match.group("symbol"))
                retries += 1
        for row in _read_jsonl(policy_path):
            payload = row.get("payload") or {}
            recorded_at = _to_utc(row.get("recorded_at"))
            state = payload.get("policy_state") or {}
            if state.get("pending_broker_reconcile"):
                pending_reconcile_count += 1
                pending_first = recorded_at if pending_first is None else min(pending_first, recorded_at)
                pending_last = recorded_at if pending_last is None else max(pending_last, recorded_at)
        if entry_limits or pending_reconcile_count:
            rows.append(
                {
                    "run": run,
                    "symbols": ",".join(sorted(symbols)),
                    "open_long_submissions": len(entry_limits),
                    "open_long_retries": retries,
                    "min_entry_limit": min(entry_limits) if entry_limits else np.nan,
                    "max_entry_limit": max(entry_limits) if entry_limits else np.nan,
                    "entry_limit_raise": (max(entry_limits) - min(entry_limits)) if entry_limits else np.nan,
                    "pending_reconcile_records": pending_reconcile_count,
                    "pending_reconcile_seconds": (pending_last - pending_first).total_seconds() if pending_first is not None and pending_last is not None else 0,
                }
            )
    return pd.DataFrame(rows).sort_values("run")


def main() -> None:
    args = _parse_args()
    live_root = Path(args.live_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    one_min = _load_one_min(Path(args.one_min))
    signals = _load_signals(live_root, min_prob=args.min_prob)
    events = _event_rows(signals, one_min, max_confirm_min=args.max_confirm_min, forward_min=args.forward_min)
    summary = _summarize(events)
    execution = _execution_summary(live_root)

    events.to_csv(out_dir / "entry_control_events.csv", index=False)
    summary.to_csv(out_dir / "entry_control_summary.csv", index=False)
    execution.to_csv(out_dir / "execution_state_summary.csv", index=False)

    print(f"signals={len(signals)} events={len(events)}")
    print(f"wrote={out_dir}")
    print(summary.to_string(index=False))
    if not execution.empty:
        print("\nexecution_state")
        print(execution.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
