"""Build a unified trades dataset from multi-ticker swing live audit logs.

Reads UI/swing_audit/swing_session_*.jsonl files in a date window and produces a
single parquet/CSV with one row per trade lifecycle, joining:
  - signal-time context (p_dir, ev_score, ref_high/low, atr, signal_ts)
  - confirmation context (bars_watched_to_confirm, confirmation close)
  - fresh open context (entry_price, entry_time, direction, tier, atr_at_entry,
    sl_price, option_symbol)
  - close context (exit_price, exit_reason, bars_held, option_last_price, etc.)

Cross-session chaining: when a fresh open in session N has no same-session close,
we chase it forward through later sessions by matching (ticker, option_symbol) to
restored opens, and pick up the eventual close from whichever session it lands in.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

_ET = ZoneInfo("America/New_York")

# OCC option symbol pattern: ROOT + YYMMDD + C/P + 8-digit strike (strike * 1000)
_OCC_RE = re.compile(r"^([A-Z\.]+?)(\d{6})([CP])(\d{8})$")


@dataclass
class FreshOpen:
    session_file: str
    ticker: str
    entry_time_iso: str        # original entry_time stamped by open event
    entry_price: float
    direction: int
    tier: int | None
    atr_at_entry: float | None
    sl_price: float | None
    option_symbol: str | None
    option_entry_price: float | None
    qty: int
    open_event_ts: str         # event ts (wall clock when event was logged)
    # signal context joined later:
    signal_ts: str | None = None
    p_dir: float | None = None
    ev_score: float | None = None
    ref_high: float | None = None
    ref_low: float | None = None
    signal_atr: float | None = None
    confirm_bars_watched: int | None = None
    confirm_close: float | None = None
    confirm_event_ts: str | None = None


@dataclass
class CloseEvent:
    session_file: str
    ticker: str
    matched_entry_time_iso: str
    exit_price: float | None
    exit_pnl_pct: float | None
    exit_reason: str | None
    bars_held: int | None
    option_last_price: float | None
    option_best_price: float | None
    best_price: float | None
    trail_armed: bool | None
    deferred_trail_active: bool | None
    deferred_trail_trigger_pnl_pct: float | None
    close_event_ts: str
    order_error: str | None
    option_symbol: str | None


def _iter_events(path: str):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _parse_occ(sym: str | None) -> tuple[str | None, datetime | None, str | None, float | None]:
    if not sym:
        return None, None, None, None
    m = _OCC_RE.match(sym.strip())
    if not m:
        return None, None, None, None
    root, yymmdd, cp, strike8 = m.groups()
    try:
        expiry = datetime.strptime(yymmdd, "%y%m%d").date()
    except Exception:
        expiry = None
    try:
        strike = int(strike8) / 1000.0
    except Exception:
        strike = None
    return root, expiry, cp, strike


def collect_sessions(audit_dir: str, since_basename: str) -> list[str]:
    return sorted([
        f for f in glob.glob(os.path.join(audit_dir, "swing_session_*.jsonl"))
        if os.path.basename(f) >= since_basename
    ])


def build_dataset(audit_dir: str, since: str) -> pd.DataFrame:
    files = collect_sessions(audit_dir, f"swing_session_{since}")
    print(f"Sessions in window: {len(files)}")

    # Per-session collections
    sessions_data = {}  # path -> dict
    for f in files:
        restored_keys: set[tuple[str, str]] = set()   # (ticker, entry_time_iso)
        opens: list[dict] = []
        closes: list[dict] = []
        signals: list[dict] = []             # by ticker, in time order
        confirmations: list[dict] = []
        for rec in _iter_events(f):
            t = rec.get("type")
            p = rec.get("payload") or {}
            ts = rec.get("ts")
            if t == "broker_sync":
                for pos in (p.get("positions") or []):
                    if isinstance(pos, dict):
                        restored_keys.add((pos.get("ticker"), pos.get("entry_time")))
            elif t == "position_opened":
                opens.append({**p, "_event_ts": ts})
            elif t == "position_closed":
                closes.append({**p, "_event_ts": ts})
            elif t == "signal":
                signals.append({**p, "_event_ts": ts})
            elif t == "confirmation":
                confirmations.append({**p, "_event_ts": ts})
        sessions_data[f] = dict(
            restored_keys=restored_keys,
            opens=opens,
            closes=closes,
            signals=signals,
            confirmations=confirmations,
        )

    # Build fresh-open records (per session) + join signal/confirmation
    fresh_opens: list[FreshOpen] = []
    for f, s in sessions_data.items():
        # index signals/confirmations by ticker for quick lookup of latest preceding
        sigs_by_tkr = defaultdict(list)
        for sig in s["signals"]:
            sigs_by_tkr[sig.get("ticker")].append(sig)
        for v in sigs_by_tkr.values():
            v.sort(key=lambda r: r.get("_event_ts") or "")
        confs_by_tkr = defaultdict(list)
        for c in s["confirmations"]:
            confs_by_tkr[c.get("ticker")].append(c)
        for v in confs_by_tkr.values():
            v.sort(key=lambda r: r.get("_event_ts") or "")

        for op in s["opens"]:
            key = (op.get("ticker"), op.get("entry_time"))
            if key in s["restored_keys"]:
                continue  # restored — handled by chaining
            fo = FreshOpen(
                session_file=os.path.basename(f),
                ticker=op.get("ticker"),
                entry_time_iso=op.get("entry_time"),
                entry_price=float(op.get("entry_price") or float("nan")),
                direction=int(op.get("direction") or 0),
                tier=op.get("tier"),
                atr_at_entry=op.get("atr_at_entry"),
                sl_price=op.get("sl_price"),
                option_symbol=op.get("option_symbol"),
                option_entry_price=op.get("option_entry_price"),
                qty=int(op.get("qty") or 1),
                open_event_ts=op.get("_event_ts"),
            )
            # Latest preceding signal for this ticker (within 35min — one 30m bar + buffer)
            open_dt = _parse_iso(fo.open_event_ts)
            if open_dt is not None:
                for sig in reversed(sigs_by_tkr.get(fo.ticker, [])):
                    sdt = _parse_iso(sig.get("_event_ts"))
                    if sdt is None:
                        continue
                    delta = (open_dt - sdt).total_seconds()
                    if 0 <= delta <= 35 * 60:
                        if int(sig.get("direction") or 0) == fo.direction:
                            fo.signal_ts = sig.get("_event_ts")
                            fo.p_dir = sig.get("p_dir")
                            fo.ev_score = sig.get("ev_score")
                            fo.ref_high = sig.get("ref_high")
                            fo.ref_low = sig.get("ref_low")
                            fo.signal_atr = sig.get("atr")
                            break
                # Latest preceding confirmation (within 5min)
                for c in reversed(confs_by_tkr.get(fo.ticker, [])):
                    cdt = _parse_iso(c.get("_event_ts"))
                    if cdt is None:
                        continue
                    delta = (open_dt - cdt).total_seconds()
                    if 0 <= delta <= 5 * 60:
                        if int(c.get("direction") or 0) == fo.direction:
                            fo.confirm_bars_watched = c.get("bars_watched")
                            fo.confirm_close = c.get("close")
                            fo.confirm_event_ts = c.get("_event_ts")
                            break
            fresh_opens.append(fo)

    print(f"Fresh opens collected: {len(fresh_opens)}")

    # Build close index per session by (ticker, entry_time_iso) and by (ticker, option_symbol)
    closes_by_key: dict[tuple[str, str, str], CloseEvent] = {}
    closes_by_opt: dict[tuple[str, str], CloseEvent] = {}
    for f, s in sessions_data.items():
        for cl in s["closes"]:
            ce = CloseEvent(
                session_file=os.path.basename(f),
                ticker=cl.get("ticker"),
                matched_entry_time_iso=cl.get("entry_time"),
                exit_price=cl.get("exit_price"),
                exit_pnl_pct=cl.get("exit_pnl_pct"),
                exit_reason=cl.get("exit_reason"),
                bars_held=cl.get("bars_held"),
                option_last_price=cl.get("option_last_price"),
                option_best_price=cl.get("option_best_price"),
                best_price=cl.get("best_price"),
                trail_armed=cl.get("trail_armed"),
                deferred_trail_active=cl.get("deferred_trail_active"),
                deferred_trail_trigger_pnl_pct=cl.get("deferred_trail_trigger_pnl_pct"),
                close_event_ts=cl.get("_event_ts"),
                order_error=cl.get("order_error"),
                option_symbol=cl.get("option_symbol"),
            )
            closes_by_key[(os.path.basename(f), cl.get("ticker"), cl.get("entry_time"))] = ce
            if cl.get("option_symbol"):
                # Use last close for that option as canonical (in case of dupes)
                closes_by_opt[(cl.get("ticker"), cl.get("option_symbol"))] = ce

    # Match each fresh open to its close
    rows: list[dict] = []
    same_session_match = 0
    cross_session_match = 0
    no_match = 0

    for fo in fresh_opens:
        # 1) Try same-session match by entry_time
        ce = closes_by_key.get((fo.session_file, fo.ticker, fo.entry_time_iso))
        match_type = None
        if ce is not None:
            match_type = "same_session"
            same_session_match += 1
        else:
            # 2) Try cross-session match by (ticker, option_symbol)
            if fo.option_symbol:
                ce = closes_by_opt.get((fo.ticker, fo.option_symbol))
                if ce is not None:
                    match_type = "cross_session_optsym"
                    cross_session_match += 1
        if ce is None:
            match_type = "unclosed_in_window"
            no_match += 1

        # Derive features
        entry_dt_et: datetime | None = None
        if fo.open_event_ts:
            edt = _parse_iso(fo.open_event_ts)
            if edt:
                entry_dt_et = edt.astimezone(_ET)
        # OCC parsing
        root, expiry, cp, strike = _parse_occ(fo.option_symbol)
        entry_date = entry_dt_et.date() if entry_dt_et else None
        dte = (expiry - entry_date).days if (expiry and entry_date) else None
        strike_dist_pct = None
        if strike is not None and fo.entry_price and not math.isnan(fo.entry_price):
            strike_dist_pct = (strike - fo.entry_price) / fo.entry_price
        ref_range_pct = None
        entry_vs_ref = None
        if (fo.ref_high is not None and fo.ref_low is not None
                and fo.entry_price and not math.isnan(fo.entry_price)
                and fo.ref_high > fo.ref_low):
            ref_range_pct = (fo.ref_high - fo.ref_low) / fo.entry_price
            entry_vs_ref = (fo.entry_price - fo.ref_low) / (fo.ref_high - fo.ref_low)
        signal_to_entry_secs = None
        if fo.signal_ts and fo.open_event_ts:
            sdt = _parse_iso(fo.signal_ts)
            edt = _parse_iso(fo.open_event_ts)
            if sdt and edt:
                signal_to_entry_secs = (edt - sdt).total_seconds()

        # Underlying PnL: prefer ce.exit_pnl_pct when available; else compute from prices
        underlying_pnl_pct = None
        if ce and ce.exit_pnl_pct is not None:
            underlying_pnl_pct = ce.exit_pnl_pct
        elif ce and ce.exit_price is not None and fo.entry_price and not math.isnan(fo.entry_price):
            raw = (ce.exit_price - fo.entry_price) / fo.entry_price
            underlying_pnl_pct = raw if fo.direction == 1 else -raw

        # Option PnL (only when both prices populated)
        option_pnl_pct = None
        if (ce and fo.option_entry_price not in (None, 0)
                and ce.option_last_price not in (None, 0)):
            option_pnl_pct = (ce.option_last_price - fo.option_entry_price) / fo.option_entry_price

        # Wins
        is_win_underlying = (underlying_pnl_pct > 0) if underlying_pnl_pct is not None else None
        is_win_option = (option_pnl_pct > 0) if option_pnl_pct is not None else None

        row = {
            "session_file": fo.session_file,
            "ticker": fo.ticker,
            "direction": fo.direction,
            "tier": fo.tier,
            "entry_time_iso": fo.entry_time_iso,
            "open_event_ts": fo.open_event_ts,
            "entry_date_et": entry_date.isoformat() if entry_date else None,
            "entry_time_et": entry_dt_et.strftime("%H:%M") if entry_dt_et else None,
            "entry_minute_of_day": (entry_dt_et.hour * 60 + entry_dt_et.minute) if entry_dt_et else None,
            "day_of_week": entry_dt_et.weekday() if entry_dt_et else None,
            "entry_price": fo.entry_price,
            "atr_at_entry": fo.atr_at_entry,
            "atr_pct_of_entry": (fo.atr_at_entry / fo.entry_price) if (fo.atr_at_entry and fo.entry_price) else None,
            "sl_price": fo.sl_price,
            "sl_distance_pct": (abs(fo.sl_price - fo.entry_price) / fo.entry_price) if (fo.sl_price and fo.entry_price) else None,
            "option_symbol": fo.option_symbol,
            "option_root": root,
            "option_expiry": expiry.isoformat() if expiry else None,
            "option_cp": cp,
            "option_strike": strike,
            "option_dte": dte,
            "option_strike_distance_pct": strike_dist_pct,
            "option_entry_price": fo.option_entry_price,
            "qty": fo.qty,
            # signal context
            "signal_ts": fo.signal_ts,
            "p_dir": fo.p_dir,
            "ev_score": fo.ev_score,
            "ref_high": fo.ref_high,
            "ref_low": fo.ref_low,
            "signal_atr": fo.signal_atr,
            "ref_range_pct": ref_range_pct,
            "entry_vs_ref": entry_vs_ref,
            # confirmation
            "confirm_bars_watched": fo.confirm_bars_watched,
            "confirm_close": fo.confirm_close,
            "signal_to_entry_secs": signal_to_entry_secs,
            # close
            "match_type": match_type,
            "close_session_file": ce.session_file if ce else None,
            "close_event_ts": ce.close_event_ts if ce else None,
            "exit_price": ce.exit_price if ce else None,
            "exit_reason": ce.exit_reason if ce else None,
            "bars_held": ce.bars_held if ce else None,
            "option_last_price": ce.option_last_price if ce else None,
            "option_best_price": ce.option_best_price if ce else None,
            "best_price": ce.best_price if ce else None,
            "trail_armed": ce.trail_armed if ce else None,
            "deferred_trail_active": ce.deferred_trail_active if ce else None,
            "deferred_trail_trigger_pnl_pct": ce.deferred_trail_trigger_pnl_pct if ce else None,
            "order_error": ce.order_error if ce else None,
            # PnL
            "underlying_pnl_pct": underlying_pnl_pct,
            "option_pnl_pct": option_pnl_pct,
            "is_win_underlying": is_win_underlying,
            "is_win_option": is_win_option,
        }
        rows.append(row)

    print(f"same-session matches: {same_session_match}")
    print(f"cross-session (option_symbol) matches: {cross_session_match}")
    print(f"unclosed (no close anywhere in window): {no_match}")

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-dir", default="UI/swing_audit")
    p.add_argument("--since", default="20260514", help="YYYYMMDD lower bound (inclusive)")
    p.add_argument("--out", default="local_artifacts/swing_analysis_20260525/trades.parquet")
    args = p.parse_args()

    df = build_dataset(args.audit_dir, args.since)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {len(df)} rows -> {out_path}")
    print(f"      {len(df)} rows -> {csv_path}")
    # Quick coverage
    n_with_signal = df["p_dir"].notna().sum()
    n_with_close = df["exit_price"].notna().sum()
    n_with_underlying_pnl = df["underlying_pnl_pct"].notna().sum()
    n_with_option_pnl = df["option_pnl_pct"].notna().sum()
    print(f"\nCoverage:")
    print(f"  with signal context (p_dir): {n_with_signal}/{len(df)}")
    print(f"  with a matched close: {n_with_close}/{len(df)}")
    print(f"  with underlying PnL: {n_with_underlying_pnl}/{len(df)}")
    print(f"  with option PnL: {n_with_option_pnl}/{len(df)}")


if __name__ == "__main__":
    main()
