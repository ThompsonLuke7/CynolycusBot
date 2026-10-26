"""Continuous ACE entry/management/exit imitation diagnostic, research only.

The observed symbol/session panel is supplied for evaluation. The rule engine
sees completed bars and strictly earlier source plans, never alert/exit times.
Both directions are scanned. Source events are matched only after replay.
Historical test dates are preserved but were previously inspected; results
are exploratory. This script imports no order-submission path.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from core.calendar.us_market_calendar import is_trading_day, next_trading_day, prev_trading_day
from scripts.discord_ledger.build import available_at, digest, read_jsonl, write_jsonl
from scripts.discord_ledger.extract_context import extract_symbols
from scripts.discord_ledger.run_ace_setup_replay import DEFAULT_EVENTS, DEFAULT_LEDGER
from scripts.discord_ledger.run_ace_state_machine_experiment import daily_structure, prepare_daily_bars

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/discord_ledger_2026-09-21"
PRIMARY = BASE / "ace_setup_outcomes_20260923_02/source_alert_outcomes.jsonl"
OLD_RESULTS = BASE / "ace_fixed_state_machine_20261008_01/results.json"
MESSAGES = BASE / "source/messages.jsonl"
EVIDENCE = BASE / "ace_screenshot_evidence_20261009.json"
DAILY = ROOT / "Data/shared/bars/1d"
BAR_MINUTES = 5
TOLERANCES = (1, 2, 3)
SEED = 1009
NULL_DRAWS = 100
BOOTSTRAP_DRAWS = 2000


def utc(value: Any) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError(f"timezone-aware timestamp required: {value}")
    return t.tz_convert("UTC")


def session_bounds(day: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Explicit US equity early closes in this frozen 2025/2026 study."""
    d = date.fromisoformat(day)
    if d.year not in {2025, 2026} or not is_trading_day(d):
        raise ValueError(f"unsupported/non-trading session {day}")
    half_days = {"2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24"}
    local = pd.Timestamp(day, tz="America/New_York")
    return ((local + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC"),
            (local + pd.Timedelta(hours=13 if day in half_days else 16)).tz_convert("UTC"))


def completed_candles(frame: pd.DataFrame, symbol: str, day: str) -> tuple[pd.DataFrame, dict]:
    """Aggregate exactly five distinct minute bars; never fill a missing minute."""
    start, end = session_bounds(day)
    f = frame[frame.symbol.astype(str).str.upper().eq(symbol)].copy()
    f["timestamp"] = pd.to_datetime(f.timestamp, utc=True)
    f = f[(f.timestamp >= start) & (f.timestamp < end)]
    if f.timestamp.duplicated().any():
        raise ValueError(f"duplicate minute observations: {symbol} {day}")
    f = f.sort_values("timestamp", kind="stable")
    for col in ("open", "high", "low", "close", "volume", "vwap"):
        if not np.isfinite(f[col]).all():
            raise ValueError(f"nonfinite minute {col}: {symbol} {day}")
    if (f[["open", "high", "low", "close"]] <= 0).any().any() or (f.volume < 0).any():
        raise ValueError(f"invalid price/volume: {symbol} {day}")
    rows = []
    for stamp, g in f.groupby(f.timestamp.dt.floor("5min"), sort=True):
        expected = pd.date_range(stamp, periods=5, freq="min", tz="UTC")
        if len(g) != 5 or not pd.DatetimeIndex(g.timestamp).equals(expected):
            continue
        vol = float(g.volume.sum())
        rows.append({"timestamp": stamp, "decision_at_utc": stamp + pd.Timedelta(minutes=5),
                     "open": float(g.open.iloc[0]), "high": float(g.high.max()),
                     "low": float(g.low.min()), "close": float(g.close.iloc[-1]),
                     "volume": vol, "vwap": float((g.vwap * g.volume).sum() / vol) if vol > 0 else None})
    out = pd.DataFrame(rows, columns=["timestamp", "decision_at_utc", "open", "high", "low", "close", "volume", "vwap"])
    if not out.empty:
        out["session_vwap"] = (out.vwap * out.volume).cumsum() / out.volume.cumsum()
    return out, {"symbol": symbol, "session_date_et": day, "minute_rows": len(f),
                 "expected_minutes": int((end - start).total_seconds() / 60),
                 "complete_5m_candles": len(out), "expected_5m_candles": int((end - start).total_seconds() / 300),
                 "gaps_5m": int((end - start).total_seconds() / 300) - len(out),
                 "session_close_observed": bool(len(out) and out.decision_at_utc.iloc[-1] == end)}


def source_labels() -> tuple[list[dict], list[dict], dict]:
    """Deduplicate literal entry decisions; retain management link uncertainty."""
    primary = read_jsonl(PRIMARY)
    primary_keys = {(r["symbol"], int(r["direction_sign"]), utc(r["decision_at_utc"])) for r in primary}
    raw = read_jsonl(DEFAULT_LEDGER)
    grouped: dict[tuple, dict] = {}
    entry_trade: dict[str, dict] = {}
    exclusions: Counter = Counter()
    for r in raw:
        snap = r.get("entry_underlying_snapshot") or {}
        sign = r.get("underlying_direction_sign")
        if sign not in {-1, 1} or snap.get("status") != "ok":
            exclusions[f"{snap.get('status', 'missing')}:direction_{sign}"] += 1
            continue
        key = (snap["proxy_symbol"], sign, utc(r["entry_available_at_utc"]))
        if key not in grouped:
            grouped[key] = {"label_id": f"entry:{r['entry_message_id']}:{key[0]}:{sign}", "kind": "entry",
                            "symbol": key[0], "direction_sign": sign, "decision_at_utc": key[2].isoformat(),
                            "session_date_et": key[2].tz_convert("America/New_York").date().isoformat(),
                            "source_message_ids": [], "trade_ids": [], "source_session_file": snap["source_session_file"],
                            "primary_cohort": key in primary_keys, "link_confidence": "entry_source",
                            "source_symbol": r["symbol"], "proxy_mapping": snap.get("proxy_mapping")}
        g = grouped[key]
        # Only ENTRY IDs label an entry. Later management IDs are never inputs.
        g["source_message_ids"] = sorted(set(g["source_message_ids"] + [str(r["entry_message_id"])]))
        g["trade_ids"].append(r["trade_id"])
        entry_trade[r["trade_id"]] = g
    entries = sorted(grouped.values(), key=lambda r: (r["decision_at_utc"], r["symbol"]))
    if sum(r["primary_cohort"] for r in entries) != len(primary):
        raise ValueError("primary label identity mismatch with frozen 120-alert panel")
    management = {}

    def add_management(entry: dict, mid: str, when: str, kind: str, confidence: str, text: str, basis: str) -> None:
        # A sale of one contract / 'mostly out' does not establish a full close.
        if kind == "exit" and re.search(r"\b(?:most|majority|runner|closed\s+1|stc\s+1)\b", text, re.I):
            kind = "trim"
        if kind not in {"trim", "exit"}:
            return
        t = utc(when)
        key = (str(mid), entry["symbol"], entry["direction_sign"], kind)
        existing = management.get(key)
        row = {"label_id": f"{kind}:{mid}:{entry['symbol']}", "kind": kind, "symbol": entry["symbol"],
               "direction_sign": entry["direction_sign"], "decision_at_utc": t.isoformat(),
               "session_date_et": t.tz_convert("America/New_York").date().isoformat(),
               "source_message_ids": [str(mid)], "entry_label_ids": [entry["label_id"]],
               "primary_cohort": entry["primary_cohort"], "link_confidence": confidence,
               "source_quote": text, "link_basis": basis}
        if existing:
            row["entry_label_ids"] = sorted(set(existing["entry_label_ids"] + row["entry_label_ids"]))
            row["primary_cohort"] = bool(existing["primary_cohort"] or row["primary_cohort"])
            # Preserve strongest existing association; append the other evidence.
            rank = {"high": 3, "medium": 2, "low": 1, "very_low": 0}
            if rank.get(existing["link_confidence"], -1) > rank.get(confidence, -1):
                row["link_confidence"] = existing["link_confidence"]
        management[key] = row

    for e in read_jsonl(DEFAULT_EVENTS):
        entry = entry_trade.get(e.get("trade_id"))
        if entry and e.get("event_type") in {"trim", "exit"}:
            add_management(entry, e["message_id"], e["available_at_utc"], e["event_type"],
                           e.get("link_confidence", "unknown"), e.get("source_quote") or "", "curated_lifecycle_link")
    for r in raw:
        a = r.get("management_anchor")
        entry = entry_trade.get(r["trade_id"])
        if not a or not entry or a.get("management_event_type") not in {"trim", "exit"}:
            continue
        add_management(entry, a["management_message_id"], a["management_available_at_utc"],
                       a["management_event_type"], a["link_confidence"], a["management_text"], "reconciled_management_anchor")
    return entries, list(management.values()), {"raw_lifecycles": len(raw), "deduplicated_covered_entry_decisions": len(entries),
           "primary_entry_decisions": len(primary), "exclusions": dict(exclusions),
           "entry_identity_rule": "symbol + direction + source availability; source IDs are original entry IDs only",
           "management_label_counts": dict(Counter(f"{r['kind']}:{r['link_confidence']}" for r in management.values()))}


def published_plans() -> tuple[list[dict], list[dict]]:
    """Only explicit entry prices in unambiguous single-symbol ACE messages."""
    plans, excluded = [], []
    pattern = re.compile(r"\bEntry(?:\s+Zone)?\s*[:=]?\s*(?:at\s*)?\$?(\d+(?:\.\d+)?)(?:\s*[–—-]\s*\$?(\d+(?:\.\d+)?))?", re.I)
    for m in read_jsonl(MESSAGES):
        text = m.get("text") or ""
        match = pattern.search(text)
        if m.get("author_name") != "ACE" or not match:
            continue
        if re.search(r'\b\d+(?:\.\d+)?\s*(?:[cp]|calls?|puts?)\b|\b(?:contract|premium)\b',text,re.I):
            excluded.append({'source_message_ids':[m['message_id']],'status':'option_premium_is_not_underlying_entry_zone','source_quote':text})
            continue
        symbols = {s for s, _raw in extract_symbols(text)}
        when, basis = available_at(m)
        if len(symbols) != 1 or when is None:
            excluded.append({"source_message_ids": [m["message_id"]], "status": "ambiguous_symbol_or_availability", "source_quote": text})
            continue
        zone = sorted([float(match[1]), float(match[2] or match[1])])
        plans.append({"symbol": next(iter(symbols)), "entry_zone": zone, "available_at_utc": when,
                      "availability_basis": basis, "source_message_ids": [m["message_id"]], "source_quote": text})
    return sorted(plans, key=lambda r: r["available_at_utc"]), excluded


@dataclass(frozen=True)
class Policy:
    name: str
    trigger: str
    sessions: int = 1
    level_names: tuple[str,...] = ('sma_5','ema_21','sma_34','sma_55')


POLICIES = (
    Policy("strict_snapshot", "strict"),
    Policy("daily_touch_reaction", "daily"),
    Policy("daily_touch_reaction_swing", "daily", 2),
    Policy("intraday_retest_reaction", "intraday"),
    Policy("published_entry_zone", "published", 2),
    Policy("sma5_touch_reaction_swing", "daily", 2, ('sma_5',)),
    Policy("ema21_touch_reaction_swing", "daily", 2, ('ema_21',)),
)


def replay_symbol(
    symbol: str, sessions: list[tuple[str, pd.DataFrame, dict]], policy: Policy, plans: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Own one position across both directions, with causal arming and management."""
    events: list[dict] = []
    trades: list[dict] = []
    traces: list[dict] = []
    position = None
    armed: dict[int, dict] = {}
    cooldown_until: pd.Timestamp | None = None
    prior_decision = None
    seq = 0
    for session_index, (day, frame, daily) in enumerate(sessions):
        start, end = session_bounds(day)
        previous = None
        prior_bar_rows: list[dict] = []
        for b in frame.to_dict("records"):
            t = b["decision_at_utc"]
            contiguous = prior_decision is not None and t - prior_decision == pd.Timedelta(minutes=5)
            if not contiguous:
                armed.clear()
                prior_bar_rows.clear()
            prior_decision = t
            close = float(b["close"])
            if position is not None:
                p = position
                p["observed_holding_bars"] += 1
                move = p["direction_sign"] * (close - p["entry_price_reference"])
                p["best_close_move"] = max(p["best_close_move"], move)
                p["max_favorable_excursion_pct"] = max(p["max_favorable_excursion_pct"],
                    100 * p["direction_sign"] * ((b["high"] if p["direction_sign"] == 1 else b["low"]) / p["entry_price_reference"] - 1))
                p["max_adverse_excursion_pct"] = min(p["max_adverse_excursion_pct"],
                    100 * p["direction_sign"] * ((b["low"] if p["direction_sign"] == 1 else b["high"]) / p["entry_price_reference"] - 1))
                stop_touched = b["low"] <= p["stop"] if p["direction_sign"] == 1 else b["high"] >= p["stop"]
                tp1 = b["high"] >= p["target_1"] if p["direction_sign"] == 1 else b["low"] <= p["target_1"]
                tp2 = b["high"] >= p["target_2"] if p["direction_sign"] == 1 else b["low"] <= p["target_2"]
                reason = "invalidation_or_trailing_stop" if stop_touched else "target_2" if tp2 else None
                if reason is None and policy.sessions == 1 and p["observed_holding_bars"] >= 6 and p["best_close_move"] < .25 * p["risk"]:
                    reason = "no_progress_30m"
                if reason is None and session_index - p["entry_session_index"] + 1 >= policy.sessions and t == end:
                    reason = "holding_session_cap"
                if not stop_touched and tp1 and not p["trimmed"]:
                    event = {"kind": "trim", "decision_at_utc": t.isoformat(), "symbol": symbol,
                             "direction_sign": p["direction_sign"], "policy": policy.name,
                             "position_id": p["position_id"], "reason": "target_1", "price_reference": close}
                    events.append(event)
                    p["trimmed"] = True
                    p["trim_price_reference"] = close
                    p["stop"] = p["entry_price_reference"]  # next-candle effective
                if reason:
                    event = {"kind": "exit", "decision_at_utc": t.isoformat(), "symbol": symbol,
                             "direction_sign": p["direction_sign"], "policy": policy.name, "position_id": p["position_id"],
                             "reason": reason, "price_reference": close, "both_stop_and_target_touched": bool(stop_touched and tp1)}
                    events.append(event)
                    p.update({"exit_at_utc": t.isoformat(), "exit_price_reference": close, "exit_reason": reason, "censored": False})
                    p["gross_underlying_return_pct"] = 100 * p["direction_sign"] * (close / p["entry_price_reference"] - 1)
                    trades.append(dict(p))
                    position = None
                    cooldown_until = t + pd.Timedelta(minutes=15)
                else:
                    # Daily-ATR trailing proxy, applied only after this candle.
                    if p["best_close_move"] >= p["risk"]:
                        proposal = p["entry_price_reference"] + p["direction_sign"] * .5 * p["best_close_move"]
                        p["stop"] = max(p["stop"], proposal) if p["direction_sign"] == 1 else min(p["stop"], proposal)
            if position is None and (cooldown_until is None or t > cooldown_until):
                candidates = []
                for sign in (1, -1):
                    state, level, plan_ids = "WATCHING", None, []
                    trend = daily.get("daily_feature_status") == "ok" and (
                        daily["ema_21"] >= daily["sma_55"] if sign == 1 else daily["ema_21"] <= daily["sma_55"])
                    if daily.get("daily_feature_status") != "ok":
                        traces.append({"policy": policy.name, "symbol": symbol, "direction_sign": sign,
                                       "decision_at_utc": t.isoformat(), "state": "DATA_UNAVAILABLE", "reason": daily["daily_feature_status"]})
                        continue
                    atr = daily["atr_14"]
                    prev_close = float(previous["close"]) if previous is not None and contiguous else float(daily["prior_close"])
                    reaction = sign * (close - b["open"]) > 0 and sign * (close - prev_close) > 0
                    if policy.trigger == "strict":
                        level = min((daily[k] for k in ("sma_5", "ema_21", "sma_34", "sma_55")), key=lambda x: abs(close-x))
                        pullback = ((daily["prior_high_20"] - close) if sign == 1 else (close - daily["prior_low_20"])) / atr
                        strict_order = trend and (daily["sma_5"] >= daily["ema_21"] if sign == 1 else daily["sma_5"] <= daily["ema_21"])
                        volume_ratio = b["volume"] / np.mean([v["volume"] for v in prior_bar_rows[-4:]]) if len(prior_bar_rows) >= 4 and np.mean([v["volume"] for v in prior_bar_rows[-4:]]) > 0 else None
                        confirmed = bool(strict_order and .25 <= pullback <= 5 and abs(close-level) <= .35*atr
                                         and len(prior_bar_rows) >= 4 and sign*(close-prev_close)>0
                                         and sign*(close-b["session_vwap"])>0 and volume_ratio is not None and volume_ratio>=1)
                        state = "CONFIRMED" if confirmed else "SIMULTANEOUS_GATE_MISS"
                    else:
                        levels = []
                        if policy.trigger == "daily" and trend:
                            levels = [(daily[k], .10*atr, [], "daily_"+k) for k in policy.level_names]
                        elif policy.trigger == "intraday" and previous is not None and contiguous:
                            levels = [(previous["session_vwap"], .05*atr, [], "vwap_retest")]
                            opening = [v for v in prior_bar_rows if v["timestamp"] < start + pd.Timedelta(minutes=30)]
                            if len(opening) == 6 and t >= start + pd.Timedelta(minutes=35):
                                levels += [(max(v["high"] for v in opening), .05*atr, [], "opening_range_high"),
                                           (min(v["low"] for v in opening), .05*atr, [], "opening_range_low")]
                            levels = [v for v in levels if sign*(prev_close-v[0]) >= 0]
                        elif policy.trigger == "published":
                            eligible = [p for p in plans if p["symbol"] == symbol and utc(p["available_at_utc"]) < t
                                        and t-utc(p["available_at_utc"]) <= pd.Timedelta(days=7)]
                            # No inferred short plans from an unsigned entry-zone note.
                            if eligible and sign == 1:
                                plan = eligible[-1]
                                lo, hi = plan["entry_zone"]
                                levels = [((lo+hi)/2, (hi-lo)/2 + .02*atr, plan["source_message_ids"], "published_entry_zone")]
                        hits = [v for v in levels if b["low"] <= v[0]+v[1] and b["high"] >= v[0]-v[1]
                                and abs(close-v[0]) <= .75*atr]
                        if hits:
                            chosen = min(hits, key=lambda x: abs(close-x[0]))
                            armed[sign] = {"level": chosen[0], "band": chosen[1], "plan_ids": chosen[2],
                                           "level_kind": chosen[3], "expires": t+pd.Timedelta(minutes=15),
                                           "touch_at_utc": t.isoformat(), "touch_low": b["low"], "touch_high": b["high"]}
                        arm = armed.get(sign)
                        if arm and t > arm["expires"]:
                            armed.pop(sign)
                            arm = None
                        confirmed = bool(arm and reaction and sign*(close-arm["level"]) >= -arm["band"]
                                         and abs(close-arm["level"]) <= .75*atr)
                        if arm:
                            level, plan_ids = arm["level"], arm["plan_ids"]
                        state = "CONFIRMED" if confirmed else "ARMED_AT_LEVEL" if arm else "NO_LEVEL_TOUCH" if levels else "NO_SUPPORTING_TREND_OR_PLAN"
                    traces.append({"policy": policy.name, "symbol": symbol, "direction_sign": sign,
                                   "decision_at_utc": t.isoformat(), "state": state, "level": level,
                                   "trend_supportive": bool(trend), "reaction": bool(reaction),
                                   "source_message_ids": plan_ids})
                    if confirmed:
                        candidates.append((sign, level, plan_ids))
                if len(candidates) == 1:
                    sign, level, plan_ids = candidates[0]
                    arm = armed.get(sign) or {"touch_low": b["low"], "touch_high": b["high"], "touch_at_utc": t.isoformat(), "level_kind": "strict_nearest_ma"}
                    risk = max(.20*daily["atr_14"], sign*(close - (arm["touch_low"] if sign == 1 else arm["touch_high"])) + .05*daily["atr_14"])
                    target_distance = max(risk, sign*((daily["prior_high_20"] if sign == 1 else daily["prior_low_20"]) - close)) if policy.trigger in {"daily", "published"} else risk
                    seq += 1
                    pid = f"{policy.name}:{symbol}:{seq}"
                    position = {"position_id": pid, "policy": policy.name, "symbol": symbol, "direction_sign": sign,
                                "entry_at_utc": t.isoformat(), "entry_price_reference": close, "entry_session_index": session_index,
                                "stop": close-sign*risk, "initial_invalidation": close-sign*risk, "risk": risk,
                                "target_1": close+sign*target_distance, "target_2": close+sign*max(2*risk,target_distance+risk),
                                "trimmed": False, "observed_holding_bars": 0, "best_close_move": 0.0,
                                "max_favorable_excursion_pct": 0.0, "max_adverse_excursion_pct": 0.0,
                                "touch_at_utc": arm["touch_at_utc"], "level_kind": arm["level_kind"],
                                "source_message_ids": plan_ids, "source_session_file": daily["source_session_file"],
                                "price_basis": "completed_5m_close_reference_not_verified_fill", "censored": True}
                    events.append({"kind": "entry", "decision_at_utc": t.isoformat(), "symbol": symbol,
                                   "direction_sign": sign, "policy": policy.name, "position_id": pid,
                                   "price_reference": close, "reason": arm["level_kind"], "source_message_ids": plan_ids})
                    armed.clear()
                elif len(candidates) > 1:
                    traces.append({"policy": policy.name, "symbol": symbol, "decision_at_utc": t.isoformat(), "state": "AMBIGUOUS_BOTH_DIRECTIONS_ABSTAIN"})
            else:
                for sign in (1,-1):
                    traces.append({"policy":policy.name,"symbol":symbol,"direction_sign":sign,
                                   "decision_at_utc":t.isoformat(),
                                   "state":"POSITION_ALREADY_OPEN" if position else "POST_EXIT_COOLDOWN",
                                   "position_id":position['position_id'] if position else None})
            previous = b
            prior_bar_rows.append(b)
        if position is not None:
            next_day = sessions[session_index+1][0] if session_index+1 < len(sessions) else None
            complete = bool(len(frame) and frame.decision_at_utc.iloc[-1] == end)
            adjacent = next_day == next_trading_day(date.fromisoformat(day)).isoformat()
            if policy.sessions == 1 or not complete or not adjacent:
                # Missing continuation is not a synthetic sale. Preserve the censored trade.
                position.update({"censor_at_utc": None if frame.empty else frame.decision_at_utc.iloc[-1].isoformat(),
                                 "censor_reason": "missing_complete_adjacent_session", "gross_underlying_return_pct": None})
                trades.append(dict(position))
                position = None
    return events, trades, traces


def one_to_one_matches(labels: list[dict], signals: list[dict], tolerance_minutes: int) -> list[dict]:
    """Maximum cardinality, then minimum absolute timing error; no signal reuse."""
    if not labels or not signals:
        return []
    cost = np.full((len(labels), len(signals)+len(labels)), 100_000.0)
    cost[:,len(signals):] = 10_000.0
    for i, a in enumerate(labels):
        for j, b in enumerate(signals):
            if a["symbol"] != b["symbol"] or a["direction_sign"] != b["direction_sign"] or a["kind"] != b["kind"]:
                continue
            # A near-midnight/overnight signal cannot match a different session.
            ta, tb = utc(a["decision_at_utc"]), utc(b["decision_at_utc"])
            if ta.tz_convert("America/New_York").date() != tb.tz_convert("America/New_York").date():
                continue
            delta = (tb-ta).total_seconds()/60
            if abs(delta) <= tolerance_minutes:
                cost[i,j] = abs(delta)
    rows, cols = linear_sum_assignment(cost)
    return [{"label_id": labels[i]["label_id"], "position_id": signals[j]["position_id"],
             "signal_index": int(j), "label_index": int(i), "kind": labels[i]["kind"],
             "signed_offset_minutes": (utc(signals[j]["decision_at_utc"])-utc(labels[i]["decision_at_utc"])).total_seconds()/60,
             "source_message_ids": labels[i]["source_message_ids"], "symbol": labels[i]["symbol"],
             "direction_sign": labels[i]["direction_sign"], "session_date_et": labels[i]["session_date_et"]}
            for i,j in zip(rows, cols) if j < len(signals) and cost[i,j] < 10_000]


def entry_metrics(labels: list[dict], signals: list[dict], tolerance_candles: int) -> dict:
    matches = one_to_one_matches(labels, signals, tolerance_candles*5)
    return {"source_entries": len(labels), "algorithm_entries": len(signals), "matched_entries": len(matches),
            "entry_recall": len(matches)/len(labels) if labels else None,
            "export_associated_signal_fraction": len(matches)/len(signals) if signals else None,
            "algorithm_entries_unmatched_in_export": len(signals)-len(matches),
            "median_absolute_offset_minutes": float(np.median([abs(m["signed_offset_minutes"]) for m in matches])) if matches else None}


def shifted_null(labels: list[dict], signals: list[dict], grids: dict[tuple, list[str]] | None = None) -> dict:
    """Session-preserving circular shifts; preserve frequency/direction and paths."""
    observed = one_to_one_matches(labels, signals, 10)
    days = sorted({r["session_date_et"] for r in labels})
    if not labels:
        return {"status": "no_source_labels"}
    if not signals:
        return {"status":"no_algorithm_entries_to_test","draws":0,"observed_entry_recall":0.0,
                "mean_shifted_entry_recall":0.0,"session_equal_weight_lift_pp":None,
                "session_bootstrap_95pct_ci_pp":None,"session_mde_pp":None,"session_count":len(days)}
    obs = Counter(r["session_date_et"] for r in observed)
    counts = Counter(r["session_date_et"] for r in labels)
    null_day = defaultdict(list)
    rates = []
    rng = np.random.default_rng(SEED)
    for _ in range(NULL_DRAWS):
        keys=sorted({(s['symbol'],utc(s['decision_at_utc']).tz_convert('America/New_York').date().isoformat()) for s in signals})
        offsets={key:int(rng.integers(1,max(2,len(grids[key])))) if grids and grids.get(key)
                 else int(rng.integers(4,max(5,int((session_bounds(key[1])[1]-session_bounds(key[1])[0]).total_seconds()/300)-3)))
                 for key in keys}
        shifted = []
        for signal in signals:
            d = utc(signal["decision_at_utc"]).tz_convert("America/New_York").date().isoformat()
            key=(signal['symbol'],d)
            start,end = session_bounds(d)
            n = int((end-start).total_seconds()/300)
            old = int((utc(signal["decision_at_utc"])-start).total_seconds()/300)-1
            b = dict(signal)
            if grids and grids.get(key):
                times=grids[key]
                ix=times.index(utc(signal['decision_at_utc']).isoformat())
                b['decision_at_utc']=times[(ix+offsets[key])%len(times)]
            else:
                b["decision_at_utc"] = (start+pd.Timedelta(minutes=5*((old+offsets[key])%n+1))).isoformat()
            shifted.append(b)
        match = one_to_one_matches(labels, shifted, 10)
        counter = Counter(m["session_date_et"] for m in match)
        rates.append(len(match)/len(labels))
        for d in days:
            null_day[d].append(counter[d]/counts[d])
    delta = np.array([obs[d]/counts[d]-np.mean(null_day[d]) for d in days])
    if len(delta) >= 2:
        mde = float((NormalDist().inv_cdf(.975)+NormalDist().inv_cdf(.80))*delta.std(ddof=1)/np.sqrt(len(delta))*100)
        boot = rng.choice(delta,size=(BOOTSTRAP_DRAWS,len(delta)),replace=True).mean(axis=1)*100
        ci = [float(v) for v in np.quantile(boot,[.025,.975])]
    else:
        mde,ci = None,None
    return {"draws": NULL_DRAWS, "seed": SEED, "mean_shifted_entry_recall": float(np.mean(rates)),
            "observed_entry_recall": len(observed)/len(labels),
            "session_equal_weight_lift_pp": float(delta.mean()*100),
            "session_bootstrap_95pct_ci_pp": ci, "session_mde_pp": mde,
            "session_count": len(days), "null_scope": "incidental timing overlap at same symbol/session signal frequency, shifted only onto observed complete candles; no return/profitability inference"}


def lifecycle_metrics(entries: list[dict], management: list[dict], signals: list[dict], tolerance: int) -> dict:
    entry_signals = [s for s in signals if s["kind"] == "entry"]
    entry_matches = one_to_one_matches(entries, entry_signals, tolerance*5)
    by_entry = {m["label_id"]:m["position_id"] for m in entry_matches}
    result = {}
    for confidence_group, allowed in (("high_medium", {"high","medium"}), ("including_low_estimates", {"high","medium","low","very_low"})):
        for kind in ("trim","exit"):
            refs = [m for m in management if m["kind"] == kind and m["link_confidence"] in allowed]
            paired = 0
            offsets = []
            eligible_after_entry = 0
            consumed: set[int] = set()
            for r in sorted(refs,key=lambda x:x["decision_at_utc"]):
                pids = {by_entry[k] for k in r["entry_label_ids"] if k in by_entry}
                if not pids:
                    continue
                eligible_after_entry += 1
                choices = [(abs((utc(s["decision_at_utc"])-utc(r["decision_at_utc"])).total_seconds()/60),i,s)
                           for i,s in enumerate(signals) if i not in consumed and s["kind"]==kind and s["position_id"] in pids]
                if choices:
                    distance,i,s = min(choices,key=lambda x:x[0])
                    if distance <= tolerance*5:
                        paired += 1
                        consumed.add(i)
                        offsets.append((utc(s["decision_at_utc"])-utc(r["decision_at_utc"])).total_seconds()/60)
            result[f"{kind}:{confidence_group}"] = {"source_management_labels":len(refs),
                "source_labels_with_matched_algorithm_entry":eligible_after_entry,
                "entry_and_management_matched":paired,
                "full_sequence_recall":paired/len(refs) if refs else None,
                "conditional_management_recall":paired/eligible_after_entry if eligible_after_entry else None,
                "median_absolute_offset_minutes":float(np.median(np.abs(offsets))) if offsets else None,
                "status":"measured" if refs else "no_exported_labels_in_partition"}
    return result


def trade_summary(trades: list[dict]) -> dict:
    closed = [t for t in trades if not t["censored"]]
    x = np.array([t["gross_underlying_return_pct"] for t in closed])
    return {"generated_positions":len(trades),"closed_reference_paths":len(closed),"censored_paths":len(trades)-len(closed),
            "mean_gross_directional_return_pct":float(x.mean()) if len(x) else None,
            "positive_gross_return_fraction":float((x>0).mean()) if len(x) else None,
            "mean_return_less_10bps_pct":float(x.mean()-.10) if len(x) else None,
            "mean_return_less_25bps_pct":float(x.mean()-.25) if len(x) else None,
            "median_observed_holding_minutes":float(np.median([t['observed_holding_bars']*5 for t in closed])) if closed else None,
            "exit_reasons":dict(Counter(t["exit_reason"] for t in closed)),
            "price_warning":"close-to-close single-unit underlying path diagnostic; trims log timing only; no option P&L, fill, quantity, portfolio return or calibrated execution cost"}


def screenshot_parity() -> dict:
    evidence = json.loads(EVIDENCE.read_text())
    comparisons = []
    for chart in evidence["charts"]:
        d = prepare_daily_bars(pd.read_parquet(DAILY/f"{chart['symbol']}.parquet"))
        candidates = d[np.abs(d.close-chart["close"])<.005]
        checks = []
        for row in candidates.itertuples():
            # End-of-chart daily values are used only to identify that chart's
            # observation session, never fed to an earlier decision.
            hist = d[d.timestamp<=row.timestamp]
            values = {"sma_5":hist.close.tail(5).mean(),"ema_21":hist.close.ewm(span=21,adjust=False).mean().iloc[-1],
                      "sma_34":hist.close.tail(34).mean(),"sma_55":hist.close.tail(55).mean()}
            error = max(abs(float(v)-chart[k]) for k,v in values.items())
            checks.append({"chart_observation_session_et":row.session_date_et.isoformat(),"max_absolute_ma_error":error,
                           "local_values":{k:float(v) for k,v in values.items()},"all_four_match_within_0_01":error<=.01})
        comparisons.append({"symbol":chart["symbol"],"source_image_ordinals":chart["source_image_ordinals"],
                            "checks":checks,"message_available_at_utc":None})
    return {"chart_checks":comparisons,"absolute_message_times_missing":True,
            "note":"The four daily averages validate local price scale/calculation. A matched chart date does not supply a message timestamp or execution date."}


def session_daily_features(daily: pd.DataFrame | None, day: str, frame: pd.DataFrame, path: str) -> dict:
    start,_end=session_bounds(day)
    if daily is None or frame.empty:
        return {'daily_feature_status':'missing_daily_bars' if daily is None else 'no_complete_5m_candle','source_session_file':path}
    features=daily_structure(daily,decision_at_utc=start,entry_price=float(frame.open.iloc[0]),direction_sign=1)
    if features['daily_feature_status']=='ok':
        h=daily[daily.session_date_et<date.fromisoformat(day)]
        if features['latest_completed_daily_session_et'] != prev_trading_day(date.fromisoformat(day)).isoformat():
            features['daily_feature_status']='stale_prior_daily_session'
        features.update({'prior_close':float(h.close.iloc[-1]),'prior_high_20':float(h.high.tail(20).max()),'prior_low_20':float(h.low.tail(20).min())})
    features['source_session_file']=path
    return features


def screenshot_case_diagnostics(bars_dir: Path) -> tuple[list[dict],list[dict]]:
    """Real October paths and model signals; no fabricated screenshot event labels."""
    zones={'INTC':(111.37,113.0),'ARM':(298.0,298.0)}
    cases,events=[],[]
    for symbol in ('ARM','INTC','NBIS','ALAB','COHR','MU'):
        daily=prepare_daily_bars(pd.read_parquet(DAILY/f'{symbol}.parquet'))
        sessions=[]
        for day in ('2026-10-06','2026-10-07','2026-10-08'):
            p=bars_dir/f'{day}.parquet'
            if not p.exists():continue
            raw=pd.read_parquet(p)
            raw['timestamp']=pd.to_datetime(raw.timestamp,utc=True)
            start,end=session_bounds(day)
            raw=raw[(raw.symbol==symbol)&(raw.timestamp>=start)&(raw.timestamp<end)]
            candles,cov=completed_candles(raw,symbol,day)
            features=session_daily_features(daily,day,candles,str(p))
            sessions.append((day,candles,features))
            exact_hits=pd.DataFrame()
            if symbol in zones:
                lo,hi=zones[symbol];exact_hits=raw[(raw.low<=hi)&(raw.high>=lo)]
            cases.append({'symbol':symbol,'session_date_et':day,'coverage':cov,
                          'daily_feature_status':features['daily_feature_status'],
                          'image_reported_zone':list(zones[symbol]) if symbol in zones else None,
                          'exact_zone_first_observed_bar_start_utc':None if exact_hits.empty else utc(exact_hits.timestamp.iloc[0]).isoformat(),
                          'image_message_available_at_utc':None,'entry_timing_recall':None,
                          'source_session_file':str(p),'status':'price_path_diagnostic_without_absolute_screenshot_event_timestamp'})
        for policy in (Policy('daily_touch_reaction','daily'),Policy('daily_touch_reaction_swing','daily',2),Policy('intraday_retest_reaction','intraday')):
            es,_trades,_traces=replay_symbol(symbol,sessions,policy,[]);events+=es
    return cases,events


def render_report(out: Path, results: dict) -> None:
    lines = ["# ACE continuous sequence reproduction — 2026-10-09", "",
        "This study evaluates continuously generated entries, trims and exits within ±1/2/3 five-minute candles of exported ACE events. All variants run on the same observed symbol/session panel in both directions. The historical comparison dates have been inspected before and these new hypotheses remain exploratory.", "",
        "## Audit of the previous conclusion", "",
        "The old program's state_machine function is a stateless simultaneous gate evaluated at sampled alert/control instants. It does not retain a touched level, scan intervening candles, generate a position, or test exits. Its empty outcome cells require both sides of a group to pass and do not mean the original bars are missing. It rejects 52% of test directions at the daily ordering gate and its later gates select zero confirmed source alerts. That is a result for one imposed proxy, not a test of all screenshot hypotheses.", "",
        "Its one-observation bootstrap interval is degenerate and should not support an inference. Its constant daily-state zero MDE is also a design artifact: same-symbol/session controls share daily values. This rerun uses session-level uncertainty only where at least two independent sessions exist.", "",
        "## Screenshot findings", "",
        "All 16 displayed SMA5/EMA21/SMA34/SMA55 values match local October 5 daily bars within 0.01. INTC's entry zone contains EMA21 and ARM's 298 zone is close to SMA5. NBIS/ALAB have BEAR cycle labels alongside bullish planning. The screenshots' literal Due-low values conflict with a simple numerical countdown interpretation. Their message IDs, absolute publication dates and display timezone are missing, so they remain image-cited evidence rather than manufactured UTC labels.", "",
        "ARM, ALAB and COHR have no entry labels in the older scored export; NBIS has only one. The old cohort cannot establish whether this particular October AI/momentum basket has been reproduced. `screenshot_case_price_paths.jsonl` and `screenshot_case_algorithm_events.jsonl` test the real October 6–8 price paths independently, while leaving screenshot event-time recall unscored.", "",
        "## Data and protocol", "",
        f"- Original lifecycles: {results['source_audit']['raw_lifecycles']}; expanded covered/deduplicated entry decisions: {results['source_audit']['deduplicated_covered_entry_decisions']}; primary frozen entries: {results['source_audit']['primary_entry_decisions']}.",
        f"- Symbol/session panels: {len(results['coverage'])}; complete candles: {sum(v['complete_5m_candles'] for v in results['coverage'])}; omitted incomplete candles: {sum(v['gaps_5m'] for v in results['coverage'])}. No missing minutes are filled.",
        "- Prior-session daily history is required and stale histories fail closed. Source-plan availability respects edits. Every modeled entry owns its invalidation, targets and management state; management cannot occur before its entry.",
        "- Source entry times, symbols/directions and management labels are used only in scoring. The observed panel itself supplies candidate symbols/dates; full-universe/theme discovery is not assessed.",
        "- One-to-one matching consumes each generated event at most once. Management matches additionally require a matching entry in the same modeled position. Unmatched generated signals mean no corresponding exported alert; the export is incomplete.", "",
        f"- Explicit underlying entry-zone messages extracted: {results['published_plan_count']}; ambiguous or derivative-price entry messages excluded: {results['excluded_ambiguous_plan_count']}. A zero-signal published-zone arm is untested where no compatible source plan is available; it is not evidence against precomputed zones.", "",
        "## Primary fixed comparison partition: entry timing", "",
        "| Policy | Source entries | Algorithm entries | ±1 candle matches | ±2 candle matches | ±3 candle matches | ±2 recall | ±2 associated signal fraction |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for policy,part in results["policies"].items():
        t=part["primary"]["test"];m=t["entries"]
        fmt=lambda v: "not estimable" if v is None else f"{100*v:.1f}%"
        lines.append(f"| {policy} | {m['2']['source_entries']} | {m['2']['algorithm_entries']} | {m['1']['matched_entries']} | {m['2']['matched_entries']} | {m['3']['matched_entries']} | {fmt(m['2']['entry_recall'])} | {fmt(m['2']['export_associated_signal_fraction'])} |")
    lines += ["", "## ETF versus single-stock entry reproduction (primary test, ±2 candles)", "",
        "| Policy | Cohort | Source entries | Matches | Algorithm entries |", "| --- | --- | ---: | ---: | ---: |"]
    for policy,part in results['policies'].items():
        for stratum,m in part['primary_test_strata'].items():
            lines.append(f"| {policy} | {stratum} | {m['source_entries']} | {m['matched_entries']} | {m['algorithm_entries']} |")
    lines += ["", "## Same-frequency shifted-time baseline (±2 candles)", "",
        "| Policy | Recall | Shifted recall | Session-weighted lift (pp) | 95% session CI (pp) | MDE (pp) |", "| --- | ---: | ---: | ---: | --- | ---: |"]
    for policy,part in results["policies"].items():
        n=part["primary"]["test"]["shifted_null"]
        f=lambda v:"not estimable" if v is None else f"{v:.2f}"
        lines.append(f"| {policy} | {f(100*n['observed_entry_recall'])} | {f(100*n['mean_shifted_entry_recall'])} | {f(n['session_equal_weight_lift_pp'])} | {n['session_bootstrap_95pct_ci_pp']} | {f(n['session_mde_pp'])} |")
    lines += ["", "## Entry and management confluence (primary test, ±2 candles)", "",
        "| Policy | Management kind | High/medium source labels | Labels after matched entry | Matching full sequences | Status |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for policy,part in results["policies"].items():
        for kind in ("trim","exit"):
            m=part['primary']['test']['management'][f'{kind}:high_medium']
            lines.append(f"| {policy} | {kind} | {m['source_management_labels']} | {m['source_labels_with_matched_algorithm_entry']} | {m['entry_and_management_matched']} | {m['status']} |")
    lines += ["", "## Expanded exported cohort, all dates (±2 candles)", "",
              "| Policy | Source entries | Algorithm entries | Entry matches | Entry recall | High/medium trims: full matches/labels | High/medium exits: full matches/labels |", "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for policy,part in results["policies"].items():
        m=part['expanded_all']['entries']['2'];lm=part['expanded_all']['management']
        lines.append(f"| {policy} | {m['source_entries']} | {m['algorithm_entries']} | {m['matched_entries']} | {100*m['entry_recall']:.1f}% | {lm['trim:high_medium']['entry_and_management_matched']}/{lm['trim:high_medium']['source_management_labels']} | {lm['exit:high_medium']['entry_and_management_matched']}/{lm['exit:high_medium']['source_management_labels']} |")
    lines += ["", "## Modeled underlying path diagnostics on primary test sessions", "",
              "These are completed-candle price references. No ACE option return, quantity or broker execution is inferred; trims are timing events and do not change the single-unit return diagnostic.", "",
              "| Policy | Closed/censored | Mean gross (%) | Gross-positive paths | Less 10bps (%) | Less 25bps (%) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for policy,part in results["policies"].items():
        m=part['primary']['test']['path_diagnostics'];f=lambda v:"not estimable: no closed paths" if v is None else f"{v:.3f}"
        lines.append(f"| {policy} | {m['closed_reference_paths']}/{m['censored_paths']} | {f(m['mean_gross_directional_return_pct'])} | {f(m['positive_gross_return_fraction'])} | {f(m['mean_return_less_10bps_pct'])} | {f(m['mean_return_less_25bps_pct'])} |")
    lines += ["", "## Why an alert was not reproduced", "",
        "`entry_match_audit.jsonl` lists every primary alert for every policy, its nearest same-direction entry, ±2-candle status, data coverage and nearby gate states. `state_trace.jsonl` records arming/confirmation and missing/stale data. `management_labels.jsonl` preserves source IDs and confidence; missing exit evidence stays unlabelled. `results.json` includes unchanged train/validation/test dates, all tolerances and the low-confidence sensitivity.", "",
        "## Next testable variant", "",
        "Use the exact prepublished support/target zones and timestamped October entry-hit/trim/exit export with the existing Intraday Structure detector and exit policy. Those fields would distinguish cycle timing, level selection and discretionary confirmation. Keep candidate selection/theme tests separate until their historical observation coverage is verified. This study does not exhaust the screenshot-derived strategy family."]
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def plot_representative_sessions(out: Path, session_data: dict, entries: list[dict], events: list[dict], test_dates: list[str]) -> None:
    """Choose by alert count/date, never by return, and show full observed sessions."""
    os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib_ace')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from core.shared_plotting import extract_ohlc, plot_candles
    primary=[r for r in entries if r['primary_cohort'] and r['session_date_et'] in test_dates]
    counts=Counter((r['symbol'],r['session_date_et']) for r in primary)
    ordered=sorted(counts,key=lambda k:(-counts[k],k[1],k[0]))
    chosen=ordered[:1]
    second=next((key for key in ordered if key[0] not in {'SPY','QQQ','IWM','GLD','SLV','XLE','XLF'}),None)
    if second and second not in chosen:chosen.append(second)
    elif len(ordered)>1:chosen.append(ordered[1])
    policies=['strict_snapshot','daily_touch_reaction','intraday_retest_reaction']
    fig,axes=plt.subplots(len(chosen)*len(policies),1,figsize=(17,4*len(chosen)*len(policies)),squeeze=False,constrained_layout=True)
    for row,(symbol,day,policy) in enumerate((s,d,p) for s,d in chosen for p in policies):
        ax=axes[row,0]
        f=next(f for d,f,_ in session_data[symbol] if d==day)
        if f.empty:continue
        plot_candles(ax,*extract_ohlc(f),width=.65)
        times=pd.DatetimeIndex(f.decision_at_utc)
        for ref in primary:
            if ref['symbol']==symbol and ref['session_date_et']==day:
                x=int(np.argmin(np.abs(times.asi8-utc(ref['decision_at_utc']).value)))
                side='long' if ref['direction_sign']==1 else 'short'
                ax.axvline(x,color='#6b21a8',alpha=.65,linestyle='--',label=f'ACE source {side} entry (nearest candle for drawing)')
        for e in events:
            if e['policy']!=policy or e['symbol']!=symbol or utc(e['decision_at_utc']).tz_convert('America/New_York').date().isoformat()!=day:
                continue
            x=int(np.argmin(np.abs(times.asi8-utc(e['decision_at_utc']).value)))
            marker,color={'entry':('^','#0891b2'),'trim':('s','#d97706'),'exit':('x','#dc2626')}[e['kind']]
            if e['kind']=='entry' and e['direction_sign']==-1:marker='v'
            side='long' if e['direction_sign']==1 else 'short'
            ax.scatter([x],[e['price_reference']],marker=marker,c=color,s=75,zorder=4,label=f"Algorithm {side} {e['kind']}")
        tick=np.arange(0,len(f),max(1,len(f)//10))
        ax.set_xticks(tick,[times[i].tz_convert('America/New_York').strftime('%H:%M') for i in tick])
        ax.set_ylabel(f'{symbol} underlying price')
        ax.set_xlabel('Completed five-minute candles, ET; missing candles omitted without fill')
        ax.set_title(f'{day} | {symbol} | {policy} | fixed historical comparison partition | full observed session')
        handles,labels=ax.get_legend_handles_labels();unique=dict(zip(labels,handles));ax.legend(unique.values(),unique.keys(),fontsize=8,loc='best')
        ax.grid(alpha=.15)
    fig.savefig(out/'representative_entry_exit_overlays.png',dpi=135)
    plt.close(fig)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bars-dir',type=Path,help='Immutable alternative raw-session directory; preserves all original label timestamps.')
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    entries,management,audit=source_labels()
    plans,excluded_plans=published_plans()
    old=json.loads(OLD_RESULTS.read_text())
    date_splits=old['chronological_split_dates']
    files={str(p):digest(p) for p in (PRIMARY,DEFAULT_LEDGER,DEFAULT_EVENTS,MESSAGES,OLD_RESULTS,EVIDENCE,Path(__file__))}
    panel=defaultdict(dict)
    for r in entries:
        panel[r['symbol']][r['session_date_et']]=str(args.bars_dir/f"{r['session_date_et']}.parquet") if args.bars_dir else r['source_session_file']
        if args.bars_dir:
            following=next_trading_day(date.fromisoformat(r['session_date_et'])).isoformat()
            p=args.bars_dir/f'{following}.parquet'
            if p.exists():panel[r['symbol']][following]=str(p)
    # Management days with cached observed bars are added for path continuation.
    raw_dir=args.bars_dir or BASE/'underlying_proxy_prices_20260922_06/raw_session_bars'
    for r in management:
        p=raw_dir/f"{r['session_date_et']}.parquet"
        if p.exists():
            panel[r['symbol']][r['session_date_et']]=str(p)
    file_cache={}
    daily_input_frames={}
    session_data={}
    grids={}
    coverage=[]
    for symbol,days in sorted(panel.items()):
        p=DAILY/f'{symbol}.parquet'
        d=prepare_daily_bars(pd.read_parquet(p)) if p.exists() else None
        if d is not None:daily_input_frames[symbol]=d
        if p.exists():files[str(p)]=digest(p)
        ss=[]
        for day,path in sorted(days.items()):
            if path not in file_cache:
                file_cache[path]=pd.read_parquet(path)
                files[path]=digest(Path(path))
            frame,cov=completed_candles(file_cache[path],symbol,day)
            grids[(symbol,day)]=[utc(t).isoformat() for t in frame.decision_at_utc]
            features=session_daily_features(d,day,frame,path)
            cov['daily_feature_status']=features['daily_feature_status'];coverage.append(cov)
            ss.append((day,frame,features))
        session_data[symbol]=ss
    results={'schema':'ace_sequence_replay_v1','research_only':True,'source_audit':audit,'coverage':coverage,
             'chronological_split_dates':date_splits,'historical_test_previously_inspected':True,
             'screenshot_parity':screenshot_parity(),'published_plan_count':len(plans),'excluded_ambiguous_plan_count':len(excluded_plans),
             'normalization':'stable timestamp sort; no duplicate deletion, interpolation or missing-minute fill',
             'price_data_provenance':'retrospective IEX raw minute caches; daily cache adjustment/feed publication metadata not certified; screenshot parity verified',
             'policies':{},'policy_specs':[asdict(p) for p in POLICIES],
             'management_parameters':{'touch_memory_candles':3,'daily_touch_band_atr':.10,
                 'intraday_touch_band_atr':.05,'minimum_initial_risk_daily_atr':.20,
                 'trailing_giveback_fraction':.50,'post_exit_cooldown_minutes':15,
                 'no_progress_minutes':30,'no_progress_required_r':.25},'inputs_sha256':files}
    if args.bars_dir and (args.bars_dir.parent/'manifest.json').exists():
        manifest=args.bars_dir.parent/'manifest.json'
        results['price_data_capture_manifest']=str(manifest)
        results['price_data_provenance']=json.loads(manifest.read_text())['feed']+' explicitly requested, raw, retrospective; no historical publication receipt'
        files[str(manifest)]=digest(manifest)
    all_events,all_trades,all_traces,match_audit=[],[],[],[]
    for policy in POLICIES:
        events,trades,traces=[],[],[]
        for symbol,ss in session_data.items():
            es,ts,qs=replay_symbol(symbol,ss,policy,plans);events+=es;trades+=ts;traces+=qs
        metrics={'primary':{},'expanded_all':{}}
        for part,days in date_splits.items():
            refs=[r for r in entries if r['primary_cohort'] and r['session_date_et'] in days]
            ms=[r for r in management if r['primary_cohort'] and r['session_date_et'] in days]
            ev=[s for s in events if utc(s['decision_at_utc']).tz_convert('America/New_York').date().isoformat() in days]
            # Signal denominator uses the same symbol/session panel as references.
            panels={(r['symbol'],r['session_date_et']) for r in refs}
            ev=[s for s in ev if (s['symbol'],utc(s['decision_at_utc']).tz_convert('America/New_York').date().isoformat()) in panels or s['kind']!='entry']
            e=[s for s in ev if s['kind']=='entry']
            pt=[r for r in trades if (r['symbol'],utc(r['entry_at_utc']).tz_convert('America/New_York').date().isoformat()) in panels]
            metrics['primary'][part]={'entries':{str(k):entry_metrics(refs,e,k) for k in TOLERANCES},
                                     'management':lifecycle_metrics(refs,ms,ev,2),'shifted_null':shifted_null(refs,e,grids),
                                     'path_diagnostics':trade_summary(pt)}
            scored_matches={m['label_id']:m for m in one_to_one_matches(refs,e,10)}
            for r in refs:
                closest=[s for s in e if s['symbol']==r['symbol'] and s['direction_sign']==r['direction_sign']
                         and utc(s['decision_at_utc']).tz_convert('America/New_York').date().isoformat()==r['session_date_et']]
                nearest=min(closest,key=lambda s:abs((utc(s['decision_at_utc'])-utc(r['decision_at_utc'])).total_seconds())) if closest else None
                qs=[q for q in traces if q['symbol']==r['symbol'] and q.get('direction_sign')==r['direction_sign']
                    and abs((utc(q['decision_at_utc'])-utc(r['decision_at_utc'])).total_seconds())<=600]
                match_audit.append({'policy':policy.name,'partition':part,'source_message_ids':r['source_message_ids'],
                                    'source_entry_at_utc':r['decision_at_utc'],'symbol':r['symbol'],'direction_sign':r['direction_sign'],
                                    'nearest_entry_at_utc':nearest['decision_at_utc'] if nearest else None,
                                    'matched_one_to_one_within_2_candles':r['label_id'] in scored_matches,
                                    'matched_position_id':scored_matches[r['label_id']]['position_id'] if r['label_id'] in scored_matches else None,
                                    'nearby_gate_states':dict(Counter(q['state'] for q in qs))})
        metrics['expanded_all']={'entries':{str(k):entry_metrics(entries,[s for s in events if s['kind']=='entry'],k) for k in TOLERANCES},
                                 'management':lifecycle_metrics(entries,management,events,2),'path_diagnostics':trade_summary(trades)}
        metrics['primary_test_strata']={}
        for stratum,etf in [('index_etf',True),('single_stock',False)]:
            refs=[r for r in entries if r['primary_cohort'] and r['session_date_et'] in date_splits['test']
                  and (r['symbol'] in {'SPY','QQQ','IWM','GLD','SLV','XLE','XLF','XLP','LQD'})==etf]
            panels={(r['symbol'],r['session_date_et']) for r in refs}
            sig=[s for s in events if s['kind']=='entry' and (s['symbol'],utc(s['decision_at_utc']).tz_convert('America/New_York').date().isoformat()) in panels]
            metrics['primary_test_strata'][stratum]=entry_metrics(refs,sig,2)
        metrics['gate_states']=dict(Counter(t['state'] for t in traces))
        results['policies'][policy.name]=metrics
        all_events+=events;all_trades+=trades;all_traces+=traces
    args.out.mkdir(parents=True,exist_ok=False)
    input_dir=args.out/'normalized_daily_inputs';input_dir.mkdir()
    for symbol,frame in daily_input_frames.items():
        path=input_dir/f'{symbol}.parquet';frame.to_parquet(path,index=False)
    results['normalized_daily_inputs']={'directory':str(input_dir),
        'sha256':{p.name:digest(p) for p in input_dir.glob('*.parquet')},
        'rule':'immutable copy of validated daily input values, with derived ET session-date column; daily history remains strictly before each decision date'}
    for name,rows in [('entry_labels',entries),('management_labels',management),('published_plans',plans),('excluded_plans',excluded_plans),
                      ('algorithm_events',all_events),('algorithm_paths',all_trades),('state_trace',all_traces),('entry_match_audit',match_audit)]:
        write_jsonl(args.out/f'{name}.jsonl',rows)
    if args.bars_dir:
        cases,case_events=screenshot_case_diagnostics(args.bars_dir)
        write_jsonl(args.out/'screenshot_case_price_paths.jsonl',cases)
        write_jsonl(args.out/'screenshot_case_algorithm_events.jsonl',case_events)
    (args.out/'results.json').write_text(json.dumps(results,indent=2,sort_keys=True,allow_nan=False)+'\n')
    render_report(args.out,results)
    plot_representative_sessions(args.out,session_data,entries,all_events,date_splits['test'])
    print(json.dumps({'out':str(args.out),'source_audit':audit,'test':{p:r['primary']['test']['entries']['2'] for p,r in results['policies'].items()}},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
