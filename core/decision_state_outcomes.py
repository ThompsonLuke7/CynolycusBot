"""Forward outcomes for archived decisions (``decision_state_outcomes_v1``).

Kept apart from the raw archive on purpose: the archive is what was known at
decision time and is never rewritten; outcomes are derived later, from bars
that did not exist yet, and are recomputed until every horizon has matured.

Definitions (all from an EXECUTABLE entry, never the decision bar's own close):

* entry: the open of the first 1H bar whose start is at or after the
  decision's wall-clock time (``decision_ts``). The 1H cache starts its session
  at 10:00 ET, so a pre-open decision enters at 10:00 ET.
* horizon h (sessions 1/5/10/20): exit at the close of the last 1H bar of the
  session h sessions after the entry session.
* ``fret_h`` close/entry - 1, ``mfe_h`` max high/entry - 1, ``mae_h`` min
  low/entry - 1, over the entry bar through the exit bar; ``t_mfe_h`` sessions
  from entry to the bar of the max high. Raw long-direction numbers: the
  archive's ``side`` says which way the module meant it.
* first passage ``fp_h``: barriers at entry x (1 +/- ``FP_ATR_MULT`` x ATR%),
  ATR% = 14-day ATR over close from daily bars strictly before the entry
  session. +1 upper first, -1 lower first, 0 neither by the exit. A bar that
  touches both counts as -1: with hourly bars the order is unknowable, and the
  adverse reading is the one that cannot flatter a module.
* ``spy_fret_h``: SPY over the same entry timestamp and exit session.
* status per horizon: ``ok``; ``immature`` (exit session not complete yet);
  ``void_corporate_action`` (a non-organic gap inside the window: the 1H cache
  is not split-adjusted); ``no_exit_bar``; ``no_entry``; ``entry_gap`` (first
  bar after the decision is more than one session later); ``no_bars``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

OUTCOME_SCHEMA = "decision_state_outcomes_v1"
HORIZONS = (1, 5, 10, 20)
FP_ATR_MULT = 1.0
ATR_DAYS = 14
FINAL_STATUSES = {"ok", "void_corporate_action", "no_exit_bar", "no_entry", "entry_gap", "no_bars"}


def session_dates(ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(ts)
    return idx.tz_convert("America/New_York").normalize().tz_localize(None)


def last_complete_session(calendar: pd.DatetimeIndex, now: datetime | None = None) -> pd.Timestamp | None:
    """Latest session whose regular session (and after-hours bar) is over."""
    if len(calendar) == 0:
        return None
    now_et = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("America/New_York")
    today = now_et.normalize().tz_localize(None)
    done = calendar[calendar < today] if now_et.hour < 17 else calendar[calendar <= today]
    return done.max() if len(done) else None


def lagged_atr_pct(daily: pd.DataFrame | None) -> pd.Series:
    """ATR% by session date, usable for decisions on the NEXT session onward."""
    if daily is None or daily.empty:
        return pd.Series(dtype=float)
    d = daily.sort_values("timestamp")
    prev = d["close"].shift(1)
    tr = pd.concat([(d["high"] - d["low"]), (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_DAYS, min_periods=ATR_DAYS).mean() / d["close"]
    return pd.Series(atr.to_numpy(), index=session_dates(d["timestamp"]))


def _empty_row(decision: pd.Series, horizons: Iterable[int], status: str) -> dict:
    row = {"run_id": decision["run_id"], "ticker": decision["ticker"], "decision_ts": decision["decision_ts"],
           "entry_ts": pd.NaT, "entry_px": np.nan, "atr_pct_lag": np.nan}
    for h in horizons:
        row.update({f"fret_{h}": np.nan, f"mfe_{h}": np.nan, f"mae_{h}": np.nan, f"t_mfe_{h}": np.nan,
                    f"fp_{h}": np.nan, f"fp_sessions_{h}": np.nan, f"spy_fret_{h}": np.nan,
                    f"status_{h}": status})
    return row


def ticker_outcomes(
    decisions: pd.DataFrame,
    bars: pd.DataFrame | None,
    *,
    calendar: pd.DatetimeIndex,
    last_complete: pd.Timestamp | None,
    spy: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    flagged: set[pd.Timestamp] | None = None,
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    """Outcomes for one ticker's archived decisions (``run_id``, ``ticker``,
    ``decision_ts``). ``bars``/``spy`` are 1H bars with a UTC ``timestamp``."""
    horizons = tuple(horizons)
    if bars is None or bars.empty:
        return pd.DataFrame([_empty_row(r, horizons, "no_bars") for _, r in decisions.iterrows()])
    flagged = flagged or set()
    b = bars.sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(b["timestamp"])
    sess_idx = calendar.get_indexer(session_dates(ts))
    keep = sess_idx >= 0
    b, ts, sess_idx = b.loc[keep].reset_index(drop=True), ts[keep], sess_idx[keep]
    opens, highs, lows, closes = (b[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
    last_bar_of = pd.Series(np.arange(len(b))).groupby(sess_idx).max().to_dict()
    flagged_idx = {int(i) for i in calendar.get_indexer(sorted(flagged)) if i >= 0}

    s = spy.sort_values("timestamp").reset_index(drop=True)
    s_ts = pd.DatetimeIndex(s["timestamp"])
    s_sess = calendar.get_indexer(session_dates(s_ts))
    s_last = pd.Series(np.arange(len(s))).groupby(s_sess).max().to_dict()
    s_open, s_close = s["open"].to_numpy(dtype=float), s["close"].to_numpy(dtype=float)

    atr = lagged_atr_pct(daily)
    complete_idx = int(calendar.get_loc(last_complete)) if last_complete is not None and last_complete in calendar else -1

    rows = []
    for _, dec in decisions.iterrows():
        dts = pd.Timestamp(dec["decision_ts"])
        pos = int(ts.searchsorted(dts, side="left"))
        # First session on or after the decision's ET date (a weekend decision
        # belongs to Monday).
        dec_sess = int(calendar.searchsorted(session_dates([dts])[0]))
        if pos >= len(b):
            # No bar since the decision: final only once a full session has passed.
            rows.append(_empty_row(dec, horizons, "no_entry" if dec_sess + 1 <= complete_idx else "immature"))
            continue
        entry_sess = int(sess_idx[pos])
        if entry_sess - dec_sess > 1:
            rows.append(_empty_row(dec, horizons, "entry_gap"))
            continue
        entry = opens[pos]
        prior = atr[atr.index < calendar[entry_sess]]
        atr_pct = float(prior.iloc[-1]) if len(prior) and np.isfinite(prior.iloc[-1]) else np.nan
        row = {"run_id": dec["run_id"], "ticker": dec["ticker"], "decision_ts": dts,
               "entry_ts": ts[pos], "entry_px": entry, "atr_pct_lag": atr_pct}
        s_pos = int(s_ts.searchsorted(ts[pos], side="left"))
        for h in horizons:
            target = entry_sess + h
            out = {f"fret_{h}": np.nan, f"mfe_{h}": np.nan, f"mae_{h}": np.nan, f"t_mfe_{h}": np.nan,
                   f"fp_{h}": np.nan, f"fp_sessions_{h}": np.nan, f"spy_fret_{h}": np.nan}
            if target > complete_idx or target >= len(calendar):
                status = "immature"
            elif any(entry_sess < f <= target for f in flagged_idx):
                status = "void_corporate_action"
            elif target not in last_bar_of:
                status = "no_exit_bar"
            else:
                status = "ok"
                end = int(last_bar_of[target])
                w_hi, w_lo = highs[pos:end + 1], lows[pos:end + 1]
                imax = int(np.argmax(w_hi))
                out.update({
                    f"fret_{h}": closes[end] / entry - 1.0,
                    f"mfe_{h}": w_hi.max() / entry - 1.0,
                    f"mae_{h}": w_lo.min() / entry - 1.0,
                    f"t_mfe_{h}": float(sess_idx[pos + imax] - entry_sess),
                })
                if np.isfinite(atr_pct) and atr_pct > 0:
                    up = np.flatnonzero(w_hi >= entry * (1 + FP_ATR_MULT * atr_pct))
                    dn = np.flatnonzero(w_lo <= entry * (1 - FP_ATR_MULT * atr_pct))
                    first_up = up[0] if len(up) else None
                    first_dn = dn[0] if len(dn) else None
                    if first_up is None and first_dn is None:
                        fp, hit = 0.0, None
                    elif first_up is None or (first_dn is not None and first_dn <= first_up):
                        fp, hit = -1.0, first_dn
                    else:
                        fp, hit = 1.0, first_up
                    out[f"fp_{h}"] = fp
                    out[f"fp_sessions_{h}"] = float(sess_idx[pos + hit] - entry_sess) if hit is not None else np.nan
                if s_pos < len(s) and target in s_last:
                    out[f"spy_fret_{h}"] = s_close[int(s_last[target])] / s_open[s_pos] - 1.0
            out[f"status_{h}"] = status
            row.update(out)
        rows.append(row)
    return pd.DataFrame(rows)


def is_final(outcomes: pd.DataFrame, horizons: Iterable[int] = HORIZONS) -> bool:
    """True once no horizon can change any more (nothing immature)."""
    cols = [f"status_{h}" for h in horizons if f"status_{h}" in outcomes.columns]
    if outcomes.empty or not cols:
        return False
    return bool(outcomes[cols].isin(FINAL_STATUSES).all().all())
