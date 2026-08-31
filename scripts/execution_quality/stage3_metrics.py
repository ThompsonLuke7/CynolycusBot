"""Stage 3 — attach dense price paths and compute the A/B/C decomposition.

Everything is measured on the UNDERLYING, never on option prices. That is not a
convenience: `research/options_experiment/10_RETRACTION_option_pnl_invalid.md`
records an entire study retracted because option "prices" were stale trade
prints (corr with the underlying was +0.09). Option P&L is carried alongside as
an outcome; it is never the timing signal.

All price differences are normalised by the ticker's daily ATR(14) as of the
session BEFORE the decision, so a $6 name and a $600 name are comparable and the
normaliser itself contains no look-ahead.

Definitions (long; shorts are mirrored):

  A. signal value   fwd MFE/MAE from `available_at` over fixed horizons
  B. entry cost     entry_slip  = (P_fill - P_avail)/ATR      -- paid up vs the signal
                    phase_error = T_fill - T_move_start        -- early (<0) or late (>0)
                    pre_entry_adverse = drawdown suffered before the move began
                    missed_leg  = the part of the move already spent at T_fill
  C. exit cost      giveback    = (MFE_since_fill - realized)/ATR
                    prematurity = MFE in the H after the exit, from the exit price

`T_move_start` is the left edge of the maximum-gain contiguous subinterval of log
returns (Kadane) inside the evaluation window — the start of the real move,
rather than an arbitrary lookback low.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "research/execution_quality/data"
BARS = DATA / "bars_1m"
DAILY = REPO_ROOT / "Data/shared/bars/1d"

_cache: dict[str, pd.DataFrame | None] = {}
_atr: dict[str, pd.Series | None] = {}


def P(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def bars(ticker: str) -> pd.DataFrame | None:
    if ticker in _cache:
        return _cache[ticker]
    path = BARS / f"{ticker}.parquet"
    df = None
    if path.exists():
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        # RTH only. Two reasons: an after-hours print on a thin name is a stale
        # print rather than a mark, and it would inflate MFE/MAE; and with the
        # extended session included, "N bars" stops meaning "N trading minutes",
        # which the fixed horizons rely on.
        et = df["timestamp"].dt.tz_convert("America/New_York")
        mins = et.dt.hour * 60 + et.dt.minute
        df = df.loc[(mins >= 570) & (mins < 960)]
        df = df.set_index("timestamp")
    _cache[ticker] = df
    return df


def atr_series(ticker: str) -> pd.Series | None:
    """Daily ATR(14) indexed by session date. Shifted by one session at use
    time so the normaliser never sees the decision day."""
    if ticker in _atr:
        return _atr[ticker]
    path = DAILY / f"{ticker}.parquet"
    out = None
    if path.exists():
        d = pd.read_parquet(path)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.sort_values("timestamp")
        prev_close = d["close"].shift(1)
        tr = pd.concat([
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        out = pd.Series(tr.rolling(14, min_periods=5).mean().values,
                        index=d["timestamp"].dt.tz_convert("America/New_York").dt.date)
    _atr[ticker] = out
    return out


def atr_at(ticker: str, when: datetime) -> float:
    s = atr_series(ticker)
    if s is None or when is None:
        return float("nan")
    day = when.astimezone(timezone.utc).date()
    prior = s[s.index < day]
    if prior.empty:
        return float("nan")
    v = float(prior.iloc[-1])
    return v if math.isfinite(v) and v > 0 else float("nan")


def window(ticker: str, lo: datetime, hi: datetime) -> pd.DataFrame | None:
    df = bars(ticker)
    if df is None or lo is None or hi is None or hi <= lo:
        return None
    w = df.loc[(df.index >= lo) & (df.index <= hi)]
    return w if len(w) else None


def price_at(ticker: str, when: datetime, *, tol_min: int = 30) -> float:
    """Last trade price at or before `when`, refusing a stale print."""
    df = bars(ticker)
    if df is None or when is None:
        return float("nan")
    prior = df.loc[df.index <= when]
    if prior.empty:
        return float("nan")
    age = (when - prior.index[-1]).total_seconds() / 60.0
    if age > tol_min:
        return float("nan")
    return float(prior["close"].iloc[-1])


def move_start(w: pd.DataFrame, sign: int, peak_ts: datetime | None) -> datetime | None:
    """When did the move WE were positioned for actually begin?

    Anchored on our own trade rather than on the biggest run anywhere in the
    window: take the favourable peak reached during the hold, then walk back to
    the extreme immediately preceding it. An unanchored Kadane over a 20-day hold
    finds the largest run in the whole period, which may start days after entry
    and has nothing to do with whether this entry was early or late.
    """
    if w is None or len(w) < 2 or peak_ts is None:
        return None
    pre = w.loc[w.index <= peak_ts]
    if len(pre) < 2:
        return None
    series = pre["low"] if sign > 0 else pre["high"]
    idx = series.idxmin() if sign > 0 else series.idxmax()
    return idx.to_pydatetime()


def excursions(w: pd.DataFrame, ref: float, sign: int) -> tuple[float, float, datetime | None]:
    """(MFE, MAE, time-of-MFE) in price units relative to `ref`."""
    if w is None or not math.isfinite(ref) or ref <= 0:
        return float("nan"), float("nan"), None
    if sign > 0:
        fav = w["high"].to_numpy(dtype=float) - ref
        adv = ref - w["low"].to_numpy(dtype=float)
    else:
        fav = ref - w["low"].to_numpy(dtype=float)
        adv = w["high"].to_numpy(dtype=float) - ref
    i = int(np.nanargmax(fav)) if len(fav) else 0
    return float(np.nanmax(fav)), float(np.nanmax(adv)), w.index[i].to_pydatetime()


HORIZONS = {"30m": 30, "2h": 120, "1d": 390, "3d": 1170, "10d": 3900}

# How long the order policy could plausibly wait for a better entry, per module,
# scaled to how long that module actually holds. Trading minutes.
WAIT_WINDOW = {
    "spy_daytrader": 30,
    "multi_ticker_swing": 195,
    "dealer_ranker": 390,
    "momentum_expansion": 390,
    "meta_ranker": 390,
    "multi_ticker_swing_htf": 390,
}


def signal_rows():
    rows = []
    for line in (DATA / "stage2_signal_spine.jsonl").open():
        r = json.loads(line)
        if not r.get("submit"):
            continue
        t, avail = r["ticker"], P(r["available_at"])
        sign = -1 if str(r.get("side", "long")).lower() in ("short", "sell") else 1
        a = atr_at(t, avail)
        ref = price_at(t, avail)
        out = dict(r)
        out["atr"] = a
        out["ref_price"] = ref
        if math.isfinite(a) and math.isfinite(ref):
            for name, mins in HORIZONS.items():
                w = window(t, avail, avail + timedelta(minutes=mins * 3))
                if w is None:
                    continue
                # `mins` counts TRADING minutes: walk that many bars forward.
                w = w.iloc[:mins] if len(w) > mins else w
                mfe, mae, _ = excursions(w, ref, sign)
                out[f"mfe_{name}_atr"] = mfe / a
                out[f"mae_{name}_atr"] = mae / a
                out[f"ret_{name}_atr"] = (float(w["close"].iloc[-1]) - ref) * sign / a
        rows.append(out)
    return rows


def trade_rows():
    rows = []
    for line in (DATA / "stage1_trade_spine.jsonl").open():
        r = json.loads(line)
        if not r.get("module"):
            continue
        t = r["ticker"]
        fill = P(r["first_entry_fill"])
        avail_real = P(r.get("signal_available_at"))
        avail = avail_real or fill
        exit_t = P(r.get("exit_last_fill"))
        sign = -1 if str(r.get("signal_side", "long")).lower() in ("short", "sell") else 1
        a = atr_at(t, fill)
        out = dict(r)
        out["atr"] = a
        p_avail = price_at(t, avail)
        p_fill = price_at(t, fill)
        p_exit = price_at(t, exit_t) if exit_t else float("nan")
        out.update({"u_avail": p_avail, "u_fill": p_fill, "u_exit": p_exit})
        if not (math.isfinite(a) and math.isfinite(p_fill)):
            rows.append(out)
            continue

        # B — entry cost. Only meaningful where the module actually stamped an
        # availability time; defaulting avail=fill would manufacture a zero.
        if avail_real is not None and math.isfinite(p_avail):
            out["entry_slip_atr"] = (p_fill - p_avail) * sign / a
            out["signal_to_fill_min"] = round((fill - avail_real).total_seconds() / 60.0, 1)
        # Oracle entry: the best price the policy could have paid inside a
        # realistic waiting window after the signal became available.
        wait = WAIT_WINDOW.get(r["module"], 390)
        w_wait = window(t, avail, avail + timedelta(minutes=wait * 3))
        if w_wait is not None:
            w_wait = w_wait.iloc[:wait] if len(w_wait) > wait else w_wait
            series = w_wait["low"] if sign > 0 else w_wait["high"]
            o_idx = series.idxmin() if sign > 0 else series.idxmax()
            p_oracle = float(series.loc[o_idx])
            out["oracle_entry_px"] = p_oracle
            out["oracle_entry_lag_min"] = round((o_idx.to_pydatetime() - avail).total_seconds() / 60.0, 1)
            out["entry_vs_oracle_atr"] = (p_fill - p_oracle) * sign / a

        # C — exit cost (computed first: the peak anchors the move-start)
        tpk = None
        if exit_t:
            w_hold = window(t, fill, exit_t)
            if w_hold is not None:
                mfe, mae, tpk = excursions(w_hold, p_fill, sign)
                out["mfe_hold_atr"] = max(0.0, mfe) / a
                out["mae_hold_atr"] = max(0.0, mae) / a
                out["time_to_peak_min"] = round((tpk - fill).total_seconds() / 60.0, 1) if tpk else None
                if math.isfinite(p_exit):
                    realized = (p_exit - p_fill) * sign / a
                    out["realized_move_atr"] = realized
                    out["giveback_atr"] = max(0.0, mfe) / a - realized
                    out["hold_efficiency"] = realized / (mfe / a) if mfe > 0 else None

        # B — early or late, anchored on the peak this trade actually reached
        w_eval = window(t, fill - timedelta(minutes=390), exit_t or (fill + timedelta(minutes=390)))
        ms = move_start(w_eval, sign, tpk or (exit_t or fill))
        if ms is not None:
            out["move_start"] = ms.isoformat()
            out["phase_error_min"] = round((fill - ms).total_seconds() / 60.0, 1)
            p_ms = price_at(t, ms)
            if math.isfinite(p_ms):
                if fill >= ms:
                    out["missed_leg_atr"] = (p_fill - p_ms) * sign / a
                else:
                    w_pre = window(t, fill, ms)
                    if w_pre is not None:
                        _, mae_pre, _ = excursions(w_pre, p_fill, sign)
                        out["pre_entry_adverse_atr"] = max(0.0, mae_pre) / a
        if exit_t:
            for name, mins in (("1d", 390), ("3d", 1170), ("10d", 3900)):
                w_post = window(t, exit_t, exit_t + timedelta(minutes=mins * 3))
                if w_post is None or not math.isfinite(p_exit):
                    continue
                w_post = w_post.iloc[:mins] if len(w_post) > mins else w_post
                mfe_p, _, _ = excursions(w_post, p_exit, sign)
                out[f"prematurity_{name}_atr"] = max(0.0, mfe_p) / a
        rows.append(out)
    return rows


def main() -> None:
    sr = signal_rows()
    with (DATA / "stage3_signal_metrics.jsonl").open("w") as fh:
        for r in sr:
            fh.write(json.dumps(r, default=str) + "\n")
    have = sum(1 for r in sr if r.get("mfe_1d_atr") is not None)
    print(f"signal metrics: {len(sr)} rows, {have} with a 1d path")

    tr = trade_rows()
    with (DATA / "stage3_trade_metrics.jsonl").open("w") as fh:
        for r in tr:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"trade metrics : {len(tr)} rows, "
          f"{sum(1 for r in tr if r.get('entry_slip_atr') is not None)} with entry slip, "
          f"{sum(1 for r in tr if r.get('giveback_atr') is not None)} with giveback, "
          f"{sum(1 for r in tr if r.get('phase_error_min') is not None)} with phase error")


if __name__ == "__main__":
    main()
