"""Timing review for trade-history episodes: how early/late entries and exits were, and what the
options would have paid under other exits. Pure functions over loaded bars.

Underlying timing (split-adjusted daily bars; everything here except entry_style and
exit_trigger looks forward on purpose, so these are OUTCOMES for review, never entry signals):
  entry_style      known at entry: 'buying weakness' (directional 10-session return <= -4% and in the
                   lower 35% of the 20-session range), 'buying strength' (>= +4% and in the top 20%),
                   else 'in between'. Direction-adjusted: a put bought after a rally is buying strength
                   in the stock = buying weakness in the trade's direction, so it is classed by the trade's side.
  entry_pctile     entry price within the low..high of the 11 sessions centred on the entry session,
                   direction-adjusted (0 = bought the exact low of that window, 1 = the high).
  entry_wait_5d    best price in the next 5 sessions vs the entry, directional (-3% = could have paid 3% less).
  entry_prerun_5d  move from the 5-session extreme before entry up to the entry price (how much was already gone).
  exit_trigger     known at the exit: 'into weakness' (directional move since the prior close <= -1%, or
                   >= 3% below the best close/price seen since entry), 'into strength' (within 1% of it),
                   else 'in between'.
  exit_pctile      exit price within the 11 sessions centred on the exit (1 = sold the high).
  post_exit_Nd     directional move from the exit price to the close N sessions later (N = 1, 2, 5, 10).
  post_exit_best_10d / post_exit_worst_10d  best / worst directional excursion over the next 10 sessions.
  drift_2d         the same name's average directional 2-session move over the trade's surrounding
                   +-60 sessions: the baseline a "should have held 2 more days" result must beat.

Option counterfactuals (per contract position, fees as actually paid):
  hold_to_expiry_pnl  EXACT: at expiry an option is worth its intrinsic value at the expiry close.
  iv_entry / iv_exit  Black-Scholes implied vol backed out of the actual fills (European, no dividends,
                      r by year). Undefined when a fill printed below intrinsic (common for deep ITM).
  model_err_exit      log(model / actual) for the exit fill priced with iv_entry: how well the model
                      tracks real option prices in this account. Use it as the error bar on the rows below.
  hold_plus_Nd_pnl    MODEL: the same exits N sessions later, priced at that close with the exit's own IV.
  repo_policy_pnl     MODEL: the repo's live 4H exit policy (core.live_4h_exec.ExecPolicy defaults) applied
                      from the first entry fill on daily bars; long options only.
  take_profit_50_pnl  MODEL: sell the first daily close the option is up 50%, else the actual exit.
  roll_all_pnl        MODEL entry, EXACT payoff: at the exit, roll all proceeds into a 30-day ATM option held
                      to its expiry ("roll up and out").
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from core.personal_portfolio.data import ET
from core.personal_portfolio.trade_analysis import _factor, _fill_price_raw, _known_session, _session_of, direction_sign

# core.live_4h_exec.ExecPolicy defaults (2026-07-18 "tail-rider") and RiskPass underlying stop. Copied, not
# imported, so the read-only personal page never loads the trading stack; a test pins them to the source.
REPO_POLICY = {"underlying_stop_atr": 1.5, "take_profit": 0.30, "scale_frac": 0.16, "horizon_bars": 53,
               "horizon_sessions": 21, "stop_loss": 0.39}
RATE_BY_YEAR = {2019: 0.02, 2020: 0.005, 2021: 0.001, 2022: 0.02, 2023: 0.05, 2024: 0.05, 2025: 0.042, 2026: 0.038}
EXPIRY_CLOSE = time(16, 0)


# ---------- Black-Scholes ----------
def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(s: float, k: float, t: float, r: float, sigma: float, call: bool) -> float:
    if t <= 0 or sigma <= 0:
        return max(0.0, s - k) if call else max(0.0, k - s)
    d1 = (math.log(s / k) + (r + sigma * sigma / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if call:
        return s * _ncdf(d1) - k * math.exp(-r * t) * _ncdf(d2)
    return k * math.exp(-r * t) * _ncdf(-d2) - s * _ncdf(-d1)


def implied_vol(price: float, s: float, k: float, t: float, r: float, call: bool) -> float | None:
    """Bisection on [1%, 500%]. None when the price is outside what any vol can produce."""
    if not (price and s and k) or t <= 0:
        return None
    lo, hi = 0.01, 5.0
    if price < bs_price(s, k, t, r, lo, call) - 1e-9 or price > bs_price(s, k, t, r, hi, call):
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if bs_price(s, k, t, r, mid, call) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _rate(t: datetime) -> float:
    return RATE_BY_YEAR.get(t.year, 0.03)


def _years_to_expiry(t: datetime, expiry: str) -> float:
    end = datetime.combine(date.fromisoformat(expiry), EXPIRY_CLOSE, ET)
    return max(0.0, (end - t.astimezone(ET)).total_seconds() / (365 * 86400))


def _close_time(day: pd.Timestamp) -> datetime:
    return datetime.combine(day.date(), EXPIRY_CLOSE, ET)


# ---------- underlying timing ----------
def _dir_pctile(px: float, lo: float, hi: float, sign: int) -> float | None:
    if hi <= lo:
        return None
    p = (px - lo) / (hi - lo)
    return float(min(1.0, max(0.0, p if sign > 0 else 1 - p)))


def timing_metrics(ep: dict, ctx: dict, split: pd.DataFrame | None, raw: pd.DataFrame | None) -> dict:
    """Entry/exit timing on the underlying. ctx must hold trade_analysis.episode_features output."""
    if split is None or raw is None or ctx.get("under_entry") is None:
        return {}
    sign = direction_sign(ep)
    t0 = datetime.fromisoformat(ep["open_time"])
    idx = split.index
    known = _known_session(t0, idx)
    d0 = _session_of(t0, idx)
    fac0 = _factor(split, raw, d0)
    if known is None or d0 is None or fac0 is None:
        return {}
    e = ctx["under_entry"] * fac0
    out: dict = {}
    pre = split.loc[:known]
    if len(pre) >= 20:
        r10 = sign * (pre["close"].iloc[-1] / pre["close"].iloc[-11] - 1)
        hi20, lo20 = pre["high"].tail(20).max(), pre["low"].tail(20).min()
        pos = _dir_pctile(pre["close"].iloc[-1], lo20, hi20, sign)
        out.update(ret_10d_dir=float(r10), range_pos_20=pos)
        if pos is not None:
            out["entry_style"] = ("buying weakness" if r10 <= -0.04 and pos <= 0.35 else
                                  "buying strength" if r10 >= 0.04 and pos >= 0.80 else "in between")
    i0 = int(idx.get_loc(d0))
    win = split.iloc[max(0, i0 - 5): i0 + 6]
    if len(win) >= 8:
        out["entry_pctile"] = _dir_pctile(e, float(win["low"].min()), float(win["high"].max()), sign)
    after = split.iloc[i0 + 1: i0 + 6]
    if len(after) == 5:
        best = after["low"].min() if sign > 0 else after["high"].max()
        out["entry_wait_5d"] = float(sign * (best / e - 1))
    before = split.iloc[max(0, i0 - 5): i0]
    if len(before) == 5:
        ext = before["low"].min() if sign > 0 else before["high"].max()
        out["entry_prerun_5d"] = float(sign * (e / ext - 1))
    if ep["status"] != "closed" or ctx.get("under_exit") is None:
        return out
    exits = [f for f in ep["fills"] if f["role"] == "exit"]
    t1 = datetime.fromisoformat(exits[-1]["time"])
    d1 = _session_of(t1, idx)
    fac1 = _factor(split, raw, d1)
    if d1 is None or fac1 is None:
        return out
    x = ctx["under_exit"] * fac1
    i1 = int(idx.get_loc(d1))
    prior = split["close"].iloc[i1 - 1] if t1.astimezone(ET).time() < EXPIRY_CLOSE and i1 >= 1 else split["close"].iloc[i1]
    # the peak known at the exit: sessions before the exit day (their full range is in the past), the entry
    # and the exit price itself; the exit day's own high/low may print after the sale, so it is excluded
    held = split.iloc[i0: i1]
    cands = [e, x] + ([float(held["high"].max())] if sign > 0 and len(held) else
                      [float(held["low"].min())] if len(held) else [])
    peak = max(cands) if sign > 0 else min(cands)
    since_prior, off_peak = sign * (x / prior - 1), sign * (x / peak - 1)
    out.update(exit_day_move=float(since_prior), exit_off_peak=float(off_peak))
    if exits[-1].get("source") != "expiration":
        out["exit_trigger"] = ("into weakness" if since_prior <= -0.01 or off_peak <= -0.03 else
                               "into strength" if off_peak >= -0.01 else "in between")
    win = split.iloc[max(0, i1 - 5): i1 + 6]
    if len(win) >= 8:
        out["exit_pctile"] = _dir_pctile(x, float(win["low"].min()), float(win["high"].max()), sign)
    closes = split["close"].iloc[i1 + 1:]
    for n in (1, 2, 5, 10):
        if len(closes) >= n:
            out[f"post_exit_{n}d"] = float(sign * (closes.iloc[n - 1] / x - 1))
    nxt = split.iloc[i1 + 1: i1 + 11]
    if len(nxt) == 10:
        hi, lo = float(nxt["high"].max()), float(nxt["low"].min())
        out["post_exit_best_10d"] = float(sign * ((hi if sign > 0 else lo) / x - 1))
        out["post_exit_worst_10d"] = float(sign * ((lo if sign > 0 else hi) / x - 1))
    around = split["close"].iloc[max(0, i0 - 60): i1 + 61]
    fwd2 = sign * (around.shift(-2) / around - 1)
    if fwd2.notna().sum() >= 20:
        out["drift_2d"] = float(fwd2.mean())
    return out


# ---------- option counterfactuals ----------
def _intrinsic(s: float, k: float, call: bool) -> float:
    return max(0.0, s - k) if call else max(0.0, k - s)


def option_counterfactuals(ep: dict, raw: pd.DataFrame | None, intraday: pd.DataFrame | None, tf: str | None) -> dict:
    """Hold-to-expiry (exact), model checks, hold +N sessions and the repo exit policy (model)."""
    if ep["kind"] != "option" or ep["status"] != "closed" or raw is None or not ep.get("standard_deliverable", True):
        return {}
    k, call, mult = float(ep["strike"]), ep["put_call"] == "CALL", float(ep.get("multiplier") or 100.0)
    expiry = ep["expiry"]
    entries = [f for f in ep["fills"] if f["role"] == "entry"]
    exits = [f for f in ep["fills"] if f["role"] == "exit"]
    sold = [f for f in exits if f["source"] != "expiration"]
    out: dict = {}
    exp_day = pd.Timestamp(expiry)
    s_t = float(raw.at[exp_day, "close"]) if exp_day in raw.index else None
    if s_t is None:
        prior = raw.loc[:exp_day]
        # an expiry on a market holiday settles off the prior session; anything else is missing data
        if len(prior) and (exp_day - prior.index[-1]).days <= 3 and exp_day <= raw.index[-1]:
            s_t = float(prior["close"].iloc[-1])
    if s_t is not None:
        repl = sum(f["qty"] * mult * _intrinsic(s_t, k, call) for f in sold)   # qty<0 for a sale
        out.update(hold_to_expiry_pnl=float(ep["pnl"] - sum(f["cash"] for f in sold) - repl),
                   expiry_close=s_t)
    e0 = entries[0]
    t0 = datetime.fromisoformat(e0["time"])
    s0, _ = _fill_price_raw(ep, e0, intraday, tf, raw)
    iv0 = implied_vol(e0["price"], s0, k, _years_to_expiry(t0, expiry), _rate(t0), call) if s0 else None
    out["iv_entry"] = iv0
    if not sold:
        return out
    ivs, errs = [], []
    for f in sold:
        t = datetime.fromisoformat(f["time"])
        s, _ = _fill_price_raw(ep, f, intraday, tf, raw)
        if not s:
            ivs.append(None)
            continue
        tt = _years_to_expiry(t, expiry)
        ivs.append(implied_vol(f["price"], s, k, tt, _rate(t), call))
        if iv0 and f["price"] and f["price"] > 0.05:
            errs.append(math.log(max(bs_price(s, k, tt, _rate(t), iv0, call), 0.01) / f["price"]))
    out["iv_exit"] = next((v for v in reversed(ivs) if v), None)
    if errs:
        out["model_err_exit"] = float(np.mean(errs))
    for n in (1, 2, 5):
        total, ok = 0.0, True
        for f, iv in zip(sold, ivs):
            t = datetime.fromisoformat(f["time"])
            d = _session_of(t, raw.index)
            if d is None:
                ok = False
                break
            i = int(raw.index.get_loc(d)) + n
            day = raw.index[i] if i < len(raw.index) else None
            if day is None:
                ok = False
                break
            if day >= exp_day:
                if s_t is None:
                    ok = False
                    break
                v = _intrinsic(s_t, k, call)
            else:
                sigma = iv or iv0
                if not sigma:
                    ok = False
                    break
                ct = _close_time(day)
                v = bs_price(float(raw.at[day, "close"]), k, _years_to_expiry(ct, expiry), _rate(ct), sigma, call)
            total += f["qty"] * mult * v
        if ok:
            out[f"hold_plus_{n}d_pnl"] = float(ep["pnl"] - sum(f["cash"] for f in sold) - total)
    return out


def repo_policy_sim(ep: dict, ctx: dict, split: pd.DataFrame | None, raw: pd.DataFrame | None) -> dict:
    """The live 4H exit policy on this entry: underlying stop at entry - 1.5 ATR (checked on daily lows),
    trim 16% at +30% premium, ride the rest to 21 sessions, flatten on expiry day. MODEL-priced with iv_entry
    until the actual exit, then iv_exit (the last vol actually observed)."""
    if ep["kind"] != "option" or ep["direction"] != "long" or split is None or raw is None \
            or not ep.get("standard_deliverable", True) or not ctx.get("iv_entry") or ctx.get("under_entry") is None:
        return {}
    k, call = float(ep["strike"]), ep["put_call"] == "CALL"
    sign = 1 if call else -1
    e0 = [f for f in ep["fills"] if f["role"] == "entry"][0]
    t0, p0 = datetime.fromisoformat(e0["time"]), float(e0["price"])
    known, d0 = _known_session(t0, split.index), _session_of(t0, raw.index)
    if known is None or d0 is None:
        return {}
    pre = split.loc[:known].tail(15)
    if len(pre) < 15:
        return {}
    tr = np.maximum(pre["high"] - pre["low"], np.maximum((pre["high"] - pre["close"].shift()).abs(),
                                                         (pre["low"] - pre["close"].shift()).abs())).iloc[1:]
    fac = _factor(split, raw, d0)
    if not fac:
        return {}
    atr_raw = float(tr.mean()) / fac
    s0 = ctx["under_entry"]
    stop = s0 - sign * REPO_POLICY["underlying_stop_atr"] * atr_raw
    exp_day = pd.Timestamp(ep["expiry"])
    t_exit = datetime.fromisoformat(ep["close_time"])
    iv0, iv1 = ctx["iv_entry"], ctx.get("iv_exit") or ctx["iv_entry"]
    i0 = int(raw.index.get_loc(d0))
    left, value_out, trimmed = 1.0, 0.0, False
    reason, n_used = "data ended", 0
    for n, day in enumerate(raw.index[i0 + 1:], start=1):
        bar = raw.loc[day]
        n_used = n
        if (sign > 0 and bar["low"] <= stop) or (sign < 0 and bar["high"] >= stop):
            ct = datetime.combine(day.date(), time(12, 0), ET)
            sigma = iv0 if ct <= t_exit else iv1
            value_out += left * bs_price(stop, k, _years_to_expiry(ct, ep["expiry"]), _rate(ct), sigma, call)
            left, reason = 0.0, "underlying stop"
            break
        ct = _close_time(day)
        if day >= exp_day:
            value_out += left * _intrinsic(float(bar["close"]), k, call)
            left, reason = 0.0, "expiry"
            break
        sigma = iv0 if ct <= t_exit else iv1
        v = bs_price(float(bar["close"]), k, _years_to_expiry(ct, ep["expiry"]), _rate(ct), sigma, call)
        if not trimmed and v >= p0 * (1 + REPO_POLICY["take_profit"]):
            value_out += REPO_POLICY["scale_frac"] * v
            left -= REPO_POLICY["scale_frac"]
            trimmed = True
        if n >= REPO_POLICY["horizon_sessions"]:
            value_out += left * v
            left, reason = 0.0, "horizon"
            break
    if left > 0:
        return {"repo_policy_note": "bars ended before the policy closed the trade"}
    ret = value_out / p0 - 1
    return {"repo_policy_return": float(ret), "repo_policy_pnl": float(ret * abs(ep["cost"]) + ep["fees"]),
            "repo_policy_exit": reason, "repo_policy_sessions": n_used, "repo_policy_trimmed": trimmed}


def take_profit_pnl(ep: dict, ctx: dict, raw: pd.DataFrame | None, tp: float) -> float | None:
    """MODEL: sell at the first daily close where the option (priced with iv_entry) is up `tp`, else the actual
    exit. Long options only. Only closes strictly between the entry and exit sessions are checked."""
    if ep["kind"] != "option" or ep["direction"] != "long" or ep["status"] != "closed" or raw is None \
            or not ctx.get("iv_entry") or not ep.get("standard_deliverable", True):
        return None
    e0 = [f for f in ep["fills"] if f["role"] == "entry"][0]
    d0 = _session_of(datetime.fromisoformat(e0["time"]), raw.index)
    d1 = _session_of(datetime.fromisoformat(ep["close_time"]), raw.index)
    if d0 is None or d1 is None:
        return None
    k, call, p0 = float(ep["strike"]), ep["put_call"] == "CALL", float(e0["price"])
    for day in raw.index[(raw.index > d0) & (raw.index < d1)]:
        ct = _close_time(day)
        v = bs_price(float(raw.at[day, "close"]), k, _years_to_expiry(ct, ep["expiry"]), _rate(ct), ctx["iv_entry"], call)
        if v >= p0 * (1 + tp):
            return float((v / p0 - 1) * abs(ep["cost"]) + ep["fees"])
    return float(ep["pnl"])


def roll_pnl(ep: dict, ctx: dict, raw: pd.DataFrame | None, *, budget: str = "all", days: int = 30,
             iv_mult: float = 1.0, fee_per_contract: float = 0.65) -> float | None:
    """MODEL entry, EXACT payoff: at the actual exit, put the proceeds ('all') or only the profit ('profit')
    into an at-the-money option of the same type expiring `days` calendar days later, held to that expiry
    (intrinsic at the last close on or before it). Priced with iv_exit (else iv_entry) x iv_mult."""
    if ep["kind"] != "option" or ep["direction"] != "long" or ep["status"] != "closed" or raw is None \
            or ctx.get("under_exit") is None or ep["exit_reason"] == "expired":
        return None
    sigma = (ctx.get("iv_exit") or ctx.get("iv_entry") or 0) * iv_mult
    if not sigma:
        return None
    t1 = datetime.fromisoformat(ep["close_time"])
    expiry = pd.Timestamp((t1.astimezone(ET) + timedelta(days=days)).date())
    seen = raw.loc[:expiry]
    if seen.empty or seen.index[-1] < expiry - pd.Timedelta(days=4):
        return None                                                     # bars do not reach the expiry yet
    call, s = ep["put_call"] == "CALL", float(ctx["under_exit"])
    px = bs_price(s, s, days / 365, _rate(t1), sigma, call)
    cash = sum(f["cash"] for f in ep["fills"] if f["role"] == "exit") if budget == "all" else ep["pnl"]
    n = math.floor(cash / (px * 100)) if px > 0 and cash > 0 else 0
    s_t = float(seen["close"].iloc[-1])
    return float(ep["pnl"] + n * (_intrinsic(s_t, s, call) - px) * 100 - 2 * n * fee_per_contract)


ENTRY_VARIANTS = ("base", "delay1", "delay2", "retest2", "retest3", "retest5", "confirm2", "confirm4")


def entry_variant_returns(ep: dict, ctx: dict, raw: pd.DataFrame | None, split: pd.DataFrame | None) -> dict:
    """Return on premium (options, MODEL with iv_entry) or on price (stock, exact) for other entries, all with the
    trade's own holding period in sessions, exiting at a daily close:
      base       your entry (underlying at the fill), exit at the close of your exit session
      delayN     enter at the close N sessions later
      retestX    a limit X% better than your entry, filled in the next 5 sessions or skipped (None)
      confirmX   enter at the first close X% in your favour within 5 sessions, or skipped (None)
    Keys are prefixed 'ev_'. Empty when a split sits inside the window (raw prices would not be comparable)."""
    if raw is None or split is None or ep["status"] != "closed" or ctx.get("under_entry") is None:
        return {}
    opt = ep["kind"] == "option"
    if opt and (ep["direction"] != "long" or not ctx.get("iv_entry") or not ep.get("standard_deliverable", True)):
        return {}
    sign = direction_sign(ep)
    t0, t1 = datetime.fromisoformat(ep["open_time"]), datetime.fromisoformat(ep["close_time"])
    d0, d1 = _session_of(t0, raw.index), _session_of(t1, raw.index)
    if d0 is None or d1 is None:
        return {}
    i0, i1 = int(raw.index.get_loc(d0)), int(raw.index.get_loc(d1))
    hold = i1 - i0
    if i0 + 5 + hold + 1 >= len(raw):
        return {}
    ratio = (split["close"] / raw["close"]).iloc[i0: i0 + 6 + hold + 1]
    if (ratio.pct_change().abs() > 0.01).any():
        return {}
    s_e, k, call = float(ctx["under_entry"]), ep.get("strike"), ep.get("put_call") == "CALL"
    exp_day = pd.Timestamp(ep["expiry"]) if opt else None

    def value(s: float, t: datetime) -> float:
        return bs_price(s, k, _years_to_expiry(t, ep["expiry"]), _rate(t), ctx["iv_entry"], call) if opt else s

    def ret(s_in: float, t_in: datetime, j_out: int) -> float | None:
        day = raw.index[j_out]
        s_out = float(raw["close"].iloc[j_out])
        if not opt:
            return sign * (s_out / s_in - 1)
        v_in = value(s_in, t_in)
        v_out = _intrinsic(s_out, k, call) if day >= exp_day else value(s_out, _close_time(day))
        return v_out / v_in - 1 if v_in > 0.01 else None

    out = {"ev_base": ret(s_e, t0, i1)}
    for n in (1, 2):
        out[f"ev_delay{n}"] = ret(float(raw["close"].iloc[i0 + n]), _close_time(raw.index[i0 + n]), i0 + n + hold)
    for x in (2, 3, 5):
        lvl, out[f"ev_retest{x}"] = s_e * (1 - sign * x / 100), None
        for j in range(i0 + 1, i0 + 6):
            if (raw["low"].iloc[j] <= lvl) if sign > 0 else (raw["high"].iloc[j] >= lvl):
                out[f"ev_retest{x}"] = ret(lvl, datetime.combine(raw.index[j].date(), time(12), ET), j + hold)
                break
    for x in (2, 4):
        out[f"ev_confirm{x}"] = None
        for j in range(i0 + 1, i0 + 6):
            if sign * (float(raw["close"].iloc[j]) / s_e - 1) >= x / 100:
                out[f"ev_confirm{x}"] = ret(float(raw["close"].iloc[j]), _close_time(raw.index[j]), j + hold)
                break
    return out


def premium_stop_pnl(ep: dict, ctx: dict, raw: pd.DataFrame | None, frac: float = 0.5, max_dte: int = 14) -> float | None:
    """MODEL: sell at the first daily close (while you still held it) where the option is worth <= frac of the
    entry price with <= max_dte calendar days left; else the actual P&L. Long options only."""
    if ep["kind"] != "option" or ep["direction"] != "long" or ep["status"] != "closed" or raw is None \
            or not ctx.get("iv_entry") or not ep.get("standard_deliverable", True):
        return None
    e0 = [f for f in ep["fills"] if f["role"] == "entry"][0]
    d0 = _session_of(datetime.fromisoformat(e0["time"]), raw.index)
    d1 = _session_of(datetime.fromisoformat(ep["close_time"]), raw.index)
    if d0 is None or d1 is None:
        return None
    k, call, p0, exp_day = float(ep["strike"]), ep["put_call"] == "CALL", float(e0["price"]), pd.Timestamp(ep["expiry"])
    for day in raw.index[(raw.index > d0) & (raw.index < d1)]:
        if (exp_day - day).days > max_dte:
            continue
        ct = _close_time(day)
        v = bs_price(float(raw.at[day, "close"]), k, _years_to_expiry(ct, ep["expiry"]), _rate(ct), ctx["iv_entry"], call)
        if v <= frac * p0:
            return float((v / p0 - 1) * abs(ep["cost"]) + ep["fees"])
    return float(ep["pnl"])


def drawdown_outcome(ep: dict, ctx: dict, raw: pd.DataFrame | None, frac: float = 0.5) -> dict:
    """The first daily close (MODEL value) where the option is worth <= frac of its entry price, whether or not you
    still held it, and what it was worth at expiry (EXACT intrinsic). Keys prefixed 'dd{pct}_'."""
    if ep["kind"] != "option" or ep["direction"] != "long" or raw is None or not ctx.get("iv_entry") \
            or not ep.get("standard_deliverable", True):
        return {}
    exp_day = pd.Timestamp(ep["expiry"])
    if raw.index[-1] < exp_day:
        return {}
    e0 = [f for f in ep["fills"] if f["role"] == "entry"][0]
    d0 = _session_of(datetime.fromisoformat(e0["time"]), raw.index)
    if d0 is None:
        return {}
    k, call, p0 = float(ep["strike"]), ep["put_call"] == "CALL", float(e0["price"])
    s_t = float(raw.loc[:exp_day, "close"].iloc[-1])
    tag = f"dd{int(round((1 - frac) * 100))}_"
    for day in raw.index[(raw.index > d0) & (raw.index <= exp_day)]:
        ct = _close_time(day)
        v = bs_price(float(raw.at[day, "close"]), k, _years_to_expiry(ct, ep["expiry"]), _rate(ct), ctx["iv_entry"], call)
        if v <= frac * p0:
            held = ep["status"] != "closed" or datetime.fromisoformat(ep["close_time"]).date() > day.date()
            return {tag + "dte": int((exp_day - day).days), tag + "expiry_value": _intrinsic(s_t, k, call) / p0,
                    tag + "held_past": bool(held)}
    return {}


def review(ep: dict, ctx: dict, split, raw, intraday, tf) -> dict:
    """All timing and counterfactual fields for one episode (merged into ctx by trade_analysis.enrich)."""
    out = timing_metrics(ep, ctx, split, raw)
    out.update(option_counterfactuals(ep, raw, intraday, tf))
    full = {**ctx, **out}
    out.update(repo_policy_sim(ep, full, split, raw))
    out["take_profit_50_pnl"] = take_profit_pnl(ep, full, raw, 0.5)
    out["roll_all_pnl"] = roll_pnl(ep, full, raw, budget="all")
    out["stop50_late_pnl"] = premium_stop_pnl(ep, full, raw, 0.5, 14)
    out.update(entry_variant_returns(ep, full, raw, split))
    out.update(drawdown_outcome(ep, full, raw, 0.5))
    return out


def comeback_stats(episodes: list[dict], max_dte: int = 14) -> dict | None:
    """From history: options that fell to half their entry price with <= max_dte days left. Feeds the page flag."""
    rows = [e["ctx"] for e in episodes if (e.get("ctx") or {}).get("dd50_dte") is not None
            and e["ctx"]["dd50_dte"] <= max_dte]
    stopped = [e for e in episodes if (e.get("ctx") or {}).get("stop50_late_pnl") is not None
               and e["status"] == "closed" and not e.get("incomplete") and e["ctx"]["stop50_late_pnl"] != e["pnl"]]
    if len(rows) < 10:
        return None
    vals = np.array([r["dd50_expiry_value"] for r in rows])
    return {"n": len(rows), "max_dte": max_dte, "breakeven_share": float((vals >= 1).mean()),
            "worthless_share": float((vals == 0).mean()), "avg_expiry_value": float(vals.mean()),
            "n_stopped": len(stopped),
            "stop_better_share": float(np.mean([e["ctx"]["stop50_late_pnl"] > e["pnl"] for e in stopped])) if stopped else None}
