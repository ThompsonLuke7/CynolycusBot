"""Entry context and outcome measures for trade-history episodes. Pure functions over loaded bars.

Time discipline: entry features use only what was known at the first entry fill.
  * Daily features (returns, SMAs, highs, SPY regime) end at the last session whose close
    (16:00 ET) precedes the fill, on split-adjusted bars.
  * The underlying's price at an option fill is the OPEN of the intraday bar containing the
    fill (never a later print); for stocks it is the fill price itself.
  * Raw prices are put on the split-adjusted scale with the fill session's adjustment
    factor (split close / raw close). Splits are announced in advance, so that is not leakage.
Outcome measures (underlying move while held, excursions, drift after exit) look forward by
design and are labelled as outcomes; they are never used as entry features.

Earnings dates (Yahoo, Data/personal/earnings.parquet) are scheduled in advance, so "days to
the next report at entry" is known at entry. Yahoo gives the report date, not before/after the
bell: "held through" means opened on or before the report date and closed after it.

Bucket labels are computed here so the page and the report group trades identically.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from core.personal_portfolio.data import ET
from core.personal_portfolio.trade_bars import load_daily, load_intraday, price_at

INDEX_ETFS = {"SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "XLF", "XLE", "XLK", "SMH", "SOXX", "ARKK", "GLD", "SLV",
              "TLT", "VXX", "UVXY", "EEM", "FXI", "KWEB", "XBI", "XLV", "XLY", "XLI", "KRE", "IBIT"}
# Option descriptions name the underlying's issuer, not its type ("PROSHR QQQ 3X 12/19/2025 $60 Call",
# "ABRDN PLATINUM PHYS ..."), so the class is read from those words. 1X funds are not leveraged.
_LEVERAGED_RE = re.compile(r"\b(?:[2-9]|1\.\d+)XL?\b|(?<=[A-Z.])(?:[2-9]|1\.\d+)X\b|\bULTRA|\bDAILY TARGET\b"
                           r"|\bBULL\b|\bBEAR\b", re.I)
_ETF_RE = re.compile(r"\bETF\b|\bETN\b|\bFUND\b|\bTRUST\b|\bTR\b|\bSPDR\b|\bISHR|\bISHARES\b|\bPROSH|\bROUNDHILL\b"
                     r"|\bABRDN\b|\bINVESCO\b|\bDIREXION\b|\bDRXN|\bGRANITESHARES\b|\bDEFIANCE\b|\bTRADR\b"
                     r"|\bTRDR\b|\bVANECK\b|\bWISDOMTREE\b|\bGLOBAL X\b|\bFUTURES\b|\bFUTRS\b", re.I)
SESSION_CLOSE = time(16, 0)


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _known_session(t: datetime, idx: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Last session in idx whose close is at or before t."""
    et = t.astimezone(ET)
    cutoff = pd.Timestamp(et.date()) if et.time() >= SESSION_CLOSE else pd.Timestamp(et.date()) - pd.Timedelta(days=1)
    i = int(idx.searchsorted(cutoff, side="right")) - 1
    return idx[i] if i >= 0 else None


def _session_of(t: datetime, idx: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The session t falls in (its ET date), or the last session before it."""
    d = pd.Timestamp(t.astimezone(ET).date())
    i = int(idx.searchsorted(d, side="right")) - 1
    return idx[i] if i >= 0 else None


def _factor(split: pd.DataFrame, raw: pd.DataFrame, day: pd.Timestamp | None) -> float | None:
    if day is None or day not in split.index or day not in raw.index or not raw.at[day, "close"]:
        return None
    return float(split.at[day, "close"] / raw.at[day, "close"])


def _ret(s: pd.Series, n: int) -> float | None:
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1) if len(s) > n and s.iloc[-1 - n] else None


def _vs_sma(s: pd.Series, n: int) -> float | None:
    return float(s.iloc[-1] / s.tail(n).mean() - 1) if len(s) >= n else None


def direction_sign(ep: dict) -> int:
    """+1 when the trade profits from the underlying rising (long stock, long call, short put)."""
    up = ep["put_call"] != "PUT"
    long = ep["direction"] == "long"
    return 1 if up == long else -1


def underlying_class(ep: dict) -> str:
    if _LEVERAGED_RE.search(str(ep.get("description") or "")):
        return "leveraged ETF"
    if ep["kind"] == "etf" or ep["chart_symbol"] in INDEX_ETFS or _ETF_RE.search(str(ep.get("description") or "")):
        return "ETF"
    return "stock"


def _fill_price_raw(ep: dict, fill: dict, intraday: pd.DataFrame | None, tf: str | None,
                    raw: pd.DataFrame | None) -> tuple[float | None, str]:
    if ep["kind"] != "option" and fill.get("price"):
        return float(fill["price"]), "fill price"
    t = _ts(fill["time"])
    if fill.get("source") == "expiration" and raw is not None:
        day = _session_of(t, raw.index)
        return (float(raw.at[day, "close"]), "close on expiry day") if day is not None else (None, "no bars")
    px, how = price_at(intraday, t, tf)
    if px is not None:
        return px, how
    if raw is not None:
        day = _known_session(t, raw.index)
        if day is not None:
            return float(raw.at[day, "close"]), "prior daily close (no intraday bars)"
    return None, "no bars"


def bars_problem(ep: dict, split: pd.DataFrame, raw: pd.DataFrame) -> str | None:
    """Why these bars cannot be trusted for this trade, or None.

    Two failures seen in real data (2026-11-17): Alpaca applying a reverse split a day early (HIMZ 2026-03:
    the split-adjusted close fell 16x on 3/18 while the raw jump came 3/19), and a ticker whose bars are a
    different security than the one held (a $0.15 WISA fill against $3.50 WISA bars). P&L is unaffected
    (it is Schwab cash); price-based features and timing are skipped.
    """
    t_first = datetime.fromisoformat(ep["fills"][0]["time"])
    t_last = datetime.fromisoformat(ep["fills"][-1]["time"])
    d0, d1 = _session_of(t_first, raw.index), _session_of(t_last, raw.index)
    if d0 is None:
        return None
    i0, i1 = int(raw.index.get_loc(d0)), int(raw.index.get_loc(d1)) if d1 is not None else len(raw) - 1
    w = slice(raw.index[max(0, i0 - 60)], raw.index[min(len(raw) - 1, i1 + 11)])
    ratio = (split["close"] / raw["close"]).loc[w]
    jump = np.log(split["close"].loc[w]).diff().abs() > math.log(3)
    moved = ratio.pct_change().abs() > 0.01
    near = moved | moved.shift(1, fill_value=False) | moved.shift(-1, fill_value=False) \
        | moved.shift(2, fill_value=False) | moved.shift(-2, fill_value=False)
    bad = jump & near
    if bad.any():
        return f"split adjustment misaligned around {bad.idxmax().date().isoformat()}"
    if ep["kind"] != "option":
        for f in ep["fills"]:
            if f.get("source") != "trade" or not f.get("price"):
                continue
            day = _session_of(datetime.fromisoformat(f["time"]), raw.index)
            if day is None or day.date().isoformat() != datetime.fromisoformat(f["time"]).astimezone(ET).date().isoformat():
                continue
            lo, hi = float(raw.at[day, "low"]), float(raw.at[day, "high"])
            if not 0.85 * lo <= f["price"] <= 1.15 * hi:
                return (f"fill {f['price']:g} is outside {ep['chart_symbol']}'s bar ({lo:g}-{hi:g}) on {day.date()}: "
                        "the bars are not this security")
    return None


def episode_features(ep: dict, split: pd.DataFrame | None, raw: pd.DataFrame | None, spy: pd.DataFrame | None,
                     intraday: pd.DataFrame | None, tf: str | None) -> dict:
    out: dict = {"underlying_class": underlying_class(ep)}
    if split is not None and raw is not None and ep.get("fills"):
        problem = bars_problem(ep, split, raw)
        if problem:
            out["feature_note"] = problem
            return out
    t0 = _ts(ep["open_time"])
    et0 = t0.astimezone(ET)
    out.update(entry_hour=et0.hour + et0.minute / 60, entry_weekday=et0.strftime("%a"))
    entries = [f for f in ep["fills"] if f["role"] == "entry"]
    exits = [f for f in ep["fills"] if f["role"] == "exit"]
    if split is None or raw is None or not entries:
        out["feature_note"] = "no daily bars for the underlying"
        return out
    known = _known_session(t0, split.index)
    if known is None:
        out["feature_note"] = "no bars before the entry"
        return out
    pre = split.loc[:known]
    c, h = pre["close"], pre["high"]
    out.update(ret_5d=_ret(c, 5), ret_20d=_ret(c, 20), ret_60d=_ret(c, 60),
               vs_sma20=_vs_sma(c, 20), vs_sma50=_vs_sma(c, 50), vs_sma200=_vs_sma(c, 200),
               dist_high_20=float(c.iloc[-1] / h.tail(20).max() - 1) if len(h) >= 20 else None,
               dist_high_252=float(c.iloc[-1] / h.tail(252).max() - 1) if len(h) >= 120 else None,
               vol_20=float(c.pct_change().tail(20).std() * math.sqrt(252)) if len(c) >= 21 else None,
               known_session=known.date().isoformat())
    if spy is not None:
        sk = _known_session(t0, spy.index)
        if sk is not None:
            sc = spy.loc[:sk, "close"]
            out.update(spy_ret_20d=_ret(sc, 20), spy_vs_sma50=_vs_sma(sc, 50), spy_vs_sma200=_vs_sma(sc, 200))
    e_raw, e_how = _fill_price_raw(ep, entries[0], intraday, tf, raw)
    e_fac = _factor(split, raw, _session_of(t0, raw.index))
    out.update(under_entry=e_raw, under_entry_source=e_how)
    if e_raw is not None and e_fac is not None:
        e_adj = e_raw * e_fac
        out["move_into_entry"] = float(e_adj / c.iloc[-1] - 1)
        if ep["kind"] == "option" and ep.get("standard_deliverable") and ep.get("strike"):
            k = float(ep["strike"])
            out["moneyness_entry"] = (e_raw / k - 1) if ep["put_call"] == "CALL" else (k / e_raw - 1)
        sign = direction_sign(ep)
        if ep["status"] == "closed" and exits:
            t1 = _ts(exits[-1]["time"])
            x_raw, x_how = _fill_price_raw(ep, exits[-1], intraday, tf, raw)
            x_fac = _factor(split, raw, _session_of(t1, raw.index))
            out.update(under_exit=x_raw, under_exit_source=x_how)
            if x_raw is not None and x_fac is not None:
                x_adj = x_raw * x_fac
                out["under_move_dir"] = float((x_adj / e_adj - 1) * sign)
                out.update(_excursions(t0, t1, e_adj, sign, split, intraday, e_fac, x_fac))
                after = split.loc[split.index > pd.Timestamp(t1.astimezone(ET).date()), "close"]
                if len(after) >= 5:
                    out["post_exit_5d_dir"] = float((after.iloc[4] / x_adj - 1) * sign)
    return out


def _excursions(t0: datetime, t1: datetime, e_adj: float, sign: int, split: pd.DataFrame,
                intraday: pd.DataFrame | None, e_fac: float, x_fac: float) -> dict:
    """Best and worst directional move of the underlying while the trade was on."""
    if intraday is not None and abs(e_fac - x_fac) < 1e-9:
        w = intraday[(intraday["time"] >= pd.Timestamp(t0)) & (intraday["time"] <= pd.Timestamp(t1))]
        if not w.empty:
            hi, lo = float(w["high"].max()) * e_fac, float(w["low"].min()) * e_fac
            src = "intraday"
        else:
            return {}
    else:
        d0, d1 = pd.Timestamp(t0.astimezone(ET).date()), pd.Timestamp(t1.astimezone(ET).date())
        w = split.loc[d0:d1]
        if w.empty:
            return {}
        hi, lo, src = float(w["high"].max()), float(w["low"].min()), "daily"
    up, down = hi / e_adj - 1, lo / e_adj - 1
    return {"mfe_dir": up if sign > 0 else -down, "mae_dir": down if sign > 0 else -up, "excursion_source": src}


def earnings_context(ep: dict, dates: list[pd.Timestamp] | None) -> dict:
    if ep["kind"] == "etf" or underlying_class(ep) != "stock":
        return {"earnings_known": False}
    if not dates:
        return {"earnings_known": False}
    o = pd.Timestamp(ep["open_date"])
    c = pd.Timestamp(ep["close_date"]) if ep.get("close_date") else None
    nxt = next((d for d in dates if d >= o), None)
    out = {"earnings_known": True, "days_to_earnings": int((nxt - o).days) if nxt is not None else None,
           "next_earnings_at_entry": nxt.date().isoformat() if nxt is not None else None}
    if c is not None:
        out["held_through_earnings"] = bool(nxt is not None and nxt < c)
    return out


# ---------- buckets ----------
def _bucket(v, edges: list[tuple[float, str]], last: str) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    for edge, label in edges:
        if v < edge:
            return label
    return last


def buckets(ep: dict, ctx: dict) -> dict:
    kind = {"option": (ep.get("put_call") or "option").title(), "equity": "Stock", "etf": "ETF"}[ep["kind"]]
    same_day = ep.get("close_date") is not None and ep["close_date"] == ep["open_date"]
    hd = ep.get("hold_days")
    hour = ctx.get("entry_hour")
    return {
        "b_type": kind,
        "b_year": ep["open_date"][:4],
        "b_dte": _bucket(ep.get("dte_at_entry"), [(2, "0-1d"), (8, "2-7d"), (31, "8-30d"), (91, "31-90d")], ">90d"),
        "b_hold": None if hd is None else "same day" if same_day else _bucket(hd, [(6, "1-5d"), (21, "6-20d")], ">20d"),
        "b_moneyness": _bucket(ctx.get("moneyness_entry"), [(-0.15, "OTM >15%"), (-0.05, "OTM 5-15%"),
                                                            (0.05, "ATM ±5%")], "ITM >5%"),
        "b_trend": None if ctx.get("vs_sma50") is None else ("above 50d avg" if ctx["vs_sma50"] >= 0 else "below 50d avg"),
        "b_extension": _bucket(ctx.get("dist_high_20"), [(-0.10, ">10% below 20d high"),
                                                         (-0.03, "3-10% below 20d high")], "within 3% of 20d high"),
        "b_into_entry": _bucket(ctx.get("move_into_entry"), [(-0.02, "down >2% since prior close"),
                                                             (0.02, "within ±2%")], "up >2% since prior close"),
        "b_entry_time": None if hour is None else ("outside regular hours" if hour < 9.5 or hour >= 16 else
                                                   "9:30-10:30" if hour < 10.5 else "10:30-15:00" if hour < 15
                                                   else "15:00-16:00"),
        "b_spy": None if ctx.get("spy_vs_sma50") is None else ("SPY above 50d avg" if ctx["spy_vs_sma50"] >= 0
                                                               else "SPY below 50d avg"),
        "b_size": _bucket(abs(ep["cost"]) if ep.get("cost") is not None else None,
                          [(100, "<$100"), (300, "$100-300"), (1000, "$300-1,000")], ">$1,000"),
        "b_class": ctx.get("underlying_class"),
        "b_exit": {"expired": "held to expiry", "closed": "closed before expiry" if ep["kind"] == "option" else "sold",
                   "open": "open", "unresolved": "unresolved", "corp_action": "corporate action"}.get(ep["exit_reason"]),
        "b_scaled": "scaled in" if ep["n_entries"] > 1 else "single entry",
        "b_earnings": _earnings_bucket(ctx),
        "b_entry_style": ctx.get("entry_style"),
        "b_exit_trigger": ctx.get("exit_trigger") or ("expired" if ep["exit_reason"] == "expired" else None),
    }


def _earnings_bucket(ctx: dict) -> str | None:
    if not ctx.get("earnings_known"):
        return None
    held = ctx.get("held_through_earnings")
    if held is None:
        return "open"
    if held:
        return "held through earnings"
    d = ctx.get("days_to_earnings")
    return "exited before a report <=14d away" if d is not None and d <= 14 else "no report while held"


# ---------- enrichment ----------
def enrich(episodes: list[dict], base: Path | None = None,
           loader: Callable[[str, str], pd.DataFrame | None] | None = None,
           earnings: pd.DataFrame | None = None) -> None:
    """Attach ctx (features and outcomes) and bucket labels to each episode, in place."""
    load = loader or (lambda sym, adj: load_daily(sym, adj, base))
    dates: dict[str, list[pd.Timestamp]] = {}
    if earnings is not None and not earnings.empty and "date" in earnings.columns:
        ev = earnings.dropna(subset=["date"])
        dates = {t: sorted(pd.to_datetime(g["date"]).dt.normalize().unique()) for t, g in ev.groupby("ticker")}
    cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    def daily(sym: str, adj: str):
        if (sym, adj) not in cache:
            cache[(sym, adj)] = load(sym, adj)
        return cache[(sym, adj)]

    from core.personal_portfolio import trade_timing   # imports this module's helpers, so imported late

    spy = daily("SPY", "split")
    for ep in episodes:
        intraday, tf = load_intraday(ep["id"], base)
        split, raw = daily(ep["chart_symbol"], "split"), daily(ep["chart_symbol"], "raw")
        ctx = episode_features(ep, split, raw, spy, intraday, tf)
        ctx.update(earnings_context(ep, dates.get(ep["chart_symbol"])))
        ctx.update(trade_timing.review(ep, ctx, split, raw, intraday, tf))
        ep["ctx"] = ctx
        ep.update(buckets(ep, ctx))


# ---------- summaries ----------
def resolved(episodes: list[dict]) -> list[dict]:
    """Closed episodes whose entry and exit are both in the record: the only ones with a real P&L."""
    return [e for e in episodes if e["status"] == "closed" and not e["incomplete"] and e.get("pnl") is not None]


def summarize(rows: list[dict]) -> dict:
    pnl = np.array([r["pnl"] for r in rows], dtype=float)
    ret = np.array([r["return_pct"] for r in rows if r.get("return_pct") is not None], dtype=float)
    n = len(pnl)
    if n == 0:
        return {"n": 0}
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    hold = [r["hold_days"] for r in rows if r.get("hold_days") is not None]
    return {"n": n, "wins": int(len(wins)), "win_rate": float(len(wins) / n), "total_pnl": float(pnl.sum()),
            "avg_pnl": float(pnl.mean()), "median_pnl": float(np.median(pnl)),
            "avg_win": float(wins.mean()) if len(wins) else None, "avg_loss": float(losses.mean()) if len(losses) else None,
            "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None,
            "avg_return": float(ret.mean()) if len(ret) else None,
            "median_return": float(np.median(ret)) if len(ret) else None,
            "median_hold_days": float(np.median(hold)) if hold else None,
            "best": float(pnl.max()), "worst": float(pnl.min())}


def bootstrap_mean(values: list[float], n_boot: int = 4000, seed: int = 7) -> tuple[float, float] | None:
    x = np.asarray(values, dtype=float)
    if len(x) < 5:
        return None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def breakdown(rows: list[dict], field: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get(field) or "n/a", []).append(r)
    out = []
    for label, g in groups.items():
        s = summarize(g)
        s["label"] = label
        s["ci_avg_pnl"] = bootstrap_mean([r["pnl"] for r in g])
        s["ci_avg_return"] = bootstrap_mean([r["return_pct"] for r in g if r.get("return_pct") is not None])
        out.append(s)
    return sorted(out, key=lambda s: -s["n"])
