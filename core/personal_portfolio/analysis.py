"""Pure analytics for the personal portfolio page. No I/O and no network.

What is measured, and where it comes from:
  look-through     A leveraged ETF counts as leverage x its underlying. The pair is parsed from the
                   fund description and accepted only if the two daily return series agree
                   (corr >= 0.85, beta within 30% of the stated leverage). Otherwise the ETF stands
                   for itself. Options count as delta x multiplier x quantity x underlying price.
  risk             Historical simulation: the last `beta_window` daily returns applied to today's
                   look-through exposures (delta-linear, so it ignores gamma, decay and IV moves).
  shrugged miss    Research 07/05 (long_horizon_discount_2026-09-25): EPS surprise < -2% and a
                   2-day reaction within +-2% of SPY. Those names lagged by 2.8% over 126 sessions
                   (t = -4.3). It is an exclusion flag, not a sell signal.
  momentum         12-1 momentum rank among the 300 most liquid stocks, from the saved shadow-ledger
                   ranking. The validated use is a monthly top-20 sleeve, not timing single names.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict
from datetime import date, datetime

import numpy as np
import pandas as pd

from core.personal_portfolio.data import ET, PersonalConfig, last_completed_session

EAR_BAND = 2.0  # percent; research 07's band for both the surprise and the reaction
MIN_OBS = 40
_LEV_RE = re.compile(r"\b(\d+(?:\.\d+)?)XL?\b")
_UNDERLYING_RES = (
    re.compile(r"\b\d+(?:\.\d+)?X(?:L\b)?\s+(?:(?:LONG|LG|SHORT|SH|BULL|BEAR|INVERSE)\s+)?(?P<t>[A-Z]{1,5})\b"),
    re.compile(r"\b(?P<t>[A-Z]{1,5})\s+(?:BULL|BEAR)\s+\d+(?:\.\d+)?X\b"),
)
_INVERSE_WORDS = {"SHORT", "SH", "BEAR", "INVERSE"}
_NOT_TICKERS = {"ETF", "DAILY", "DLY", "TARGET", "LONG", "SHORT", "BULL", "BEAR", "NEW", "FUND",
                "SHARES", "TRUST", "INDEX", "THE", "LG", "SH", "INVERSE"}
SEVERITY_ORDER = {"high": 0, "warn": 1, "info": 2}


def _f(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ---------- look-through ----------
def parse_leveraged_etf(description: str | None, symbol: str = "") -> dict:
    """Stated leverage and, when the description names one, the underlying ticker."""
    text = str(description or "").upper()
    m = _LEV_RE.search(text)
    if not m:
        return {"leverage": 1.0, "underlying": None}
    sign = -1.0 if set(re.findall(r"[A-Z]+", text)) & _INVERSE_WORDS else 1.0
    underlying = None
    for rx in _UNDERLYING_RES:
        hit = rx.search(text)
        if hit and hit["t"] not in _NOT_TICKERS and hit["t"] != symbol.upper():
            underlying = hit["t"]
            break
    return {"leverage": sign * float(m[1]), "underlying": underlying}


def validate_look_through(etf_ret: pd.Series | None, und_ret: pd.Series | None, leverage: float,
                          window: int = 252) -> dict:
    """The derivative-vs-underlying check: does the ETF actually move as leverage x the underlying?"""
    if etf_ret is None or und_ret is None:
        return {"verified": False, "reason": "no price history", "n": 0}
    df = pd.concat([etf_ret, und_ret], axis=1).dropna().tail(window)
    n = len(df)
    if n < MIN_OBS:
        return {"verified": False, "reason": f"only {n} shared sessions", "n": n}
    corr = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    b = float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())
    ok = corr >= 0.85 and np.sign(b) == np.sign(leverage) and 0.7 <= b / leverage <= 1.3
    return {"verified": bool(ok), "corr": corr, "beta": b, "n": n,
            "reason": None if ok else f"corr {corr:.2f}, beta {b:.2f} vs stated {leverage:g}x"}


def look_through_candidates(descriptions: dict[str, str | None], overrides: dict[str, dict]) -> dict[str, dict]:
    """ETF symbol -> {underlying, leverage, source} before validation."""
    out = {}
    for sym, desc in descriptions.items():
        if sym in overrides:
            out[sym] = {**overrides[sym], "source": "override"}
            continue
        p = parse_leveraged_etf(desc, sym)
        if p["underlying"] or p["leverage"] != 1.0:
            out[sym] = {**p, "source": "description"}
    return out


def resolve_look_through(candidates: dict[str, dict], returns: pd.DataFrame, window: int) -> dict[str, dict]:
    out = {}
    for sym, c in candidates.items():
        und = c.get("underlying")
        if not und:
            out[sym] = {**c, "verified": False, "reason": "no underlying ticker in the description"}
            continue
        check = validate_look_through(returns.get(sym), returns.get(und), c["leverage"], window)
        out[sym] = {**c, **check}
    return out


def instrument_of(symbol: str, look_through: dict[str, dict]) -> tuple[str, float]:
    lt = look_through.get(symbol)
    if lt and lt.get("verified"):
        return lt["underlying"], float(lt["leverage"])
    return symbol, 1.0


# ---------- return statistics ----------
def daily_returns(closes: pd.DataFrame) -> pd.DataFrame:
    return closes.pct_change(fill_method=None) if not closes.empty else closes


def beta(y: pd.Series | None, x: pd.Series | None, window: int) -> float | None:
    if y is None or x is None:
        return None
    df = pd.concat([y, x], axis=1).dropna().tail(window)
    if len(df) < MIN_OBS or df.iloc[:, 1].var() == 0:
        return None
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())


def realized_vol(r: pd.Series | None, window: int) -> float | None:
    if r is None:
        return None
    r = r.dropna().tail(window)
    return float(r.std() * math.sqrt(252)) if len(r) >= 20 else None


def vol_drag(leverage: float, sigma: float | None) -> float | None:
    """Yearly log-return shortfall of a daily-reset L x fund versus L x the underlying's log return.

    Continuous time: ln F_T = L ln S_T - (L^2 - L) sigma^2 T / 2 (no fees or financing). So if the stock
    ends a year flat, the fund ends at exp(-drag) of its start; see flat_year_loss.
    """
    if sigma is None:
        return None
    return (leverage * leverage - leverage) * sigma * sigma / 2


def flat_year_loss(drag: float | None) -> float | None:
    return None if drag is None else 1 - math.exp(-drag)


def price_context(close: pd.Series | None) -> dict:
    if close is None:
        return {}
    c = close.dropna()
    if c.empty:
        return {}
    out = {"close": float(c.iloc[-1]), "close_date": c.index[-1].date().isoformat()}
    if len(c) >= 200:
        sma = float(c.tail(200).mean())
        out.update(sma200=sma, vs_sma200=float(c.iloc[-1] / sma - 1))
    if len(c) >= 22:
        out["ret_1m"] = float(c.iloc[-1] / c.iloc[-22] - 1)
    return out


# ---------- earnings ----------
def earnings_status(events: pd.DataFrame, close: pd.Series | None, spy: pd.Series | None,
                    bars_through: date, today: date, cfg: PersonalConfig) -> dict:
    """Next report date, and whether the latest known report was a shrugged-off miss (research 07)."""
    ev = events.dropna(subset=["date"]) if "date" in events.columns else events.iloc[0:0]
    out: dict = {"next_earnings": None, "days_to_next": None, "last_report": None,
                 "shrugged_miss": None, "pending_report": None}
    if ev.empty:
        return out
    upcoming = ev[(ev["date"].dt.date >= today) & ev["reported_eps"].isna()]
    if not upcoming.empty:
        nxt = upcoming["date"].min().date()
        out.update(next_earnings=nxt.isoformat(), days_to_next=(nxt - today).days)
    through = pd.Timestamp(bars_through)
    reported = ev[ev["reported_eps"].notna() & ev["surprise_pct"].notna() & (ev["date"] <= through)].sort_values("date")
    pending = ev[ev["reported_eps"].isna() & (ev["date"] <= through)]
    if not pending.empty and (reported.empty or pending["date"].max() > reported["date"].max()):
        out["pending_report"] = pending["date"].max().date().isoformat()
    if reported.empty:
        return out
    last = reported.iloc[-1]
    rep = {"date": last["date"].date().isoformat(), "surprise_pct": float(last["surprise_pct"])}
    s_grp = "beat" if rep["surprise_pct"] > EAR_BAND else "miss" if rep["surprise_pct"] < -EAR_BAND else "inline"
    rep["surprise_group"] = s_grp
    out["last_report"] = rep
    if close is None or spy is None:
        return out
    px = pd.concat([close, spy], axis=1).dropna()
    px = px[px.index <= through]
    cal = px.index
    i0 = int(cal.searchsorted(last["date"], side="left"))   # report session, or the next one
    if i0 < 1 or i0 + 1 >= len(cal):
        return out
    c, s = px.iloc[:, 0].to_numpy(), px.iloc[:, 1].to_numpy()
    ear = (c[i0 + 1] / c[i0 - 1] - s[i0 + 1] / s[i0 - 1]) * 100
    r_grp = "up" if ear > EAR_BAND else "down" if ear < -EAR_BAND else "flat"
    known = cal[i0 + 1].date()
    rep.update(reaction_pct=float(ear), reaction_group=r_grp, known_date=known.isoformat())
    if out["pending_report"] is None:
        out["shrugged_miss"] = bool(s_grp == "miss" and r_grp == "flat"
                                    and (bars_through - known).days <= cfg.shrug_window_days)
    return out


# ---------- momentum ----------
def momentum_info(ticker: str, xs: pd.DataFrame | None, sleeve: set[str]) -> dict | None:
    if xs is None:
        return None
    row = xs[xs["ticker"] == ticker]
    if row.empty:
        return {"ranked": False}
    r = row.iloc[0]
    m = _f(r["mom_12_1"])
    out = {"ranked": True, "mom_12_1": m, "dv_rank": int(r["dv_rank"]), "in_liquid_300": bool(r["dv_rank"] <= 300),
           "in_sleeve": ticker in sleeve, "rank_in_300": None}
    if out["in_liquid_300"] and m is not None:
        liquid = xs.loc[xs["dv_rank"] <= 300, "mom_12_1"].dropna()
        out["rank_in_300"] = int((liquid > m).sum()) + 1
    return out


# ---------- options ----------
def option_review(pos: dict, next_earnings: str | None, available_funds: float | None,
                  cfg: PersonalConfig, today: date, history_stats: dict | None = None) -> tuple[dict, list[dict]]:
    q = pos.get("quote") or {}
    sym, und = pos["symbol"], pos.get("underlying")
    strike, qty, mult = _f(pos.get("strike")), pos["quantity"], _f(pos.get("multiplier")) or 100.0
    s = _f(q.get("underlying_price"))
    call = pos.get("put_call") == "CALL"
    expiry = date.fromisoformat(pos["expiry"]) if pos.get("expiry") else None
    dte = int(q["dte"]) if q.get("dte") is not None else ((expiry - today).days if expiry else None)
    avg = _f(pos.get("average_price"))
    out = {"symbol": sym, "underlying": und, "put_call": pos.get("put_call"), "strike": strike,
           "expiry": pos.get("expiry"), "dte": dte, "quantity": qty, "underlying_price": s,
           "mark": _f(q.get("mark")), "delta": _f(q.get("delta")), "iv": _f(q.get("iv")),
           "theta_per_day": _f(q.get("theta")) * mult * qty if q.get("theta") is not None else None,
           "spread_pct": None, "moneyness": None, "breakeven": None, "move_to_breakeven": None,
           "exercise_notional": strike * mult * abs(qty) if strike is not None else None,
           "earnings_before_expiry": bool(next_earnings and expiry and today.isoformat() <= next_earnings <= expiry.isoformat()),
           "next_earnings": next_earnings}
    bid, ask, mark = _f(q.get("bid")), _f(q.get("ask")), _f(q.get("mark"))
    if bid is not None and ask is not None and mark:
        out["spread_pct"] = (ask - bid) / mark
    if s and strike:
        out["moneyness"] = (s / strike - 1) if call else (strike / s - 1)
        if avg is not None:
            be = strike + avg if call else strike - avg
            out.update(breakeven=be, move_to_breakeven=be / s - 1)
    flags: list[dict] = []

    def flag(sev: str, code: str, text: str) -> None:
        flags.append({"severity": sev, "code": code, "ticker": und or sym, "symbol": sym, "text": text})

    itm = out["moneyness"] is not None and out["moneyness"] > 0
    if dte is not None and dte <= cfg.option_expiry_warn_days:
        if itm and qty > 0:
            shares, notional = int(abs(qty) * mult), out["exercise_notional"]
            short_funds = available_funds is not None and notional is not None and notional > available_funds
            flag("high" if short_funds or dte <= 1 else "warn", "itm_into_expiry",
                 f"{sym}: in the money with {dte}d left. If held through expiry it is auto-exercised: "
                 f"{'buys' if call else 'sells'} {shares} {und} shares (${notional:,.0f})"
                 + (f"; available funds ${available_funds:,.0f}" if available_funds is not None else "") + ".")
        elif itm and qty < 0:
            flag("high", "itm_short_assignment", f"{sym}: short and in the money with {dte}d left: assignment risk.")
        elif out["move_to_breakeven"] is not None:
            flag("high" if dte <= 1 else "warn", "otm_near_expiry",
                 f"{sym}: out of the money with {dte}d left; {und} must move {out['move_to_breakeven']:+.1%} "
                 f"by expiry to break even.")
    if (qty > 0 and avg and mark is not None and dte is not None and dte <= cfg.late_loss_dte
            and mark <= cfg.late_loss_frac * avg):
        text = f"{sym}: worth {mark / avg:.0%} of what you paid with {dte}d left."
        hs = history_stats
        if hs and hs.get("stop_better_share") is not None:
            text += (f" In your own trade history, options at half value or less with <= {hs['max_dte']} days left got "
                     f"back to breakeven by expiry {hs['breakeven_share']:.0%} of the time and expired worthless "
                     f"{hs['worthless_share']:.0%} ({hs['n']} cases); selling at that point beat holding on "
                     f"{hs['stop_better_share']:.0%} of the {hs['n_stopped']} trades you still held.")
        flag("warn", "losing_option_late", text)
    if out["earnings_before_expiry"]:
        flag("warn", "earnings_before_expiry",
             f"{sym}: {und} reports {next_earnings}, before expiry {pos.get('expiry')} (event and IV-crush risk).")
    if out["spread_pct"] is not None and out["spread_pct"] > cfg.wide_spread_pct:
        flag("info", "wide_spread", f"{sym}: bid/ask spread is {out['spread_pct']:.0%} of the mark; "
                                    f"exiting at the bid costs about {out['spread_pct'] / 2:.0%}.")
    return out, flags


# ---------- exposures and risk ----------
def position_exposure(pos: dict, look_through: dict[str, dict]) -> tuple[str | None, float | None, float, str]:
    """(instrument, underlying-dollar exposure, leverage, basis) for one position."""
    kind = pos["kind"]
    if kind == "option":
        und = pos.get("underlying")
        if not und:
            return None, None, 1.0, "option without underlying"
        inst, lev = instrument_of(und, look_through)
        q = pos.get("quote") or {}
        delta, s = _f(q.get("delta")), _f(q.get("underlying_price"))
        if delta is None or s is None:
            return inst, None, lev, "no option greeks"
        mult = _f(pos.get("multiplier")) or 100.0
        return inst, delta * mult * pos["quantity"] * s * lev, lev, "delta"
    if kind in ("etf", "equity"):
        inst, lev = instrument_of(pos["symbol"], look_through)
        mv = _f(pos.get("market_value"))
        return inst, (mv * lev if mv is not None else None), lev, "market value" if lev == 1.0 else f"{lev:g}x look-through"
    return None, 0.0, 1.0, "cash-like"


def correlation_clusters(corr: pd.DataFrame, threshold: float) -> list[list[str]]:
    names = list(corr.columns)
    parent = {n: n for n in names}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and v >= threshold:
                parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g))


def portfolio_risk(exposure: dict[str, float], returns: pd.DataFrame, net_liq: float | None,
                   cfg: PersonalConfig, legs: list[dict] | None = None) -> dict:
    """Correlation structure of the instruments, and a historical-simulation replay of the legs.

    Each leg is {instrument, exposure, max_loss}. A long option's simulated daily loss is floored at
    its premium (max_loss); its gains stay delta-linear, so upside is understated.
    """
    insts = [i for i, e in exposure.items() if e and i in returns.columns]
    missing = sorted(i for i, e in exposure.items() if e and i not in returns.columns)
    out: dict = {"missing_history": missing, "hist_sim": None, "corr": None, "clusters": [], "n_eff": None}
    if not insts:
        return out
    window = returns[insts].tail(cfg.beta_window)
    counts = window.notna().sum()
    usable = [i for i in insts if counts[i] >= 120]
    out["short_history"] = sorted(set(insts) - set(usable))
    corr = window[insts].corr(min_periods=MIN_OBS)
    out["corr"] = {a: {b: _f(corr.loc[a, b]) for b in insts} for a in insts}
    out["clusters"] = correlation_clusters(corr, cfg.cluster_corr)
    w = pd.Series({i: exposure[i] for i in insts})
    # copy=True: the live server runs with pandas copy-on-write on (pandas_ta turns it on
    # process-wide), where to_numpy() is read-only and fill_diagonal would raise.
    c = corr.fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(c, 1.0)
    var = float(w.to_numpy() @ c @ w.to_numpy())
    if var > 0:
        out["n_eff"] = float(w.abs().sum() ** 2 / var)
    if legs is None:
        legs = [{"instrument": i, "exposure": e, "max_loss": None} for i, e in exposure.items()]
    if usable:
        r = window[usable].dropna(how="any")
        if len(r) >= 60:
            pnl = np.zeros(len(r))
            for leg in legs:
                if leg["instrument"] not in usable or not leg["exposure"]:
                    continue
                x = r[leg["instrument"]].to_numpy() * leg["exposure"]
                pnl += x if leg.get("max_loss") is None else np.maximum(x, -leg["max_loss"])
            s = pd.Series(pnl, index=r.index)
            out["hist_sim"] = {
                "n_days": len(s), "start": s.index[0].date().isoformat(), "end": s.index[-1].date().isoformat(),
                "q05": float(s.quantile(0.05)), "worst": float(s.min()), "worst_date": s.idxmin().date().isoformat(),
                "best": float(s.max()), "daily_sd": float(s.std()),
                "q05_pct": float(s.quantile(0.05) / net_liq) if net_liq else None,
                "worst_pct": float(s.min() / net_liq) if net_liq else None,
                "excluded": out["short_history"],
            }
    return out


# ---------- report ----------
def required_tickers(holdings: dict, watchlist: list[dict], overrides: dict[str, dict]) -> dict[str, list[str]]:
    """Which tickers need daily bars, and which may have earnings (everything not known to be an ETF/option)."""
    quotes = holdings.get("quotes") or {}
    positions = [p for a in holdings.get("accounts") or [] for p in a["positions"]]
    held = {p["symbol"] for p in positions if p["kind"] in ("etf", "equity")}
    unders = {p["underlying"] for p in positions if p["kind"] == "option" and p.get("underlying")}
    watch = {w["ticker"] for w in watchlist}
    descs = {s: (quotes.get(s) or {}).get("description") for s in held | unders | watch}
    descs.update({p["symbol"]: p.get("description") or descs.get(p["symbol"]) for p in positions if p["kind"] == "etf"})
    cand = look_through_candidates(descs, overrides)
    looked = {c["underlying"] for c in cand.values() if c.get("underlying")}
    bars = held | unders | watch | looked | {"SPY", "QQQ", "IWM"}
    etfs = {p["symbol"] for p in positions if p["kind"] == "etf"} | set(cand) | {"SPY", "QQQ", "IWM"}
    etfs |= {s for s, q in quotes.items() if (q or {}).get("asset_sub_type") == "ETF"}
    return {"bars": sorted(bars), "earnings": sorted(bars - etfs), "candidates": cand}


def build_report(holdings: dict, closes: pd.DataFrame, earnings: pd.DataFrame, momentum: dict,
                 themes: dict[str, str], watchlist: list[dict], overrides: dict[str, dict],
                 cfg: PersonalConfig, now: datetime, history_stats: dict | None = None) -> dict:
    today = now.astimezone(ET).date()
    want = last_completed_session(now)
    rets = daily_returns(closes)
    bars_through = closes.index.max().date() if not closes.empty else None
    req = required_tickers(holdings, watchlist, overrides)
    look_through = resolve_look_through(req["candidates"], rets, cfg.beta_window)
    accounts = holdings.get("accounts") or []
    net_liq = sum(_f(a["balances"].get("liquidation_value")) or 0.0 for a in accounts) or None
    xs, sleeve = momentum.get("cross_section"), set(momentum.get("sleeve") or [])
    spy = closes.get("SPY")
    flags: list[dict] = []

    def ev_for(t: str) -> pd.DataFrame:
        return earnings[earnings["ticker"] == t] if not earnings.empty else earnings

    def stats_for(t: str) -> dict:
        r = rets.get(t)
        st = {"beta_spy": beta(r, rets.get("SPY"), cfg.beta_window),
              "beta_qqq": beta(r, rets.get("QQQ"), cfg.beta_window),
              "vol": realized_vol(r, cfg.vol_window), "theme": themes.get(t),
              **price_context(closes.get(t)), "momentum": momentum_info(t, xs, sleeve)}
        st["earnings"] = (earnings_status(ev_for(t), closes.get(t), spy, bars_through, today, cfg)
                          if bars_through and t in req["earnings"] else None)
        return st

    positions, options, legs = [], [], []
    exposure: dict[str, float] = {}
    capital: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    unknown_exposure = []
    for acct in accounts:
        avail = _f(acct["balances"].get("available_funds"))
        if avail is None:
            avail = _f(acct["balances"].get("cash_balance"))
        for pos in acct["positions"]:
            inst, exp, lev, basis = position_exposure(pos, look_through)
            row = {**{k: v for k, v in pos.items() if k != "quote"}, "account": acct["account"],
                   "mark": _f((pos.get("quote") or {}).get("mark")), "instrument": inst, "exposure": exp,
                   "leverage": lev, "exposure_basis": basis,
                   "weight": (_f(pos.get("market_value")) or 0.0) / net_liq if net_liq else None,
                   "look_through": look_through.get(pos["symbol"])}
            if inst and exp is not None:
                mv = _f(pos.get("market_value")) or 0.0
                exposure[inst] = exposure.get(inst, 0.0) + exp
                capital[inst] = capital.get(inst, 0.0) + mv
                sources.setdefault(inst, []).append(pos["symbol"])
                long_option = pos["kind"] == "option" and pos["quantity"] > 0
                legs.append({"instrument": inst, "exposure": exp, "max_loss": mv if long_option else None})
            elif inst and exp is None:
                unknown_exposure.append(pos["symbol"])
            if pos["kind"] == "option":
                ne = None
                if inst:
                    st_e = earnings_status(ev_for(inst), None, None, bars_through or today, today, cfg) \
                        if inst in req["earnings"] else {}
                    ne = (st_e or {}).get("next_earnings")
                review, oflags = option_review(pos, ne, avail, cfg, today, history_stats)
                options.append({**review, "account": acct["account"], "open_pl": pos.get("open_pl"),
                                "market_value": pos.get("market_value")})
                flags += oflags
            positions.append(row)

    # leveraged-ETF decay and unverified mappings
    for sym, lt in look_through.items():
        if not any(p["symbol"] == sym for p in positions):
            continue
        if lt.get("verified"):
            sigma = realized_vol(rets.get(lt["underlying"]), cfg.vol_window)
            drag = vol_drag(lt["leverage"], sigma)
            loss = flat_year_loss(drag)
            lt.update(drag=drag, flat_year_loss=loss, underlying_vol=sigma)
            if drag is not None:
                flags.append({"severity": "warn" if loss >= 0.15 else "info", "code": "leveraged_decay",
                              "ticker": lt["underlying"], "symbol": sym,
                              "text": f"{sym} ({lt['leverage']:g}x {lt['underlying']}): at {lt['underlying']}'s "
                                      f"{cfg.vol_window}-day volatility of {sigma:.0%}, daily resetting alone would take "
                                      f"about {loss:.0%} off the fund over a year in which {lt['underlying']} ends flat "
                                      f"(before fees). The erosion is worst when the stock swings without trending."})
        elif lt.get("leverage", 1.0) != 1.0:
            flags.append({"severity": "info", "code": "look_through_unverified", "ticker": sym, "symbol": sym,
                          "text": f"{sym}: stated {lt['leverage']:g}x but not mapped to an underlying "
                                  f"({lt.get('reason')}); counted as itself."})

    instruments = []
    for inst, exp in sorted(exposure.items(), key=lambda kv: -abs(kv[1])):
        st = stats_for(inst)
        pct = exp / net_liq if net_liq else None
        cap = capital.get(inst, 0.0)
        instruments.append({"instrument": inst, "exposure": exp, "pct": pct, "capital": cap,
                            "capital_pct": cap / net_liq if net_liq else None,
                            "sources": sources.get(inst, []), **st})
        if pct is not None and abs(pct) > cfg.max_underlying_pct:
            flags.append({"severity": "high" if abs(pct) > 2 * cfg.max_underlying_pct else "warn",
                          "code": "name_concentration", "ticker": inst, "symbol": inst,
                          "text": f"{inst}: price exposure is {pct:.0%} of net liquidation (limit "
                                  f"{cfg.max_underlying_pct:.0%}; delta- and leverage-adjusted) via "
                                  f"{', '.join(sources.get(inst, []))}. A 1% move in {inst} is about "
                                  f"${abs(exp) / 100:,.0f}. Capital in these positions: ${cap:,.0f}"
                                  + (f" ({cap / net_liq:.0%})." if net_liq else ".")})
        e = st.get("earnings") or {}
        if e.get("shrugged_miss"):
            lr = e["last_report"]
            flags.append({"severity": "warn", "code": "shrugged_miss", "ticker": inst, "symbol": inst,
                          "text": f"{inst}: missed EPS by {lr['surprise_pct']:.1f}% on {lr['date']} and moved only "
                                  f"{lr['reaction_pct']:+.1f}% vs SPY. In research 07 such names lagged SPY by 2.8% "
                                  f"over the next 126 sessions (t = -4.3)."})
        if e.get("days_to_next") is not None and e["days_to_next"] <= cfg.earnings_soon_days:
            flags.append({"severity": "info", "code": "earnings_soon", "ticker": inst, "symbol": inst,
                          "text": f"{inst} reports {e['next_earnings']} ({e['days_to_next']}d)."})

    risk = portfolio_risk(exposure, rets, net_liq, cfg, legs)
    gross = sum(abs(e) for e in exposure.values()) / net_liq if net_liq else None
    beta_dollars = sum(exposure[i["instrument"]] * i["beta_spy"] for i in instruments if i["beta_spy"] is not None)
    no_beta = [i["instrument"] for i in instruments if i["beta_spy"] is None]
    if gross is not None and gross > cfg.max_gross_exposure:
        flags.append({"severity": "warn", "code": "gross_exposure", "ticker": "", "symbol": "",
                      "text": f"Gross price exposure (delta- and leverage-adjusted) is {gross:.2f}x net liquidation "
                              f"(limit {cfg.max_gross_exposure:.2f}x)."})
    for cl in risk["clusters"]:
        if len(cl) < 2 or not net_liq:
            continue
        pct = sum(exposure[i] for i in cl) / net_liq
        if abs(pct) > cfg.max_cluster_pct:
            flags.append({"severity": "warn", "code": "cluster_concentration", "ticker": cl[0], "symbol": "",
                          "text": f"{', '.join(cl)} move together (daily correlation >= {cfg.cluster_corr:.2f}) and are "
                                  f"{pct:.0%} of net liquidation together (limit {cfg.max_cluster_pct:.0%})."})
    if unknown_exposure:
        flags.append({"severity": "info", "code": "exposure_unknown", "ticker": "", "symbol": "",
                      "text": f"No delta/price for {', '.join(unknown_exposure)}; left out of exposure and risk."})

    watch_rows = []
    quotes = holdings.get("quotes") or {}
    for w in watchlist:
        t = w["ticker"]
        q = quotes.get(t) or {}
        lt = look_through.get(t)
        watch_rows.append({**w, "mark": _f(q.get("mark")), "net_pct": _f(q.get("net_pct")),
                           "description": q.get("description"), "look_through": lt,
                           "held": t in sources or any(p["symbol"] == t for p in positions), **stats_for(t)})

    flags.sort(key=lambda f: (SEVERITY_ORDER[f["severity"]], f["code"], f["ticker"]))
    totals = {"net_liq": net_liq,
              "market_value": sum(_f(p.get("market_value")) or 0.0 for p in positions),
              "day_pl": sum(_f(p.get("day_pl")) or 0.0 for p in positions),
              "open_pl": sum(_f(p.get("open_pl")) or 0.0 for p in positions),
              "gross_exposure": gross, "net_exposure": sum(exposure.values()) / net_liq if net_liq else None,
              "beta_spy_dollars": beta_dollars, "beta_spy_pct": beta_dollars / net_liq if net_liq else None,
              "no_beta": no_beta}
    return {
        "generated_at": now.isoformat(), "fetched_at": holdings.get("fetched_at"),
        "accounts": [{k: v for k, v in a.items() if k != "positions"} | {"n_positions": len(a["positions"])}
                     for a in accounts],
        "totals": totals, "positions": positions, "options": options, "instruments": instruments,
        "look_through": look_through, "risk": risk, "flags": flags, "watchlist": watch_rows,
        "momentum": {"decision_date": momentum.get("decision_date"), "sleeve": sorted(sleeve),
                     "record": momentum.get("record") or {}},
        "data": {"bars_through": bars_through.isoformat() if bars_through else None,
                 "bars_expected": want.isoformat(), "bars_current": bool(bars_through and bars_through >= want),
                 "missing_bars": sorted(set(req["bars"]) - set(closes.columns))},
        "config": asdict(cfg),
    }


def flag_log_records(report: dict) -> list[dict]:
    """One compact row per held instrument and watchlist name, to test the flags against outcomes later."""
    codes: dict[str, list[str]] = {}
    for f in report["flags"]:
        if f.get("ticker"):
            codes.setdefault(f["ticker"], []).append(f["code"])
    rows = []
    for role, items, key in (("held", report["instruments"], "instrument"), ("watch", report["watchlist"], "ticker")):
        for it in items:
            mom, e = it.get("momentum") or {}, it.get("earnings") or {}
            rows.append({"ticker": it[key], "role": role, "pct": it.get("pct"), "close": it.get("close"),
                         "close_date": it.get("close_date"), "flags": sorted(set(codes.get(it[key], []))),
                         "mom_12_1": mom.get("mom_12_1"), "rank_in_300": mom.get("rank_in_300"),
                         "in_sleeve": mom.get("in_sleeve"), "momentum_decision": report["momentum"]["decision_date"],
                         "shrugged_miss": e.get("shrugged_miss"), "beta_spy": it.get("beta_spy"),
                         "vol": it.get("vol"), "vs_sma200": it.get("vs_sma200")})
    return rows


def sanitize(obj):
    """JSON-safe copy: NaN/inf -> None, numpy/pandas scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj
