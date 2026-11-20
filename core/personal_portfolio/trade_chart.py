"""Chart payloads for the trade-history section (TradingView Lightweight Charts on the page).

Bars are RAW (unadjusted) so fill markers sit at the price actually paid. Option fills are
drawn at the underlying's price when the fill happened (the open of the intraday bar holding
the fill, see trade_analysis); option premiums are in the fill table and marker text, never
as a price series: option trade prints are too sparse to chart honestly.

Lightweight Charts renders timestamps as UTC, so intraday times are sent as ET wall-clock
seconds; daily times are 'YYYY-MM-DD' session dates.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from core.personal_portfolio.data import ET
from core.personal_portfolio.trade_analysis import direction_sign
from core.personal_portfolio.trade_bars import ensure_intraday, load_daily, load_intraday, price_at

DAILY_BEFORE, DAILY_AFTER = 250, 60          # sessions of daily context sent
VISIBLE_BEFORE, VISIBLE_AFTER = 60, 15       # sessions shown before the user scrolls
COLORS = {"entry": "#26a69a", "exit": "#ef5350", "expired": "#9aa4ab", "corp_action": "#c9a227"}


def _et_seconds(ts) -> int:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return int(t.tz_convert(ET).tz_localize(None).timestamp())


def _contract_label(ep: dict) -> str:
    if ep["kind"] == "option":
        return f"{ep['underlying']} {ep['expiry']} {ep['strike']:g}{ep['put_call'][0]}"
    return ep["symbol"]


def _marker_text(f: dict, ep: dict, compact: bool = False) -> str:
    """'B 2 @4.00' on a single trade's chart; 'B 60C' when many trades share one chart (labels collide)."""
    if compact:
        what = f"{ep['strike']:g}{ep['put_call'][0]}" if ep["kind"] == "option" else ""
        act = "exp" if f["source"] == "expiration" else "ca" if f["role"] == "corp_action" else \
            ("B" if f["qty"] > 0 else "S")
        return f"{act} {what}".strip()
    if f["source"] == "expiration":
        return "expired"
    if f["role"] == "corp_action":
        return "corp action"
    px = f"{f['price']:.2f}" if f.get("price") is not None else "?"
    return f"{'B' if f['qty'] > 0 else 'S'} {abs(f['qty']):g} @{px}"


def _marker(time_key, f: dict, ep: dict, compact: bool = False) -> dict:
    role = "expired" if f["source"] == "expiration" else f["role"]
    buy = f["qty"] > 0
    return {"time": time_key, "position": "belowBar" if buy else "aboveBar",
            "shape": "circle" if role in ("expired", "corp_action") else ("arrowUp" if buy else "arrowDown"),
            "color": COLORS[role], "text": _marker_text(f, ep, compact)}


def _snap_daily(idx: pd.DatetimeIndex, t: datetime) -> pd.Timestamp | None:
    d = pd.Timestamp(t.astimezone(ET).date())
    i = int(idx.searchsorted(d, side="right")) - 1
    return idx[i] if i >= 0 else None


def _snap_intraday(times: pd.Series, t: datetime) -> pd.Timestamp | None:
    i = int(times.searchsorted(pd.Timestamp(t), side="right")) - 1
    return times.iloc[i] if i >= 0 else None


def _points(rows: list[tuple]) -> list[dict]:
    """Qty-weighted average price per (time) so each series has one strictly ascending point per bar."""
    agg: dict = {}
    for tkey, sort_key, px, w in rows:
        a = agg.setdefault(tkey, [sort_key, 0.0, 0.0])
        a[1] += px * w
        a[2] += w
    return [{"time": k, "value": v[1] / v[2]} for k, v in sorted(agg.items(), key=lambda kv: kv[1][0]) if v[2]]


def _price_lines(ep: dict) -> list[dict]:
    out = []
    if ep["kind"] == "option" and ep.get("standard_deliverable") and ep.get("strike"):
        out.append({"price": float(ep["strike"]), "title": "strike", "color": "#8e99a3", "style": 2})
        if ep.get("avg_entry") is not None and ep["direction"] == "long":
            be = ep["strike"] + ep["avg_entry"] if ep["put_call"] == "CALL" else ep["strike"] - ep["avg_entry"]
            out.append({"price": float(be), "title": "breakeven at expiry", "color": "#c9a227", "style": 1})
    elif ep["kind"] != "option" and ep.get("avg_entry") is not None:
        out.append({"price": float(ep["avg_entry"]), "title": "avg entry", "color": "#26a69a", "style": 1})
    return out


def _fill_rows(ep: dict, intraday: pd.DataFrame | None, tf: str | None, raw: pd.DataFrame | None) -> list[dict]:
    rows = []
    for f in ep["fills"]:
        t = datetime.fromisoformat(f["time"])
        if ep["kind"] != "option":
            under, how = f.get("price"), "fill price"
        elif f["source"] == "expiration" and raw is not None:
            day = _snap_daily(raw.index, t)
            under, how = (float(raw.at[day, "close"]), "close on expiry day") if day is not None else (None, "no bars")
        else:
            under, how = price_at(intraday, t, tf)
        rows.append({**f, "time_et": t.astimezone(ET).strftime("%Y-%m-%d %H:%M"), "under_price": under,
                     "under_source": how})
    return rows


def _daily_bars(df: pd.DataFrame) -> list[dict]:
    return [{"time": d.strftime("%Y-%m-%d"), "open": float(r.open), "high": float(r.high), "low": float(r.low),
             "close": float(r.close), "volume": float(r.volume)} for d, r in df.iterrows()]


def episode_chart(ep: dict, timeframe: str, now: datetime, base: Path | None = None, fetch=None) -> dict:
    """Bars, markers, fill points and reference lines for one episode. timeframe: '1d' or 'intraday'."""
    raw = load_daily(ep["chart_symbol"], "raw", base)
    head = {"episode": ep["id"], "symbol": ep["chart_symbol"], "label": _contract_label(ep),
            "price_lines": _price_lines(ep), "notes": list(ep.get("notes") or [])}
    if ep.get("chart_symbol_mapped"):
        head["notes"].append(f"bars from {ep['chart_symbol']} (symbol_map.json maps {ep['underlying']})")
    intraday, itf = load_intraday(ep["id"], base)
    if timeframe == "intraday":
        if intraday is None:
            status = ensure_intraday(ep, now, base, fetch) if fetch else ensure_intraday(ep, now, base)
            intraday, itf = load_intraday(ep["id"], base)
            if intraday is None:
                return {**head, "timeframe": "intraday", "bars": [], "markers": [], "points": {},
                        "error": f"no intraday bars ({status})"}
        times = intraday["time"]
        bars = [{"time": _et_seconds(r.time), "open": float(r.open), "high": float(r.high), "low": float(r.low),
                 "close": float(r.close), "volume": float(r.volume)} for r in intraday.itertuples()]
        snap = lambda t: _snap_intraday(times, t)  # noqa: E731
        key = _et_seconds
        visible = None
        tf = itf
    else:
        if raw is None:
            return {**head, "timeframe": "1d", "bars": [], "markers": [], "points": {},
                    "error": f"no daily bars for {ep['chart_symbol']}"}
        idx = raw.index
        i0 = int(idx.searchsorted(pd.Timestamp(ep["open_date"])))
        i1 = int(idx.searchsorted(pd.Timestamp(ep.get("close_date") or now.astimezone(ET).date().isoformat()),
                                  side="right")) - 1
        win = raw.iloc[max(0, i0 - DAILY_BEFORE): max(i1, i0) + DAILY_AFTER + 1]
        bars = _daily_bars(win)
        snap = lambda t: _snap_daily(win.index, t)  # noqa: E731
        key = lambda d: d.strftime("%Y-%m-%d")  # noqa: E731
        lo = max(0, i0 - VISIBLE_BEFORE) - max(0, i0 - DAILY_BEFORE)
        hi = min(len(win) - 1, max(i1, i0) + VISIBLE_AFTER - max(0, i0 - DAILY_BEFORE))
        visible = {"from": bars[lo]["time"], "to": bars[hi]["time"]} if bars else None
        tf = "1d"
    fills = _fill_rows(ep, intraday, itf, raw)
    markers, pts = [], {"entry": [], "exit": []}
    dropped = 0
    for f in fills:
        t = datetime.fromisoformat(f["time"])
        s = snap(t)
        if s is None:
            dropped += 1
            continue
        markers.append((key(s), t, _marker(key(s), f, ep)))
        if f["role"] in pts and f.get("under_price") is not None and f["source"] != "expiration":
            pts[f["role"]].append((key(s), t, f["under_price"], abs(f["qty"])))
    markers.sort(key=lambda m: m[1])
    if dropped:
        head["notes"].append(f"{dropped} fill(s) fall before the first bar and are not drawn")
    return {**head, "timeframe": tf, "bars": bars, "markers": [m[2] for m in markers],
            "points": {k: _points(v) for k, v in pts.items()}, "visible": visible,
            "fills": [{k: f.get(k) for k in ("time_et", "role", "qty", "price", "cash", "fees", "source", "under_price",
                                             "under_source", "symbol")} for f in fills],
            "direction_sign": direction_sign(ep)}


def ticker_chart(symbol: str, episodes: list[dict], base: Path | None = None) -> dict:
    """All daily bars for a chart symbol with every episode's fills on it (no per-trade window)."""
    raw = load_daily(symbol, "raw", base)
    if raw is None:
        return {"symbol": symbol, "label": symbol, "timeframe": "1d", "bars": [], "markers": [], "points": {},
                "error": f"no daily bars for {symbol}"}
    mine = [e for e in episodes if e["chart_symbol"] == symbol]
    markers, pts = [], {"entry": [], "exit": []}
    for ep in mine:
        for f in ep["fills"]:
            t = datetime.fromisoformat(f["time"])
            s = _snap_daily(raw.index, t)
            if s is None:
                continue
            k = s.strftime("%Y-%m-%d")
            markers.append((t, _marker(k, f, ep, compact=True)))
            if ep["kind"] != "option" and f["role"] in pts and f.get("price") is not None:
                pts[f["role"]].append((k, t, f["price"], abs(f["qty"])))
    markers.sort(key=lambda m: m[0])
    return {"symbol": symbol, "label": f"{symbol}: all trades", "timeframe": "1d", "bars": _daily_bars(raw),
            "markers": [m[1] for m in markers], "points": {k: _points(v) for k, v in pts.items()},
            "price_lines": [], "notes": [], "n_episodes": len(mine), "visible": None}
