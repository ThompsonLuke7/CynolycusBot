"""Bars behind the trade-history charts and entry features.

Data/personal/history_bars/ (gitignored):
  1d_raw/{SYM}.parquet     unadjusted daily bars. Schwab fill prices are raw, so a fill marker
                           only lands on the right price against raw bars.
  1d_split/{SYM}.parquet   split-adjusted daily bars. Entry features (returns, SMAs, highs)
                           use these so a split inside the lookback is not read as a crash.
  intraday/{EPISODE}.parquet  raw 15Min (windows <= 30 sessions) or 1Hour (<= 130) bars from
                           5 sessions before entry to 3 after exit, regular hours only.
  meta.json                what each file was fetched for. A file is refetched only when a
                           wider range is needed, or when it is still in use (open position,
                           watchlist, held) and a newer session has completed. Closed trades
                           on retired symbols are therefore fetched once, not every sync.

Alpaca SIP via core.API.Alpaca_API.market_data.fetch_intraday with limit=None (alpaca-py's
limit caps TOTAL bars) and skip_delisted=False (retired names still have history).
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from core.calendar import is_trading_day, next_trading_day, prev_trading_day
from core.personal_portfolio.data import ET, last_completed_session, personal_dir

logger = logging.getLogger(__name__)
FEATURE_LOOKBACK_DAYS = 400     # 252 sessions for 52-week highs + room for the 200-day average
CONTEXT_AFTER_DAYS = 60         # daily context kept after a closed trade
EXTRA_LOOKBACK_DAYS = 730       # watchlist / held names without trades: two years of context
INTRADAY_BEFORE, INTRADAY_AFTER = 5, 3
INTRADAY_15M_MAX, INTRADAY_1H_MAX = 30, 130
TF_SECONDS = {"15Min": 900, "1Hour": 3600}
_meta_lock = threading.Lock()

Fetcher = Callable[[str, datetime, datetime, str, str], pd.DataFrame]


def _fetch(symbol: str, start: datetime, end: datetime, timeframe: str, adjustment: str) -> pd.DataFrame:
    from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday

    return fetch_intraday(ticker=symbol, start=start, end=end, timeframe=timeframe, limit=None,
                          adjustment=adjustment, feed="sip", save_path="", skip_delisted=False)


def bars_dir(base: Path | None = None) -> Path:
    return personal_dir(base) / "history_bars"


def _meta_path(base: Path | None) -> Path:
    return bars_dir(base) / "meta.json"


def _read_meta(base: Path | None) -> dict:
    p = _meta_path(base)
    return json.loads(p.read_text()) if p.exists() else {}


def _update_meta(base: Path | None, key: str, value: dict) -> None:
    with _meta_lock:
        meta = _read_meta(base)
        meta[key] = value
        p = _meta_path(base)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1, sort_keys=True))
        tmp.replace(p)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ("timestamp", "open", "high", "low", "close", "volume", "vwap") if c in df.columns]
    out = df[keep].drop_duplicates("timestamp").sort_values("timestamp")
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)


def _fetch_end(now: datetime) -> datetime:
    # The SIP feed's newest 15 minutes need a paid entitlement (as data.refresh_bars).
    return now.astimezone(timezone.utc) - timedelta(minutes=20)


# ---------- daily ----------
def bar_plan(episodes: list[dict], extra_symbols: list[str], now: datetime) -> dict[str, dict]:
    """symbol -> {start, end, active}: the daily range each chart symbol needs."""
    today = now.astimezone(ET).date()
    plan: dict[str, dict] = {}

    def need(sym: str, start: date, end: date, active: bool) -> None:
        if not sym:
            return
        cur = plan.get(sym)
        if cur is None:
            plan[sym] = {"start": start, "end": min(end, today), "active": active}
        else:
            cur.update(start=min(cur["start"], start), end=min(max(cur["end"], end), today),
                       active=cur["active"] or active)

    for ep in episodes:
        start = date.fromisoformat(ep["open_date"]) - timedelta(days=FEATURE_LOOKBACK_DAYS)
        if ep["status"] == "closed":
            end = date.fromisoformat(ep["close_date"]) + timedelta(days=CONTEXT_AFTER_DAYS)
            if ep.get("expiry"):   # hold-to-expiry counterfactual needs the expiry close
                end = max(end, date.fromisoformat(ep["expiry"]) + timedelta(days=1))
            need(ep["chart_symbol"], start, end, False)
        else:
            need(ep["chart_symbol"], start, today, True)
    for sym in extra_symbols:
        need(sym, today - timedelta(days=EXTRA_LOOKBACK_DAYS), today, True)
    return plan


def _daily_needs_fetch(meta: dict | None, spec: dict, want_session: date) -> bool:
    if not meta:
        return True
    if spec["start"] < date.fromisoformat(meta["start"]):
        return True
    have_end = date.fromisoformat(meta["end"])
    target = min(spec["end"], want_session)
    return have_end < target


def refresh_daily(plan: dict[str, dict], now: datetime, base: Path | None = None,
                  fetch: Fetcher = _fetch, workers: int = 4) -> dict[str, str]:
    """Fetch raw and split-adjusted daily bars where the plan needs more than meta.json records."""
    meta = _read_meta(base)
    want = last_completed_session(now)
    end = _fetch_end(now)

    def one(sym: str) -> tuple[str, str]:
        spec = plan[sym]
        key = f"1d/{sym}"
        if not _daily_needs_fetch(meta.get(key), spec, want):
            return sym, "current"
        old = meta.get(key)
        start_d = min(spec["start"], date.fromisoformat(old["start"])) if old else spec["start"]
        start = datetime.combine(start_d, time(0), ET).astimezone(timezone.utc)
        try:
            frames = {adj: fetch(sym, start, end, "1Day", adj) for adj in ("raw", "split")}
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the rest
            return sym, f"error: {exc}"[:160]
        if any(f is None or f.empty for f in frames.values()):
            status = "empty"
        else:
            for adj, df in frames.items():
                _write(df, bars_dir(base) / f"1d_{adj}" / f"{sym}.parquet")
            status = "fetched"
        # Record the request either way, so a symbol with no bars is not refetched every sync.
        _update_meta(base, key, {"start": start_d.isoformat(), "end": min(spec["end"], want).isoformat(),
                                 "fetched_at": now.isoformat(), "status": status})
        return sym, status

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(one, sorted(plan)))


def _session_dates(ts: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(ts, utc=True).dt.tz_convert(ET).dt.normalize().dt.tz_localize(None))


def load_daily(symbol: str, adjustment: str, base: Path | None = None) -> pd.DataFrame | None:
    """OHLCV indexed by ET session date (naive), or None."""
    path = bars_dir(base) / f"1d_{adjustment}" / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.index = _session_dates(df["timestamp"])
    return out[~out.index.duplicated(keep="last")].sort_index()


# ---------- intraday ----------
def _shift_sessions(d: date, n: int) -> date:
    step = next_trading_day if n > 0 else prev_trading_day
    for _ in range(abs(n)):
        d = step(d)
    return d


def _count_sessions(a: date, b: date) -> int:
    n, d = 0, a
    while d <= b:
        n += is_trading_day(d)
        d += timedelta(days=1)
    return n


def intraday_spec(ep: dict, now: datetime) -> dict | None:
    """Timeframe and window of the episode's intraday chart, or None when the trade is too long."""
    open_d = date.fromisoformat(ep["open_date"])
    close_d = date.fromisoformat(ep["close_date"]) if ep.get("close_date") else now.astimezone(ET).date()
    start_d = _shift_sessions(open_d, -INTRADAY_BEFORE)
    end_d = _shift_sessions(close_d, INTRADAY_AFTER)
    cap = now.astimezone(ET).date()
    end_d = min(end_d, cap)
    n = _count_sessions(start_d, end_d)
    tf = "15Min" if n <= INTRADAY_15M_MAX else "1Hour" if n <= INTRADAY_1H_MAX else None
    if tf is None:
        return None
    return {"timeframe": tf, "start": start_d.isoformat(), "end": end_d.isoformat(), "sessions": n}


def ensure_intraday(ep: dict, now: datetime, base: Path | None = None, fetch: Fetcher = _fetch) -> str:
    spec = intraday_spec(ep, now)
    if spec is None:
        return "too long"
    key = f"intraday/{ep['id']}"
    old = _read_meta(base).get(key)
    want = last_completed_session(now)
    if old and old.get("timeframe") == spec["timeframe"] and old["start"] <= spec["start"] \
            and (old["end"] >= spec["end"] or old["end"] >= want.isoformat()):
        return "current"
    start = datetime.combine(date.fromisoformat(spec["start"]), time(0), ET).astimezone(timezone.utc)
    end = min(datetime.combine(date.fromisoformat(spec["end"]) + timedelta(days=1), time(0), ET).astimezone(timezone.utc),
              _fetch_end(now))
    try:
        df = fetch(ep["chart_symbol"], start, end, spec["timeframe"], "raw")
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"[:160]
    status = "empty" if df is None or df.empty else "fetched"
    if status == "fetched":
        _write(df, bars_dir(base) / "intraday" / f"{ep['id']}.parquet")
    _update_meta(base, key, {**spec, "symbol": ep["chart_symbol"], "fetched_at": now.isoformat(), "status": status,
                             "end": min(spec["end"], want.isoformat())})
    return status


def refresh_intraday(episodes: list[dict], now: datetime, base: Path | None = None,
                     fetch: Fetcher = _fetch, workers: int = 3) -> dict[str, str]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(zip([e["id"] for e in episodes], ex.map(lambda e: ensure_intraday(e, now, base, fetch), episodes)))


def load_intraday(episode_id: str, base: Path | None = None) -> tuple[pd.DataFrame | None, str | None]:
    """(bars with a UTC 'time' column, timeframe) for an episode, or (None, None)."""
    path = bars_dir(base) / "intraday" / f"{episode_id}.parquet"
    meta = _read_meta(base).get(f"intraday/{episode_id}") or {}
    if not path.exists():
        return None, None
    df = pd.read_parquet(path)
    if df.empty:
        return None, None
    df = df.assign(time=pd.to_datetime(df["timestamp"], utc=True)).sort_values("time").reset_index(drop=True)
    return df, meta.get("timeframe")


# ---------- prices at a moment ----------
def price_at(bars: pd.DataFrame | None, t: datetime, timeframe: str | None) -> tuple[float | None, str]:
    """The underlying's price known at time t, from intraday bars: the open of the bar containing t
    (never a later print), or the last close before t when t is outside regular hours."""
    if bars is None or bars.empty or timeframe not in TF_SECONDS:
        return None, "no intraday bars"
    ts = pd.Timestamp(t).tz_convert("UTC")
    i = int(bars["time"].searchsorted(ts, side="right")) - 1
    if i < 0:
        return None, "before the bars"
    row = bars.iloc[i]
    if ts < row["time"] + pd.Timedelta(seconds=TF_SECONDS[timeframe]):
        return float(row["open"]), f"{timeframe} bar open"
    return float(row["close"]), "last close before the fill"
