"""Trade library — every completed live/paper position, reconstructed for review.

The four 4H modules append one row per filled SELL to
``Data/inference/<module>/closed_trades.jsonl``. Those rows are *legs*, not
positions: a ``take_profit_+30%`` row is a partial TRIM that leaves the rest of
the position open (see ``core.live_4h_exec.exit_action``), while a stop or
horizon row closes the whole thing. Reading the ledger leg-by-leg makes trims
look like wins and full exits look like losses, so this module folds legs back
into positions before anything is reported.

Read-only. Never writes to a ledger, never contacts a broker.

Sources
  Data/inference/<module>/closed_trades.jsonl   realized sell legs (broker fills)
  <module live state>.json ``managed``          positions still open (for open legs)
  Data/shared/bars/4h/<TICKER>.parquet          chart bars, same cache the modules read
"""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.live_4h_exec import is_partial_trim

REPO = Path(__file__).resolve().parents[1]
LEDGER_ROOT = REPO / "Data/inference"
BARS_4H_DIR = REPO / "Data/shared/bars/4h"
BARS_1D_DIR = REPO / "Data/shared/bars/1d"

logger = logging.getLogger(__name__)

# module -> live state file holding the still-open ``managed`` book.
LIVE_STATES: dict[str, Path] = {
    "momentum_expansion": REPO / "strategies/momentum_expansion/live/momentum_live_state.json",
    "multi_ticker_swing_htf": REPO / "strategies/multi_ticker_swing_htf/live/htf_live_state.json",
    "meta_ranker": REPO / "signals/meta_context/meta_ranker/live_state.json",
    "dealer_ranker": REPO / "Data/inference/dealer_ranker/live_state.json",
}

# ---------------------------------------------------------------------------
# Input-fingerprint memo for the two whole-ledger folds.
#
# WHY: the hub polls every dashboard's /api/state on a 5s timer with a 2.5s
# read timeout, and the trade library's state() re-parsed the ledgers twice
# (build_positions + ledger_health) plus ~1MB of live-state JSON on every one.
# Standalone that is ~15ms; inside the combined_server process, sharing a GIL
# with the whole live stack, the same endpoint measured 1.2-2.1s -- close
# enough to the timeout that the hub kept dropping the socket mid-response,
# which is what produced the BrokenPipeError flood of 2026-08-24.
#
# The key is (path, mtime_ns, size) over every input file, not a clock, so a
# runner appending a leg invalidates it on the very next call. That keeps the
# original no-cache guarantee -- the book is never stale -- while making a
# repeat read cost a handful of stat() calls.
# ---------------------------------------------------------------------------
_memo_lock = threading.RLock()
_memo: dict[str, tuple[tuple, Any]] = {}


def _ledger_paths() -> list[Path]:
    return sorted(LEDGER_ROOT.glob("*/closed_trades.jsonl"))


def _fingerprint(paths: list[Path]) -> tuple:
    """Identity of a set of input files: path, last write, and size."""
    out: list[tuple] = [(str(LEDGER_ROOT),)]
    for path in paths:
        try:
            st = path.stat()
        except OSError:
            out.append((str(path), None, None))   # absent is itself a state
        else:
            out.append((str(path), st.st_mtime_ns, st.st_size))
    return tuple(out)


def _memoized(key: str, paths: list[Path], build):
    """Return ``build()``, reusing the last result while ``paths`` are unchanged.

    The build runs under the lock on purpose: concurrent dashboard requests
    should wait for one fold rather than each run their own, which is the
    contention this exists to remove.
    """
    fingerprint = _fingerprint(paths)
    with _memo_lock:
        cached = _memo.get(key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        value = build()
        _memo[key] = (fingerprint, value)
        return value


# Legacy rows put the OCC symbol in ``ticker``; newer rows put the underlying
# there and the OCC in ``order_symbol``. Normalise so a position groups correctly.
_OCC_RE = re.compile(r"^([A-Z]+)\d{6}[CP]\d{8}$")


def _base_ticker(value: str) -> str:
    m = _OCC_RE.match((value or "").strip().upper())
    return m.group(1) if m else (value or "").strip().upper()


def _ts(value) -> datetime | None:
    if not value:
        return None
    try:
        import pandas as pd
        t = pd.Timestamp(value)
    except Exception:
        return None
    if t is None or (hasattr(t, "value") and t.value != t.value):  # NaT
        return None
    try:
        return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).to_pydatetime()
    except Exception:
        return None


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def read_legs() -> list[dict[str, Any]]:
    """Every realized sell leg across all module ledgers, oldest first.

    Rows the repair tooling wrote to record data loss (``event`` set, no
    ``ticker``) are audit markers, not trades, and are returned separately by
    :func:`ledger_health` rather than silently dropped here.
    """
    legs: list[dict[str, Any]] = []
    for path in _ledger_paths():
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("trade_library: unparseable row %s:%d", path, lineno)
                continue
            if not isinstance(row, dict) or not row.get("ticker"):
                continue
            ts = _ts(row.get("ts"))
            if ts is None:
                continue
            legs.append({
                "module": row.get("module") or path.parent.name,
                "ticker": _base_ticker(row.get("ticker", "")),
                "order_symbol": row.get("order_symbol") or row.get("ticker"),
                "route": row.get("route") or "option",
                "ts": ts,
                "entry_bar": _ts(row.get("entry_bar")),
                "qty": _f(row.get("qty")),
                "entry_price": _f(row.get("entry_avg_price")),
                "exit_price": _f(row.get("exit_fill_price")),
                "pnl": _f(row.get("realized_pnl")),
                "reason": row.get("exit_reason") or "",
                "runs_held": row.get("runs_held"),
                "order_id": row.get("order_id"),
                # A FULL take-profit (`take_profit_full_+N%`) closes the position;
                # only a partial scale-out leaves a runner. Classifying on the
                # shared predicate keeps this reader in step with the engine.
                "is_trim": is_partial_trim(row.get("exit_reason")),
            })
    legs.sort(key=lambda r: r["ts"])
    return legs


def _f(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def ledger_health() -> dict[str, Any]:
    """Known gaps in the ledgers, so the UI can say what it cannot show.

    Surfacing this matters: an ``audit_gap`` marker means real trades are missing
    from the file and any total shown here is understated by that much.
    """
    return _memoized("ledger_health", _ledger_paths(), _ledger_health_uncached)


def _ledger_health_uncached() -> dict[str, Any]:
    gaps, missing_pnl = [], 0
    for path in _ledger_paths():
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("event"):
                gaps.append({"module": path.parent.name, "event": row.get("event"),
                             "detail": row.get("detail"), "at": row.get("repaired_at")})
            elif row.get("ticker") and _f(row.get("realized_pnl")) is None:
                missing_pnl += 1
    return {"gaps": gaps, "rows_missing_pnl": missing_pnl}


def _open_book() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for module, path in LIVE_STATES.items():
        if not path.exists():
            continue
        try:
            managed = json.loads(path.read_text()).get("managed", {})
        except (json.JSONDecodeError, OSError):
            logger.warning("trade_library: unreadable live state %s", path)
            continue
        for tkr, st in (managed or {}).items():
            if isinstance(st, dict):
                out[(module, _base_ticker(tkr))] = st
    return out


def build_positions() -> list[dict[str, Any]]:
    """Fold sell legs into positions, newest activity first.

    A position is one (module, ticker, entry episode). Legs are walked in time
    order and a new episode starts whenever a leg carries an ``entry_bar`` later
    than the current episode's — that is a re-entry into the same name, not more
    of the same trade.

    ``status`` is ``closed`` when a non-trim leg closed the position out, and
    ``open`` when the module still lists it in ``managed`` (i.e. only trims have
    been booked and a runner leg is still live). An open position's ``realized``
    covers the booked trims only; its remaining leg is marked to
    ``last_mark_price`` from live state and reported separately as unrealized —
    never added into realized totals.
    """
    return _memoized(
        "build_positions",
        _ledger_paths() + sorted(LIVE_STATES.values()),
        _build_positions_uncached,
    )


def _build_positions_uncached() -> list[dict[str, Any]]:
    open_book = _open_book()
    episodes: dict[tuple[str, str], list[dict]] = {}

    for leg in read_legs():
        key = (leg["module"], leg["ticker"])
        bucket = episodes.setdefault(key, [])
        cur = bucket[-1] if bucket else None
        new_episode = cur is None or cur["finished"]
        if not new_episode and leg["entry_bar"] and cur["entry_bar"]:
            new_episode = leg["entry_bar"] > cur["entry_bar"]
        if new_episode:
            cur = {"module": leg["module"], "ticker": leg["ticker"], "route": leg["route"],
                   "entry_bar": leg["entry_bar"], "legs": [], "finished": False}
            bucket.append(cur)
        if cur["entry_bar"] is None and leg["entry_bar"]:
            cur["entry_bar"] = leg["entry_bar"]
        cur["legs"].append(leg)
        if not leg["is_trim"]:
            cur["finished"] = True

    positions: list[dict[str, Any]] = []
    for (module, ticker), bucket in episodes.items():
        for i, ep in enumerate(bucket):
            legs = ep["legs"]
            last = legs[-1]
            still_open = (not ep["finished"]) and i == len(bucket) - 1
            st = open_book.get((module, ticker)) if still_open else None
            # Only an episode the module still manages counts as open. Without
            # this a trimmed-then-abandoned episode would claim a live mark.
            if still_open and st is None:
                still_open = False
            realized = sum(l["pnl"] for l in legs if l["pnl"] is not None)
            entry_px = next((l["entry_price"] for l in legs if l["entry_price"]), None)
            entry_bar = ep["entry_bar"] or (_ts(st.get("entry_bar")) if st else None)
            mark = _f((st or {}).get("last_mark_price"))
            open_ret = ((mark / entry_px - 1) * 100
                        if (still_open and mark and entry_px) else None)
            positions.append({
                "id": f"{module}:{ticker}:{_iso(entry_bar) or i}",
                "module": module, "ticker": ticker, "route": ep["route"],
                "status": "open" if still_open else "closed",
                "entry_bar": _iso(entry_bar),
                "first_sell": _iso(legs[0]["ts"]), "last_sell": _iso(last["ts"]),
                "entry_price": entry_px,
                "exit_price": last["exit_price"] if not still_open else mark,
                "realized": round(realized, 2),
                "open_mark": mark if still_open else None,
                "open_ret_pct": round(open_ret, 2) if open_ret is not None else None,
                "trims": sum(1 for l in legs if l["is_trim"]),
                "reasons": [l["reason"] for l in legs],
                "final_reason": "" if still_open else last["reason"],
                "hold_days": (round((last["ts"] - entry_bar).total_seconds() / 86400, 2)
                              if entry_bar else None),
                "u_entry": _f((st or {}).get("u_entry")),
                "u_atr": _f((st or {}).get("u_atr")),
                "legs": [{"ts": _iso(l["ts"]), "qty": l["qty"], "price": l["exit_price"],
                          "pnl": l["pnl"], "reason": l["reason"], "is_trim": l["is_trim"],
                          "order_id": l["order_id"], "symbol": l["order_symbol"]}
                         for l in legs],
            })
    positions.sort(key=lambda p: p["last_sell"] or "", reverse=True)
    return positions


def price_series(ticker: str, *, timeframe: str = "4h", days: int = 180) -> dict[str, Any]:
    """OHLC bars for the chart, read-only from the same cache the modules use.

    No resampling, gap-filling, or price adjustment — a missing bar is a
    genuinely missing bar (holiday, halt, or no coverage), and a split the
    pipeline has not processed will show as a real gap rather than be smoothed
    away. The chart must not imply data the strategies did not have.
    """
    import pandas as pd

    sym = (ticker or "").strip().upper()
    root = BARS_4H_DIR if timeframe == "4h" else BARS_1D_DIR
    path = root / f"{sym}.parquet"
    if not sym or not path.exists():
        return {"ticker": sym, "timeframe": timeframe, "bars": [],
                "error": f"no {timeframe} bar cache for {sym or '(none)'}"}
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 — a bad cache file must not 500 the page
        return {"ticker": sym, "timeframe": timeframe, "bars": [], "error": str(exc)}
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.assign(_t=idx).dropna(subset=["_t"]).sort_values("_t")
    if days and days > 0:
        df = df[df["_t"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days))]
    need = {"open", "high", "low", "close"}
    if not need <= set(df.columns):
        return {"ticker": sym, "timeframe": timeframe, "bars": [],
                "error": f"{path.name} lacks OHLC columns"}
    bars = [{"t": r["_t"].isoformat(), "o": float(r["open"]), "h": float(r["high"]),
             "l": float(r["low"]), "c": float(r["close"]),
             "v": _f(r.get("volume"))}
            for r in df.to_dict("records") if _f(r.get("close")) is not None]
    return {"ticker": sym, "timeframe": timeframe, "bars": bars}


def summary(positions: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [p for p in positions if p["status"] == "closed"]
    wins = [p for p in closed if p["realized"] > 0]
    open_p = [p for p in positions if p["status"] == "open"]
    return {
        "positions": len(positions),
        "closed": len(closed),
        "open": len(open_p),
        "realized_closed": round(sum(p["realized"] for p in closed), 2),
        "realized_on_open_trims": round(sum(p["realized"] for p in open_p), 2),
        "win_rate": round(100.0 * len(wins) / len(closed), 1) if closed else None,
        "best": max(closed, key=lambda p: p["realized"], default=None),
        "worst": min(closed, key=lambda p: p["realized"], default=None),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
