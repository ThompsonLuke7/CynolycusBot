"""Decision replay: step through a position bar by bar beside what the model saw.

The trade library (``UI/trade_library.py``) answers *what happened* to a
position. This module answers *why*. For every bar of a position's life it
lines up four things: the bars the model had, the decision it made on them
(score, rank, every gate and the full feature vector), the orders it planned,
and the fills that followed. An entry or an exit can then be read in the
context the module actually had.

Read-only. It never writes and never contacts a broker.

Sources (all already on disk):
  trade_library.build_positions()                   positions folded from the closed-trade ledgers
  Data/shared/bars/{1h,4h,1d}/<T>.parquet            chart bars (stamped at their open)
  Data/archive/intraday_1m/bars_<YYYYMMDD>.jsonl     1-minute bars, Intraday Structure only (~30 sessions)
  Data/research/decision_state/<module>/             decision_state_v1: every scored candidate of every
                                                     run with its full feature vector (from 2026-11-13)
  Data/inference/<module>/live_signal_audit.jsonl    picks and order plans (from 2026-07). This is the
                                                     only model evidence for positions older than the
                                                     archive, and it has no features
  Data/inference/intraday_structure/transitions.jsonl  setup state transitions, with their evidence

Timing (the replay's no-look-ahead contract):
  * Replay frame i means "bar i has closed". Only bars 0..i are drawn, and only
    events placed on bars <= i are shown.
  * An event sits on the chart bar that contains the moment it happened. A
    moment that falls between cached bars (overnight, or a session stretch the
    cache does not hold) sits on the next bar to open, never on an earlier one.
  * That moment is the execution time for a fill, and ``decision_ts`` (the wall
    clock) for an archived decision or a setup transition. Audit rows log only
    the bar they scored, so they sit where that bar ended. A later run that
    re-scored the same bar (the next-morning pass) sits on the next bar. Either
    can be late, never early. Order plans sit with their run.
  * The bar a decision scored, the last bar the model had, is carried
    separately (``scored_frame``) and drawn as its own marker.
  * A chart bar shorter than the bar a module decides on would put a decision
    inside the bar it scored, before that bar finished. Those timeframes are
    not offered for that module (e.g. no 1h chart for the 4H modules).
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import decision_state_archive as dsa
from UI import trade_library
from UI.trade_library import _base_ticker, _ts

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO / "Data/inference"
BARS_1M_DIR = REPO / "Data/archive/intraday_1m"
TRANSITIONS_LOG = REPO / "Data/inference/intraday_structure/transitions.jsonl"
ET = dsa.ET

# Ledger module name -> the name its decision_state writer uses.
ARCHIVE_MODULES: dict[str, str] = {
    "momentum_expansion": "momentum_expansion",
    "meta_ranker": "meta_ranker",
    "multi_ticker_swing_htf": "multi_ticker_swing_htf",
    "dealer_ranker": "dealer_ranker",
    "multi_ticker_swing": "multi_ticker_swing_30m",
    "spy_daytrader": "spy_day_trader",
}
# Modules whose live_signal_audit.jsonl holds picks and order plans.
AUDIT_MODULES = frozenset({"momentum_expansion", "meta_ranker", "multi_ticker_swing_htf", "dealer_ranker"})
# Modules that score their whole universe on every run. For these, a ticker
# missing from a run file was not scored. The 30m scanner writes one file per
# batch, so a ticker missing from one of its files says nothing.
WHOLE_UNIVERSE_RUNS = frozenset({"momentum_expansion", "meta_ranker", "multi_ticker_swing_htf", "dealer_ranker"})
KNOWN_MODULES = (*ARCHIVE_MODULES, "intraday_structure")

# Hours spanned by the bar each module decides on (0 = a point-in-time snapshot).
DECISION_SPAN_H: dict[str, float] = {
    "momentum_expansion": 4.0, "meta_ranker": 4.0, "multi_ticker_swing_htf": 4.0,
    "dealer_ranker": 0.0, "multi_ticker_swing": 0.5, "spy_daytrader": 0.25,
    "intraday_structure": 0.0,
}
TIMEFRAME_H: dict[str, float] = {"1m": 1 / 60, "1h": 1.0, "4h": 4.0, "1d": 24.0}
DEFAULT_TF = {"intraday_structure": "1m", "multi_ticker_swing": "1h", "spy_daytrader": "1h"}
# Context bars drawn before the entry and after the exit.
PAD_BARS = {"1m": (45, 45), "1h": (30, 20), "4h": (24, 16), "1d": (30, 20)}
# Calendar days loaded around the position so the padding has bars to take.
LOAD_DAYS = {"1m": (0, 0), "1h": (12, 8), "4h": (30, 20), "1d": (90, 60)}
# Longest window an ad-hoc (non-position) replay may ask for, in days.
MAX_FREE_DAYS = {"1m": 3, "1h": 45, "4h": 180, "1d": 730}
MAX_1M_DAYS = 5
# Below this many scored candidates a cross-sectional percentile means nothing.
MIN_XSEC = 20

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RUN_ID_RE = re.compile(r"^[0-9a-f]{12,32}$")
_STOP_RE = re.compile(r"^stop_-(\d+(?:\.\d+)?)%$")
_TP_RE = re.compile(r"^take_profit(?:_full)?_\+(\d+(?:\.\d+)?)%$")
_USTOP_RE = re.compile(r"^underlying_stop_-(\d+(?:\.\d+)?)atr$")


# --------------------------------------------------------------------------- helpers

def _num(value: Any) -> float | None:
    """A JSON-safe float, or None for missing, NaN or infinite values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _py(value: Any, *, max_str: int | None = None) -> Any:
    """Plain JSON value for one archive cell."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _num(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return _iso(value)
    text = str(value)
    if max_str is not None and len(text) > max_str:
        return text[:max_str] + "..."
    return text


def _iso(value: Any) -> str | None:
    ts = _utc(value)
    return ts.isoformat() if ts is not None else None


def _utc(value: Any) -> pd.Timestamp | None:
    """UTC Timestamp for a stamp. A naive stamp is read as UTC, the same rule as trade_library._ts."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        dt = _ts(value)
        if dt is None:
            return None
        ts = pd.Timestamp(dt)
    if pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def timeframe_options(module: str) -> list[str]:
    """Chart timeframes at least as long as the bar the module decides on."""
    span = DECISION_SPAN_H.get(module, 4.0)
    opts = [tf for tf, h in TIMEFRAME_H.items() if h >= span]
    # 1-minute bars exist for ~30 sessions and only intraday positions are short
    # enough to chart on them.
    return [tf for tf in opts if tf != "1m" or module == "intraday_structure"]


def default_timeframe(module: str) -> str:
    tf = DEFAULT_TF.get(module, "4h")
    return tf if tf in timeframe_options(module) else timeframe_options(module)[0]


def place(times_ns: np.ndarray, ts: Any, span_ns: int) -> int | None:
    """Index of the bar an event belongs on, or None when it is off the drawn bars.

    The bar whose [open, open + span) contains ``ts``. If ``ts`` falls between
    bars, the next bar to open. Never an earlier bar: placing an event early
    would show it to the replay before it happened.
    """
    t = _utc(ts)
    if t is None or len(times_ns) == 0:
        return None
    v = t.value
    i = int(np.searchsorted(times_ns, v, side="right")) - 1
    if i >= 0 and v < times_ns[i] + span_ns:
        return i
    return i + 1 if i + 1 < len(times_ns) else None


# --------------------------------------------------------------------------- bars

def load_bars(ticker: str, tf: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[list[dict], str | None]:
    """Chart bars for ``ticker`` with open stamps in [start, end]. No resampling or filling."""
    if tf == "1m":
        return _bars_1m(ticker, start, end)
    series = trade_library.price_series(ticker, timeframe=tf, days=0, start=start, end=end)
    return series.get("bars", []), series.get("error")


def _bars_1m(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[list[dict], str | None]:
    """1-minute bars from the stream archive, one day file per ET session.

    A minute the stream delivered twice keeps its later arrival, which is the
    stream's own correction of that minute.
    """
    days = pd.date_range(start.tz_convert(ET).normalize(), end.tz_convert(ET).normalize(), freq="D")
    if len(days) > MAX_1M_DAYS:
        return [], f"1m bars are limited to {MAX_1M_DAYS} sessions"
    bars, missing = [], []
    for day in days:
        path = BARS_1M_DIR / f"bars_{day.strftime('%Y%m%d')}.jsonl"
        if not path.exists():
            missing.append(day.strftime("%Y-%m-%d"))
            continue
        bars += [b for b in _scan_1m_day(str(path), _file_key(path), (ticker,)).get(ticker, [])
                 if start <= _utc(b["t"]) <= end]
    err = None
    if not bars:
        err = (f"no 1m archive for {ticker} on {', '.join(missing)}" if missing
               else f"no 1m bars for {ticker} in this window")
    return bars, err


@lru_cache(maxsize=32)
def _scan_1m_day(path: str, key: tuple, symbols: tuple[str, ...]) -> dict[str, list[dict]]:
    """Every listed symbol's bars from one archive day file, in one pass.

    ``key`` (path, mtime, size) is part of the cache key, so a file the stream
    is still appending to is re-read once it changes. A minute delivered twice
    keeps its later arrival, which is the stream's own correction of that minute.
    """
    needles = {s: (f'"symbol":"{s}"', f'"symbol": "{s}"') for s in symbols}
    best: dict[str, dict[str, dict]] = {s: {} for s in symbols}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            sym = next((s for s, nd in needles.items() if nd[0] in line or nd[1] in line), None)
            if sym is None:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t, c = _utc(r.get("timestamp")), _num(r.get("close"))
            if t is None or c is None or str(r.get("symbol") or "").upper() != sym:
                continue
            k = t.isoformat()
            prev = best[sym].get(k)
            if prev is None or str(r.get("arrival_at") or "") >= prev["_arrival"]:
                best[sym][k] = {"t": k, "o": _num(r.get("open")), "h": _num(r.get("high")),
                                "l": _num(r.get("low")), "c": c, "v": _num(r.get("volume")),
                                "_arrival": str(r.get("arrival_at") or "")}
    return {s: [{k: v for k, v in b.items() if k != "_arrival"} for _, b in sorted(rows.items())]
            for s, rows in best.items()}


# --------------------------------------------------------------------------- decision archive

_TIMELINE_COLS = ("run_id", "decision_ts", "bar_ts", "ticker", "side", "score", "score_name",
                  "rank", "eligible", "gate_reason", "selected", "action", "model_version",
                  "n_candidates")
# Momentum features that are distances from a price level, as (feature, clip bound).
# A value at its clip bound says only "at least this far", so it implies no level.
_IMPLIED_FEATURES = {
    "close": None, "atr_14": None,
    "ema_dist_10": 10.0, "ema_dist_20": 10.0, "ema_dist_50": 10.0, "ema_dist_100": 15.0,
    "dist_20bar_high_atr": 20.0, "range_pos_20": 1.0,
    "dist_to_52w_high_atr": 50.0, "dist_to_52w_low_atr": 50.0,
}


def implied_levels(f: dict[str, float | None]) -> dict[str, float]:
    """Price levels a momentum feature row implies, using the feature definitions in
    strategies/momentum_expansion/features/feature_matrix_4h.py: ``ema_dist_N =
    (close - EMA_N) / ATR14``, ``dist_20bar_high_atr = (high20 - close) / ATR14``,
    ``range_pos_20 = (close - low20) / (high20 - low20)``, and the same for the 52w range.

    Drawn on the chart at the scored bar, these show where the model thought each
    average and ceiling was. If they sit on the chart's own lines, the model saw
    what the chart shows.
    """
    c, atr = f.get("close"), f.get("atr_14")
    if c is None or atr is None or not atr > 0:
        return {}

    def usable(name: str) -> float | None:
        v = f.get(name)
        bound = _IMPLIED_FEATURES.get(name)
        if v is None or (bound is not None and abs(v) >= bound - 1e-6):
            return None
        return v

    out: dict[str, float] = {"close": c}
    for n in (10, 20, 50, 100):
        d = usable(f"ema_dist_{n}")
        if d is not None:
            out[f"ema_{n}"] = c - d * atr
    d = usable("dist_20bar_high_atr")
    if d is not None and d > 0:
        out["high_20"] = c + d * atr
        rp = usable("range_pos_20")
        if rp is not None and 0 < rp < 1:
            out["low_20"] = (c - rp * out["high_20"]) / (1 - rp)
    d = usable("dist_to_52w_high_atr")
    if d is not None and d > 0:
        out["high_52w"] = c + d * atr
    d = usable("dist_to_52w_low_atr")
    if d is not None and d > 0:
        out["low_52w"] = c - d * atr
    return {k: round(v, 4) for k, v in out.items()}


@lru_cache(maxsize=4096)
def _archive_file_summary(path: str, ticker: str) -> dict:
    """One run file reduced to the run header and ``ticker``'s row.

    Archive files are written once and never replaced, so the result is cached
    for the life of the process. Features are not read here. They are read per
    run by :func:`run_detail` when a decision is opened.
    """
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    cols = [c for c in _TIMELINE_COLS if c in names]
    cols += [c for c in names if c.startswith((dsa.GATE_PREFIX, dsa.EXTRA_PREFIX))]
    implied_cols = [dsa.FEATURE_PREFIX + k for k in _IMPLIED_FEATURES if dsa.FEATURE_PREFIX + k in names]
    df = pq.read_table(path, columns=cols + implied_cols).to_pandas()
    tick = df["ticker"].astype(str).str.upper()
    scores = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype="float64") \
        if "score" in df else np.full(len(df), np.nan)
    finite = np.isfinite(scores)
    picks = []
    if "selected" in df:
        sel = df.loc[df["selected"].astype(bool)]
        if "rank" in sel:
            sel = sel.sort_values("rank")
        picks = [str(t).upper() for t in sel["ticker"].tolist()[:12]]
    first = df.iloc[0]
    run = {
        "run_id": _py(first.get("run_id")),
        "decision_ts": _iso(first.get("decision_ts")),
        "bar_ts": _iso(first.get("bar_ts")),
        "model_version": _py(first.get("model_version")),
        "n_candidates": int(len(df)),
        "n_scored": int(finite.sum()),
        "n_eligible": int(df["eligible"].astype(bool).sum()) if "eligible" in df else None,
        "picks": picks,
    }
    hit = df.index[tick == ticker]
    if len(hit) == 0:
        return {"run": run, "row": None}
    r = df.loc[hit[0]]
    score = _num(r.get("score"))
    score_rank = int((scores[finite] > score).sum()) + 1 if score is not None else None
    n = run["n_scored"]
    row = {
        "bar_ts": _iso(r.get("bar_ts")),
        "side": _py(r.get("side")),
        "score": score,
        "score_name": _py(r.get("score_name")),
        # rank among the rows that passed every gate (decision_state_v1 semantics)
        "rank": _num(r.get("rank")),
        "eligible": _py(r.get("eligible")),
        "gate_reason": _py(r.get("gate_reason")),
        "selected": _py(r.get("selected")),
        "action": _py(r.get("action")),
        # rank of the score among every scored candidate, gated or not
        "score_rank": score_rank,
        "score_pct": (round(1.0 - (score_rank - 1) / (n - 1), 4)
                      if score_rank is not None and n > 1 else None),
        "gates": {c[len(dsa.GATE_PREFIX):]: _py(r.get(c)) for c in cols if c.startswith(dsa.GATE_PREFIX)},
        "extras": {c[len(dsa.EXTRA_PREFIX):]: _py(r.get(c), max_str=200)
                   for c in cols if c.startswith(dsa.EXTRA_PREFIX)},
    }
    if len(implied_cols) == len(_IMPLIED_FEATURES):
        row["implied"] = implied_levels({c[len(dsa.FEATURE_PREFIX):]: _num(r.get(c)) for c in implied_cols})
    return {"run": run, "row": row}


def archive_coverage(module: str) -> dict[str, Any]:
    """First and last ET date the module has archived runs for (directory names only)."""
    am = ARCHIVE_MODULES.get(module)
    if not am:
        return {"module": None, "first": None, "last": None}
    base = dsa.archive_root() / am
    days = sorted(p.name for p in base.glob("*") if p.is_dir() and _DATE_RE.match(p.name)) \
        if base.exists() else []
    return {"module": am, "first": days[0] if days else None, "last": days[-1] if days else None}


def archive_decisions(module: str, ticker: str, start: pd.Timestamp, end: pd.Timestamp
                      ) -> tuple[list[dict], list[str]]:
    """``ticker``'s archived decision rows whose scored bar lies in [start, end]."""
    am = ARCHIVE_MODULES.get(module)
    if not am:
        return [], []
    d0 = (start.tz_convert(ET) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    d1 = (end.tz_convert(ET) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out, warnings = [], []
    for path in dsa.list_partitions(module=am, start=d0, end=d1):
        try:
            summary = _archive_file_summary(str(path), ticker)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not hide the rest
            logger.warning("decision_replay: unreadable archive file %s: %s", path, exc)
            warnings.append(f"unreadable archive file {path.name}: {exc}")
            continue
        run, row = summary["run"], summary["row"]
        if row is None and module not in WHOLE_UNIVERSE_RUNS:
            continue
        bar = _utc((row or {}).get("bar_ts") or run["bar_ts"])
        if bar is None or not (start <= bar <= end):
            continue
        rec = {"source": "archive", "run_key": f"archive:{run['run_id']}", **run}
        if row is None:
            rec.update({"scored": False, "bar_ts": run["bar_ts"]})
        else:
            rec.update({"scored": True, **row})
        out.append(rec)
    return out, warnings


def find_run_file(archive_module: str, run_id: str) -> Path | None:
    if archive_module not in ARCHIVE_MODULES.values() or not _RUN_ID_RE.match(run_id or ""):
        return None
    hits = sorted((dsa.archive_root() / archive_module).glob(f"*/*_{run_id[:12]}.parquet"))
    return hits[0] if hits else None


def _percentiles(frame: pd.DataFrame, fcols: list[str], values: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each value within its column, over finite entries only."""
    if len(frame) < MIN_XSEC:
        return np.full(len(fcols), np.nan)
    mat = frame[fcols].to_numpy(dtype="float64")
    finite = np.isfinite(mat)
    with np.errstate(invalid="ignore"):
        less = ((mat < values) & finite).sum(axis=0)
        equal = ((mat == values) & finite).sum(axis=0)
    n = finite.sum(axis=0)
    out = np.where(n >= MIN_XSEC, (less + 0.5 * equal) / np.maximum(n, 1), np.nan)
    return np.where(np.isfinite(values), out, np.nan)


def _json_cell(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return value


def _feature_view(path: Path, ticker: str) -> dict | None:
    """``ticker``'s feature vector in one run, with each value's cross-sectional percentile."""
    df = pd.read_parquet(path)
    tick = df["ticker"].astype(str).str.upper()
    hit = df.index[tick == ticker]
    if len(hit) == 0:
        return None
    fcols = [c for c in df.columns if c.startswith(dsa.FEATURE_PREFIX)]
    values = df.loc[hit[0], fcols].to_numpy(dtype="float64")
    return {"frame": df, "row": df.loc[hit[0]], "fcols": fcols,
            "values": values, "pct": _percentiles(df, fcols, values)}


def run_detail(module: str, run_id: str, ticker: str, ref_run_id: str | None = None) -> dict[str, Any]:
    """Everything one archived run knew about ``ticker``.

    Returns the run header with each input's age at decision time, the ticker's
    output and gates, its full feature vector with cross-sectional percentiles,
    and what the run picked instead. With ``ref_run_id`` (normally the entry
    decision) each feature also carries its value and percentile in that run,
    so the drift since entry can be read directly.
    """
    am = ARCHIVE_MODULES.get(module)
    ticker = (ticker or "").strip().upper()
    if not am or not _TICKER_RE.match(ticker):
        return {"error": "unknown module or ticker"}
    path = find_run_file(am, run_id)
    if path is None:
        return {"error": f"no archived run {run_id} for {am}"}
    view = _feature_view(path, ticker)
    df = view["frame"] if view else pd.read_parquet(path)
    first = df.iloc[0]
    decision_ts = _utc(first.get("decision_ts"))
    run_meta = _json_cell(first.get("run_meta_json")) or {}
    model_features = set(run_meta.pop("model_features", None) or []) if isinstance(run_meta, dict) else set()
    inputs = _json_cell(first.get("inputs_asof_json")) or {}
    asof = []
    for k, v in (inputs.items() if isinstance(inputs, dict) else []):
        # Only full timestamps get an age; a bare date would read as midnight UTC.
        t = _utc(v) if isinstance(v, str) and "T" in v and v[:4].isdigit() else None
        age_h = round((decision_ts - t).total_seconds() / 3600, 2) if (t is not None and decision_ts is not None) else None
        asof.append({"input": k, "value": _py(v, max_str=120), "age_h": age_h})
    gcols = [c for c in df.columns if c.startswith(dsa.GATE_PREFIX)]
    scores = pd.to_numeric(df["score"], errors="coerce")
    leaders = df.assign(_s=scores).dropna(subset=["_s"]).sort_values("_s", ascending=False).head(10)
    sel = df.loc[df["selected"].astype(bool)].sort_values("rank") if "selected" in df else df.iloc[0:0]
    out: dict[str, Any] = {
        "module": module, "archive_module": am, "run_id": _py(first.get("run_id")), "ticker": ticker,
        "file": str(path.relative_to(dsa.archive_root())),
        "meta": {
            "decision_ts": _iso(decision_ts),
            "inference_start": _iso(first.get("inference_start")),
            "inference_end": _iso(first.get("inference_end")),
            "model_version": _py(first.get("model_version")),
            "model_hash": _py(first.get("model_hash")),
            "feature_schema_hash": _py(first.get("feature_schema_hash")),
            "universe_hash": _py(first.get("universe_hash")),
            "git_commit": _py(first.get("git_commit")),
            "git_dirty": _py(first.get("git_dirty")),
            "n_candidates": int(len(df)),
            "n_scored": int(scores.notna().sum()),
            "n_eligible": int(df["eligible"].astype(bool).sum()) if "eligible" in df else None,
            "regime": _json_cell(first.get("regime_json")),
            "run_meta": run_meta,
            "inputs_asof": asof,
        },
        "gate_pass_rates": {c[len(dsa.GATE_PREFIX):]: round(float(df[c].astype(bool).mean()), 4) for c in gcols},
        "picks": [{"ticker": str(r["ticker"]).upper(), "score": _num(r.get("score")),
                   "rank": _num(r.get("rank")), "action": _py(r.get("action"))}
                  for _, r in sel.head(15).iterrows()],
        "leaders": [{"ticker": str(r["ticker"]).upper(), "score": _num(r["_s"]),
                     "eligible": _py(r.get("eligible")), "gate_reason": _py(r.get("gate_reason")),
                     "selected": _py(r.get("selected"))} for _, r in leaders.iterrows()],
        "xsec_note": (None if len(df) >= MIN_XSEC else
                      f"only {len(df)} candidates in this run file; percentiles need {MIN_XSEC}"),
    }
    if view is None:
        out["row"] = None
        out["features"] = []
        out["note"] = f"{ticker} was not scored in this run"
        return out
    r = view["row"]
    out["row"] = {
        "bar_ts": _iso(r.get("bar_ts")), "side": _py(r.get("side")), "score": _num(r.get("score")),
        "score_name": _py(r.get("score_name")), "rank": _num(r.get("rank")),
        "eligible": _py(r.get("eligible")), "gate_reason": _py(r.get("gate_reason")),
        "selected": _py(r.get("selected")), "action": _py(r.get("action")),
        "gates": {c[len(dsa.GATE_PREFIX):]: _py(r.get(c)) for c in gcols},
        "extras": {c[len(dsa.EXTRA_PREFIX):]: _py(r.get(c))
                   for c in df.columns if c.startswith(dsa.EXTRA_PREFIX)},
    }
    ref: dict[str, tuple[float | None, float | None]] = {}
    if ref_run_id and ref_run_id != run_id:
        ref_path = find_run_file(am, ref_run_id)
        ref_view = _feature_view(ref_path, ticker) if ref_path else None
        if ref_view is not None:
            ref = {c: (_num(v), _num(p)) for c, v, p in zip(ref_view["fcols"], ref_view["values"], ref_view["pct"])}
            out["ref"] = {"run_id": ref_run_id, "decision_ts": _iso(ref_view["row"].get("decision_ts")),
                          "bar_ts": _iso(ref_view["row"].get("bar_ts"))}
        else:
            out["ref"] = {"run_id": ref_run_id, "error": "reference run or ticker row not found"}
    out["features"] = [
        {"name": c[len(dsa.FEATURE_PREFIX):], "value": _num(v), "pct": _num(p),
         "model_input": (c[len(dsa.FEATURE_PREFIX):] in model_features) if model_features else None,
         "ref_value": ref.get(c, (None, None))[0], "ref_pct": ref.get(c, (None, None))[1]}
        for c, v, p in zip(view["fcols"], view["values"], view["pct"])
    ]
    return out


# --------------------------------------------------------------------------- signal audit

_audit_lock = threading.RLock()
_audit_memo: dict[str, tuple[tuple, dict]] = {}


def _file_key(path: Path) -> tuple:
    try:
        st = path.stat()
    except OSError:
        return (str(path), None, None)
    return (str(path), st.st_mtime_ns, st.st_size)


def _compact_audit(audit: dict | None) -> dict:
    audit = audit or {}
    extra = {k: v for k, v in (audit.get("extra") or {}).items()
             if v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 80)}
    return {"score": _num(audit.get("score")), "rank": _num(audit.get("rank")),
            "rank_pct": _num(audit.get("rank_pct")), "side": audit.get("side"),
            "extras": {k: (_num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                       for k, v in extra.items()},
            "decision_run_id": audit.get("decision_run_id")}


def audit_index(module: str) -> dict:
    """The module's signal-audit log folded into runs, rebuilt only when the file changes."""
    path = AUDIT_ROOT / module / "live_signal_audit.jsonl"
    key = _file_key(path)
    with _audit_lock:
        hit = _audit_memo.get(module)
        if hit is not None and hit[0] == key:
            return hit[1]
        idx = _build_audit_index(path, module)
        _audit_memo[module] = (key, idx)
        return idx


def _build_audit_index(path: Path, module: str) -> dict:
    """Runs and per-ticker records from ``signal_decision`` and ``order_plan`` events.

    The log has no run id before the archive existed, so runs are recovered from
    event order. Meta, HTF and Dealer write one ``signal_decision`` per run.
    Momentum writes one per checked ticker. An ``order_plan`` for the same bar
    closes the run.
    """
    runs: list[dict] = []
    tickers: dict[str, dict[int, dict]] = defaultdict(dict)
    if not path.exists():
        return {"runs": runs, "tickers": tickers}
    cur: dict | None = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"signal_decision"' not in line and '"order_plan"' not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind, bar = ev.get("event"), _utc(ev.get("bar"))
            if kind not in ("signal_decision", "order_plan") or bar is None:
                continue
            if kind == "signal_decision":
                if (cur is None or cur["closed"] or cur["bar"] != bar
                        or module != "momentum_expansion"):
                    cur = {"n": len(runs), "bar": bar, "targets": None, "skip": None,
                           "run_ids": set(), "closed": False, "plan": False}
                    runs.append(cur)
                if cur["targets"] is None:
                    cur["targets"] = set()
                if module == "momentum_expansion":
                    tkr = str(ev.get("ticker") or "").upper()
                    if not tkr:
                        cur["skip"] = ev.get("skip_reason") or "skipped"
                        continue
                    cur["targets"].add(tkr)
                    rec = tickers[tkr].setdefault(cur["n"], {})
                    rec.update(_compact_audit(ev.get("signal_audit")))
                    rec["entry_triggered"] = ev.get("entry_triggered")
                    rec["skip_reason"] = ev.get("skip_reason")
                else:
                    cur["targets"].update(str(t).upper() for t in ev.get("targets") or [])
                    for tkr, audit in (ev.get("signal_audits") or {}).items():
                        tickers[str(tkr).upper()].setdefault(cur["n"], {}).update(_compact_audit(audit))
                for rec_audit in (ev.get("signal_audits") or {}).values():
                    if isinstance(rec_audit, dict) and rec_audit.get("decision_run_id"):
                        cur["run_ids"].add(rec_audit["decision_run_id"])
                if isinstance(ev.get("signal_audit"), dict) and ev["signal_audit"].get("decision_run_id"):
                    cur["run_ids"].add(ev["signal_audit"]["decision_run_id"])
                continue
            # order_plan
            if cur is None or cur["closed"] or cur["bar"] != bar:
                cur = {"n": len(runs), "bar": bar, "targets": None, "skip": None,
                       "run_ids": set(), "closed": False, "plan": False}
                runs.append(cur)
            cur["closed"] = cur["plan"] = True
            for audit in (ev.get("signal_audits") or {}).values():
                if isinstance(audit, dict) and audit.get("decision_run_id"):
                    cur["run_ids"].add(audit["decision_run_id"])
            for tkr, sel in (ev.get("contract_selection") or {}).items():
                sel = sel or {}
                tickers[str(tkr).upper()].setdefault(cur["n"], {})["route"] = {
                    k: _py(sel.get(k), max_str=80) for k in
                    ("action", "reason", "ref_price", "shares", "occ", "strike", "expiry", "delta")
                    if sel.get(k) is not None}
            disp = ev.get("dispositions") or {}
            orders = ev.get("planned") or ev.get("plan") or []
            if not orders:
                # Deferred runs leave plan/planned empty and keep the intent in order_audits.
                orders = [{"symbol": sym, "side": a.get("side"), "qty": a.get("qty"),
                           "reason": a.get("reason") or ("entry" if a.get("side") == "buy" else None),
                           "route": a.get("instrument")}
                          for sym, a in (ev.get("order_audits") or {}).items() if isinstance(a, dict)]
            for o in orders:
                sym = str(o.get("symbol") or "")
                if not sym:
                    continue
                rec = tickers[_base_ticker(sym)].setdefault(cur["n"], {})
                rec.setdefault("orders", []).append({
                    "symbol": sym, "side": o.get("side"), "qty": _num(o.get("qty")),
                    "reason": o.get("reason"), "route": o.get("route"),
                    "disposition": o.get("disposition") or disp.get(sym)})
    for run in runs:
        run.pop("closed", None)
    return {"runs": runs, "tickers": tickers}


def audit_records(module: str, ticker: str, start: pd.Timestamp, end: pd.Timestamp,
                  archived_run_ids: set[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """``ticker``'s audit-derived decisions, order events and run list in [start, end].

    A run the archive also holds (matched by ``decision_run_id``) gives no audit
    decision row, because the archive row is richer. Its orders are still
    returned, since only the audit log has them.
    """
    if module not in AUDIT_MODULES:
        return [], [], []
    idx = audit_index(module)
    recs = idx["tickers"].get(ticker, {})
    decisions, orders, runs = [], [], []
    for run in idx["runs"]:
        bar = run["bar"]
        if not (start <= bar <= end):
            continue
        dup = bool(run["run_ids"] & archived_run_ids)
        key = f"audit:{module}:{run['n']}"
        runs.append({"run_key": key, "bar_ts": bar.isoformat(), "archived": dup})
        rec = recs.get(run["n"], {})
        if not dup and run["targets"] is not None:
            in_targets = ticker in run["targets"]
            row = {k: v for k, v in rec.items() if k not in ("orders", "route")}
            decisions.append({"source": "audit", "run_key": key, "bar_ts": bar.isoformat(),
                              "in_targets": in_targets, "run_skip": run["skip"],
                              "routing": rec.get("route"), **row})
        for o in rec.get("orders") or []:
            # ``routing`` is the contract selection; the order keeps its own ``route``.
            orders.append({"run_key": key, "bar_ts": bar.isoformat(), "routing": rec.get("route"), **o})
    return decisions, orders, runs


# --------------------------------------------------------------------------- intraday structure

def transition_decisions(ticker: str, start: pd.Timestamp, end: pd.Timestamp,
                         setup_id: str | None = None, path: Path | None = None) -> list[dict]:
    """Setup state transitions for ``ticker``, scanned by substring (the log is ~30 MB)."""
    path = path or TRANSITIONS_LOG
    if not path.exists():
        return []
    needles = (f'"ticker":"{ticker}"', f'"ticker": "{ticker}"')
    out = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if needles[0] not in line and needles[1] not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = _utc(ev.get("timestamp"))
            if t is None or not (start <= t <= end):
                continue
            to_state = str(ev.get("to_state") or "")
            out.append({
                "source": "transitions", "run_key": f"transition:{ev.get('setup_id')}:{t.isoformat()}:{to_state}",
                "bar_ts": t.isoformat(), "decision_ts": t.isoformat(),
                "setup_id": ev.get("setup_id"), "setup_type": ev.get("setup_type"),
                "from_state": ev.get("from_state"), "to_state": to_state,
                "action": f"{ev.get('from_state')} -> {to_state}", "reason": ev.get("reason"),
                "evidence": ev.get("evidence"), "spot": _num(ev.get("spot")),
                "selected": to_state == "CONFIRMED",
                "position_setup": bool(setup_id) and ev.get("setup_id") == setup_id,
            })
    return out


# --------------------------------------------------------------------------- exit policy

def _policy(module: str):
    """The exit policy the 5-minute risk pass enforces for ``module`` today, or None.

    Imported lazily: inside the combined server these modules are already
    loaded, and a standalone page should not pay for them until it is asked.
    """
    try:
        from scripts.live_risk_pass import MODULES
    except Exception as exc:  # noqa: BLE001 - levels are an annotation, not required
        logger.debug("decision_replay: risk-pass policy unavailable: %s", exc)
        return None
    spec = MODULES.get(module)
    return spec.policy if spec is not None else None


def _reason_value(reasons: list[str], pattern: re.Pattern) -> float | None:
    for reason in reasons:
        m = pattern.match(str(reason or ""))
        if m:
            return float(m.group(1))
    return None


def position_levels(pos: dict) -> tuple[list[dict], dict, dict | None, list[str]]:
    """Price levels to draw, the gain reference, the policy summary and notes.

    A level is drawn only when its inputs were recorded. A threshold named in an
    exit reason (``stop_-39%``) is the one that fired, so it is preferred.
    Otherwise the module's current risk-pass policy is used and labelled as such,
    because a policy can change after a position was opened.
    """
    module, route = pos["module"], pos.get("route")
    reasons = pos.get("reasons") or []
    policy = _policy(module)
    notes: list[str] = []
    levels: list[dict] = []
    entry_px = _num(pos.get("entry_fill_price")) or _num(pos.get("entry_price"))
    anchor = pos.get("entry_anchor") or {}
    u_entry = _num(pos.get("u_entry")) or _num(anchor.get("u_entry"))
    u_atr = _num(pos.get("u_atr")) or _num(anchor.get("u_atr"))
    ref: dict[str, Any] = {"price": None, "basis": None}

    def src(found: float | None) -> str:
        return "from exit reason" if found is not None else "current risk-pass policy"

    if route == "equity" and entry_px:
        ref = {"price": entry_px, "basis": "entry fill price (shares)"}
        levels.append({"key": "entry", "price": entry_px, "label": f"entry fill {entry_px:.2f}"})
        stop = _reason_value(reasons, _STOP_RE)
        stop_frac = stop / 100 if stop is not None else (policy.stop_loss if policy else None)
        if stop_frac:
            levels.append({"key": "stop", "price": round(entry_px * (1 - stop_frac), 4),
                           "label": f"stop -{stop_frac:.0%} ({src(stop)})"})
        tp = _reason_value(reasons, _TP_RE)
        tp_frac = tp / 100 if tp is not None else (policy.take_profit if policy else None)
        if tp_frac:
            levels.append({"key": "take_profit", "price": round(entry_px * (1 + tp_frac), 4),
                           "label": f"take profit +{tp_frac:.0%} ({src(tp)})"})
    elif route == "option":
        if u_entry:
            ref = {"price": u_entry, "basis": "underlying at entry (recorded anchor)"}
            levels.append({"key": "u_entry", "price": u_entry, "label": f"underlying at entry {u_entry:.2f}"})
        k = _reason_value(reasons, _USTOP_RE)
        k = k if k is not None else (policy.underlying_stop_atr if policy else None)
        if u_entry and u_atr and k:
            levels.append({"key": "u_stop", "price": round(u_entry - k * u_atr, 4),
                           "label": f"underlying stop: entry - {k:g} x ATR {u_atr:.2f} "
                                    f"({src(_reason_value(reasons, _USTOP_RE))})"})
        notes.append("Option premium is not replayed. The bar cache holds the underlying only, "
                     "and option trade prints are stale prints, not marks.")
    summary = None
    if policy is not None:
        summary = {"stop_loss": policy.stop_loss, "take_profit": policy.take_profit,
                   "scale_frac": policy.scale_frac, "trail_stop": policy.trail_stop,
                   "horizon_runs": policy.horizon_bars,
                   "underlying_stop_atr": policy.underlying_stop_atr,
                   "source": "current risk-pass policy (scripts/live_risk_pass.py)"}
    return levels, ref, summary, notes


# --------------------------------------------------------------------------- contract and setup

_OCC_FULL = re.compile(r"^([A-Z][A-Z.]{0,5})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")


def parse_occ(symbol: Any) -> dict | None:
    """``SPY261105P00764000`` -> underlying, expiry, right and strike."""
    m = _OCC_FULL.match(str(symbol or "").strip().upper())
    if not m:
        return None
    root, yy, mm, dd, cp, strike = m.groups()
    return {"occ": m.group(0), "underlying": root, "expiry": f"20{yy}-{mm}-{dd}",
            "right": "call" if cp == "C" else "put", "strike": int(strike) / 1000}


def contract_info(pos: dict) -> dict | None:
    """The option contract a position held, with its terms at entry.

    Read from the OCC symbol the order went out with, so it is the contract
    actually traded. Moneyness is measured against the underlying anchor the
    runner recorded at entry; premiums are broker fills.
    """
    if pos.get("route") != "option":
        return None
    symbols = [pos.get("occ"), *[leg.get("symbol") for leg in pos.get("legs") or []], pos.get("ticker")]
    c = next((parse_occ(s) for s in symbols if parse_occ(s)), None)
    if c is None:
        return {"occ": None, "note": "no OCC symbol recorded for this position"}
    entry_t = _utc(pos.get("entry_filled_at") or pos.get("entry_submitted_at") or pos.get("entry_bar"))
    if entry_t is not None:
        c["dte_at_entry"] = (pd.Timestamp(c["expiry"]).date() - entry_t.tz_convert(ET).date()).days
    anchor = pos.get("entry_anchor") or {}
    u_entry = _num(pos.get("u_entry")) or _num(anchor.get("u_entry"))
    if u_entry:
        otm = (c["strike"] / u_entry - 1) if c["right"] == "call" else (1 - c["strike"] / u_entry)
        c["underlying_at_entry"] = u_entry
        c["otm_pct_at_entry"] = round(otm, 5)
    premium = _num(pos.get("entry_fill_price")) or _num(pos.get("entry_price"))
    legs = [leg for leg in pos.get("legs") or [] if _num(leg.get("price")) is not None and _num(leg.get("qty"))]
    qty = sum(_num(leg["qty"]) for leg in legs)
    c["entry_premium"] = premium
    c["contracts"] = qty or None
    if legs and qty:
        c["exit_premium_avg"] = round(sum(_num(leg["price"]) * _num(leg["qty"]) for leg in legs) / qty, 4)
        if premium:
            c["premium_return"] = round(c["exit_premium_avg"] / premium - 1, 4)
    if premium and qty:
        c["cost"] = round(premium * 100 * qty, 2)
    return c


CLOSED_SETUPS = REPO / "Data/inference/intraday_structure/closed_setups.jsonl"
_SETUP_KEEP = ("setup_id", "setup_type", "direction", "pivot", "initial_invalidation", "final_invalidation",
               "targets", "target_level_type", "target_level_sources", "target_index_reached", "extensions",
               "entry_time", "entry_price", "exit_time", "exit_price", "exit_reason", "terminal_state",
               "confirmed_at", "candidate_timestamp", "candidate_sources", "candidate_score", "confidence",
               "runway_score", "expected_reward_risk", "realized_r_after_costs", "risk_points",
               "mfe_points", "mae_points", "context_regime", "evidence", "warnings", "options_source",
               "dealer_plate_score")


def setup_record(setup_id: str | None, near: Any, path: Path | None = None) -> dict | None:
    """The engine's closed-setup record for ``setup_id`` whose entry is nearest ``near``.

    It holds the levels the setup actually traded against: the pivot it
    reacted to, the invalidation (initial, and where it had trailed to by the
    exit) and the targets, with the level types each target was built from.
    """
    path = path or CLOSED_SETUPS
    t0 = _utc(near)
    if not setup_id or t0 is None or not path.exists():
        return None
    needles = (f'"setup_id":"{setup_id}"', f'"setup_id": "{setup_id}"')
    best, best_gap = None, pd.Timedelta(days=1)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if needles[0] not in line and needles[1] not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = _utc(rec.get("entry_time"))
            if t is None:
                continue
            gap = abs(t - t0)
            if gap <= best_gap:
                best, best_gap = rec, gap
    if best is None:
        return None
    return {k: best.get(k) for k in _SETUP_KEEP}


def setup_levels(setup: dict) -> list[dict]:
    """Price lines for the setup's own levels."""
    out = []
    if _num(setup.get("pivot")):
        out.append({"key": "pivot", "price": _num(setup["pivot"]),
                    "label": f"setup level {_num(setup['pivot']):.2f} ({setup.get('setup_type') or 'pivot'})"})
    if _num(setup.get("initial_invalidation")):
        fin = _num(setup.get("final_invalidation"))
        out.append({"key": "invalidation", "price": _num(setup["initial_invalidation"]),
                    "label": f"invalidation {_num(setup['initial_invalidation']):.2f}"
                             + (f" (trailed to {fin:.2f} by the exit)" if fin else "")})
    for i, t in enumerate(setup.get("targets") or [], 1):
        if _num(t):
            out.append({"key": f"target_{i}", "price": _num(t), "label": f"target {i} {_num(t):.2f}"})
    return out


# --------------------------------------------------------------------------- overlays

# Bars in a 52-week window per chart timeframe. Momentum's 4H feature uses 504
# (two 4H bars a day); the others are the same span on their own bar size.
_BARS_52W = {"4h": 504, "1h": 1764, "1d": 252}
_TECH_META = {
    "ema_10": ("EMA 10", "price", "moving averages"), "ema_20": ("EMA 20", "price", "moving averages"),
    "ema_50": ("EMA 50", "price", "moving averages"), "ema_100": ("EMA 100", "price", "moving averages"),
    "high_20": ("20-bar high (ceiling)", "price", "ceilings / floors"),
    "low_20": ("20-bar low (floor)", "price", "ceilings / floors"),
    "high_60": ("60-bar base high", "price", "ceilings / floors"),
    "low_60": ("60-bar base low", "price", "ceilings / floors"),
    "high_52w": ("52-week high", "price", "ceilings / floors"),
    "low_52w": ("52-week low", "price", "ceilings / floors"),
    "rsi_14": ("RSI 14", "osc", None), "adx_14": ("ADX 14", "osc", None),
    "macd_hist_norm": ("MACD hist / ATR", "osc", None), "atr_14": ("ATR 14", "osc", None),
    "atr_pct_14": ("ATR 14 / close", "osc", None), "range_pos_20": ("position in 20-bar range", "osc", None),
    "ema_dist_20": ("(close - EMA20) / ATR", "osc", None), "dist_20bar_high_atr": ("ATRs below 20-bar high", "osc", None),
    "volume": ("volume", "osc", None),
}
# Momentum model inputs that these series feed (as ATR distances or ratios).
_MOMENTUM_INPUTS = {"ema_10", "ema_20", "ema_50", "ema_100", "high_20", "low_20", "high_60", "low_60",
                    "high_52w", "low_52w", "rsi_14", "adx_14", "macd_hist_norm", "atr_pct_14",
                    "range_pos_20", "ema_dist_20", "dist_20bar_high_atr"}


def _series(values: Any) -> list[float | None]:
    return [None if v is None or not math.isfinite(v) else round(float(v), 6) for v in np.asarray(values, dtype="float64")]


def technical_series(module: str, ticker: str, tf: str, bars: list[dict]) -> dict:
    """Momentum's own indicator formulas on the chart's bars, aligned to ``bars``.

    Computed on the ticker's full cached history, then cut to the window, so
    the EMAs and rolling windows are warmed up the way the model's are. On 4H
    bars these are the values the momentum model reads (as ATR distances); on
    other bar sizes they are the same formulas, and the page says so.
    """
    from strategies.momentum_expansion.features.feature_matrix_4h import (  # same helpers as the model
        _adx, _atr, _ema, _macd_hist, _rsi,
    )
    if not bars:
        return {"series": {}, "meta": {}}
    hist = trade_library.price_series(ticker, timeframe=tf, days=0, end=pd.Timestamp(bars[-1]["t"])).get("bars", [])
    if not hist:
        return {"series": {}, "meta": {}}
    df = pd.DataFrame(hist)
    df.index = pd.to_datetime(df.pop("t"), utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    c, h, lo = df["close"], df["high"], df["low"]
    atr14 = _atr(df, 14)
    n52 = _BARS_52W.get(tf, 504)
    ema = {n: _ema(c, n) for n in (10, 20, 50, 100)}
    rh20, rl20 = h.rolling(20).max(), lo.rolling(20).min()
    out = pd.DataFrame({
        **{f"ema_{n}": e for n, e in ema.items()},
        "high_20": rh20, "low_20": rl20,
        "high_60": h.rolling(60).max(), "low_60": lo.rolling(60).min(),
        "high_52w": h.rolling(n52, min_periods=n52 // 2).max(),
        "low_52w": lo.rolling(n52, min_periods=n52 // 2).min(),
        "rsi_14": _rsi(c, 14), "adx_14": _adx(df, 14),
        "macd_hist_norm": (_macd_hist(c) / atr14).clip(-5, 5),
        "atr_14": atr14, "atr_pct_14": atr14 / c.replace(0, np.nan),
        "range_pos_20": ((c - rl20) / (rh20 - rl20).replace(0, np.nan)).clip(0, 1),
        "ema_dist_20": ((c - ema[20]) / atr14).clip(-10, 10),
        "dist_20bar_high_atr": ((rh20 - c) / atr14).clip(0, 20),
        "volume": df["volume"],
    }, index=df.index)
    want = pd.to_datetime([b["t"] for b in bars], utc=True)
    out = out.reindex(want)
    model_tf = module == "momentum_expansion" and tf == "4h"
    meta = {}
    for k, (label, kind, group) in _TECH_META.items():
        meta[k] = {"label": label, "kind": kind, "group": group,
                   "model_input": bool(model_tf and k in _MOMENTUM_INPUTS)}
    note = ("Momentum's 4H feature formulas on 4H bars: the averages and ceilings its model reads as ATR distances."
            if model_tf else
            "Momentum's indicator formulas on these bars, for reference."
            + ("" if module == "momentum_expansion" else f" They are not inputs of {module}'s model."))
    return {"series": {k: _series(out[k]) for k in out.columns}, "meta": meta, "source": note}


_ENGINE_PRICE = ("session_vwap", "ema_9", "ema_20", "opening_range_high", "opening_range_low",
                 "prior_day_high", "prior_day_low", "micro_swing_high", "micro_swing_low")
_ENGINE_OSC = ("atr", "distance_to_vwap_atr", "distance_to_ema20_atr", "trend_strength",
               "relative_volume_1m", "volume_zscore", "range_expansion", "atr_contraction",
               "opening_range_position", "relative_strength_vs_spy")
_ENGINE_GROUPS = {"session_vwap": "session", "ema_9": "moving averages", "ema_20": "moving averages",
                  "opening_range_high": "session", "opening_range_low": "session",
                  "prior_day_high": "session", "prior_day_low": "session",
                  "micro_swing_high": "ceilings / floors", "micro_swing_low": "ceilings / floors"}
ENGINE_HISTORY_BARS = 780  # IntradayStructureEngine(max_history_bars=780)


def _to_engine_bars(symbol: str, rows: list[dict]):
    from strategies.intraday_structure.models import Bar
    out = []
    for r in rows:
        try:
            out.append(Bar(symbol=symbol, timestamp=r["t"], open=r["o"], high=r["h"], low=r["l"],
                           close=r["c"], volume=r.get("v") or 0.0))
        except (TypeError, ValueError):
            continue
    return out


def engine_history(ticker: str, session_day: pd.Timestamp) -> tuple[list, list, list, str | None]:
    """The engine's bar history for a session: that day plus the latest earlier
    archived session (its prior day), and SPY/QQQ for relative strength."""
    day = (session_day.tz_localize(ET) if session_day.tzinfo is None else session_day.tz_convert(ET)).normalize()
    files = []
    prior_used = None
    for back in range(1, 8):
        d = day - pd.Timedelta(days=back)
        p = BARS_1M_DIR / f"bars_{d.strftime('%Y%m%d')}.jsonl"
        if p.exists():
            files.append(p)
            prior_used = d.strftime("%Y-%m-%d")
            break
    today = BARS_1M_DIR / f"bars_{day.strftime('%Y%m%d')}.jsonl"
    if today.exists():
        files.append(today)
    syms = tuple(dict.fromkeys((ticker, "SPY", "QQQ")))
    rows: dict[str, list[dict]] = {s: [] for s in syms}
    for p in files:
        for s, bars in _scan_1m_day(str(p), _file_key(p), syms).items():
            rows[s] += bars
    return (_to_engine_bars(ticker, rows[ticker]), _to_engine_bars("SPY", rows["SPY"]),
            _to_engine_bars("QQQ", rows["QQQ"]), prior_used)


@lru_cache(maxsize=16)
def _engine_series_cached(ticker: str, day_iso: str, key: tuple) -> dict:
    from strategies.intraday_structure.features import compute_features
    hist, spy, qqq, prior = engine_history(ticker, pd.Timestamp(day_iso))
    out: dict[str, dict[str, float]] = {}
    times = [b.timestamp for b in hist]
    spy_t, qqq_t = [b.timestamp for b in spy], [b.timestamp for b in qqq]
    import bisect
    for i, b in enumerate(hist):
        local = b.timestamp.astimezone(ET)
        if local.strftime("%Y-%m-%d") != day_iso[:10] or not (9 * 60 + 30 <= local.hour * 60 + local.minute < 16 * 60):
            continue  # regular-hours bars of the session only
        window = hist[max(0, i - ENGINE_HISTORY_BARS + 1): i + 1]
        js, jq = bisect.bisect_right(spy_t, b.timestamp), bisect.bisect_right(qqq_t, b.timestamp)
        try:
            f = compute_features(window, spy_bars=spy[max(0, js - ENGINE_HISTORY_BARS):js],
                                 qqq_bars=qqq[max(0, jq - ENGINE_HISTORY_BARS):jq]).to_dict()
        except ValueError:
            continue
        out[b.timestamp.isoformat()] = {k: f.get(k) for k in (*_ENGINE_PRICE, *_ENGINE_OSC)}
    return {"by_time": out, "prior_session": prior, "n_history": len(times)}


def engine_series(ticker: str, bars: list[dict]) -> dict:
    """The intraday engine's own features at every regular-hours bar, aligned to ``bars``.

    Each value is ``compute_features`` on the engine's history as of that bar
    (at most 780 bars, the engine's own cap), so it is what the engine had at
    that bar's close.
    """
    if not bars:
        return {"series": {}, "meta": {}}
    days = sorted({pd.Timestamp(b["t"]).tz_convert(ET).strftime("%Y-%m-%d") for b in bars})
    by_time: dict[str, dict] = {}
    priors = []
    for d in days:
        p = BARS_1M_DIR / f"bars_{d.replace('-', '')}.jsonl"
        res = _engine_series_cached(ticker, d, _file_key(p))
        by_time.update(res["by_time"])
        priors.append(f"{d} <- {res['prior_session'] or 'none archived'}")
    def value(bar: dict, k: str) -> float:
        v = (by_time.get(pd.Timestamp(bar["t"]).isoformat()) or {}).get(k)
        return np.nan if v is None else v

    series = {k: _series([value(b, k) for b in bars]) for k in (*_ENGINE_PRICE, *_ENGINE_OSC)}
    meta = {k: {"label": k.replace("_", " "), "kind": "price" if k in _ENGINE_PRICE else "osc",
                "group": _ENGINE_GROUPS.get(k), "model_input": True} for k in series}
    return {"series": series, "meta": meta,
            "source": ("The engine's own compute_features at each regular-hours bar "
                       "(strategies/intraday_structure/features.py). Prior session used for prior-day levels: "
                       + "; ".join(priors) + ".")}


DECISION_CONTEXT_DIR = REPO / "Data/inference/intraday_structure/decision_context"
DEALER_ROOT = REPO / "Data/dealer_positioning"


@lru_cache(maxsize=16)
def _context_rows_cached(ticker: str, session: str, key: tuple) -> tuple[dict, ...]:
    from strategies.intraday_structure.decision_context import read_session

    return tuple(read_session(DECISION_CONTEXT_DIR, session, ticker=ticker))


def recorded_context(ticker: str, bar_open: pd.Timestamp, setup_id: str | None = None) -> dict | None:
    """The engine's own record of a decision on the bar that opened at ``bar_open``, if it made one.

    Records exist only for bars the engine decided on (confirm, refuse, entry,
    manage) and only from 2026-11-24. The position's own setup is preferred
    when several setups on the ticker decided on the same bar.
    """
    session = bar_open.tz_convert(ET).strftime("%Y-%m-%d")
    path = DECISION_CONTEXT_DIR / f"{session.replace('-', '')}.jsonl"
    rows = [r for r in _context_rows_cached(ticker, session, _file_key(path)) if _utc(r.get("bar_ts")) == bar_open]
    if not rows:
        return None
    mine = [r for r in rows if setup_id and r.get("setup_id") == setup_id]
    return (mine or rows)[-1]


@lru_cache(maxsize=1)
def _enricher():
    from strategies.intraday_structure.config import load_config
    from strategies.intraday_structure.decision_context import DecisionContextRecorder

    return DecisionContextRecorder(load_config().decision_context)


_ARCHIVED_LADDERS: dict[str, Any] = {}


def _dealer_context(ticker: str, decided_at: pd.Timestamp):
    """Dealer levels as of ``decided_at`` from the archived per-minute ladders (index ETFs etc.)."""
    from strategies.intraday_structure.config import load_config
    from strategies.intraday_structure.models import OptionsContext
    from strategies.intraday_structure.options import ArchivedLadderOptionsProvider

    if not (DEALER_ROOT / ticker / "snapshots").exists():
        return OptionsContext(timestamp=decided_at.to_pydatetime(), source="none", warnings=("no_archived_ladders",))
    provider = _ARCHIVED_LADDERS.get("p")
    if provider is None:
        provider = _ARCHIVED_LADDERS["p"] = ArchivedLadderOptionsProvider(
            DEALER_ROOT, max_age_minutes=load_config().levels.options_max_age_minutes)
    return provider.context(ticker, decided_at.to_pydatetime(), 0.0)


def _view_level(lv: dict) -> dict:
    """A recorded (compact) level in the shape the page draws."""
    out = {"price": lv.get("price"), "type": lv.get("type"), "strength": lv.get("strength"),
           "directionality": lv.get("dir"), "sources": str(lv.get("type") or "").split("+"),
           "distance_atr": lv.get("dist_atr")}
    if lv.get("size"):
        out.update({"low": lv.get("low"), "high": lv.get("high"), "size": lv.get("size")})
    return out


def _options_view(options: Any) -> dict:
    if isinstance(options, dict):
        keep = ("source", "timestamp", "gamma_regime", "gamma_flip", "call_wall", "put_wall",
                "local_gamma_exposure", "warnings", "structure")
        return {k: options.get(k) for k in keep}
    return {"source": options.source, "timestamp": _iso(options.timestamp), "gamma_regime": options.gamma_regime,
            "gamma_flip": options.gamma_flip, "call_wall": options.call_wall, "put_wall": options.put_wall,
            "local_gamma_exposure": options.local_gamma_exposure, "warnings": list(options.warnings),
            "structure": options.structure}


def engine_view(*, ticker: str, at: Any, setup: dict | None, direction: str | None) -> dict:
    """The engine's features, clustered level map, dealer positioning, regime and news at the bar opened at ``at``.

    ``recorded``: the engine's own decision record for that bar (exactly what it
    decided on). Otherwise ``rebuilt``: the engine's compute_features and
    StructuralLevelProvider on its archived history, with dealer levels rebuilt
    as of the bar's close from the archived ladders where they exist, and the
    regime and news as of the same moment. A rebuild cannot know setup state
    (pivots, candidate timing), so it is the map the engine would have drawn.
    """
    from strategies.intraday_structure.config import load_config
    from strategies.intraday_structure.decision_context import BAR_SPAN, compact_level
    from strategies.intraday_structure.features import compute_features
    from strategies.intraday_structure.levels import StructuralLevelProvider
    from strategies.intraday_structure.models import Candidate

    t = _utc(at)
    ticker = (ticker or "").strip().upper()
    if t is None or not _TICKER_RE.match(ticker):
        return {"error": "bad ticker or time"}
    decided_at = t + pd.Timedelta(BAR_SPAN)
    rec = recorded_context(ticker, t, (setup or {}).get("setup_id"))
    if rec is not None:
        feats = rec.get("features") or {}
        return {
            "source": "recorded", "kind": rec.get("kind"), "action": rec.get("action"),
            "setup_id": rec.get("setup_id"), "at": rec.get("bar_ts"), "decided_at": rec.get("decided_at"),
            "spot": (rec.get("bar") or {}).get("c"), "features": feats,
            "levels": [_view_level(x) for x in rec.get("levels") or []],
            "raw_level_count": len(rec.get("raw_levels") or []),
            "options": _options_view(rec.get("options") or {}),
            "intraday_regime": rec.get("intraday_regime"), "daily_regime": rec.get("daily_regime"),
            "news": rec.get("news"), "market_news": rec.get("market_news"),
            "instrument_class": rec.get("instrument_class"),
            "cluster_threshold": None, "prior_session": None,
            "note": f"Recorded by the engine when it decided ({rec.get('kind')}: {rec.get('action')}).",
        }
    hist, spy, qqq, prior = engine_history(ticker, t)
    hist = [b for b in hist if b.timestamp <= t.to_pydatetime()][-ENGINE_HISTORY_BARS:]
    if len(hist) < 2:
        return {"error": f"no 1m history for {ticker} at {t.isoformat()}"}
    spy = [b for b in spy if b.timestamp <= t.to_pydatetime()][-ENGINE_HISTORY_BARS:]
    qqq = [b for b in qqq if b.timestamp <= t.to_pydatetime()][-ENGINE_HISTORY_BARS:]
    feats = compute_features(hist, spy_bars=spy, qqq_bars=qqq).to_dict()
    cand_t = _utc((setup or {}).get("candidate_timestamp")) or hist[0].timestamp
    candidate = Candidate(ticker=ticker, timestamp=cand_t, direction=direction or (setup or {}).get("direction") or "long",
                          sources=tuple((setup or {}).get("candidate_sources") or ("replay",)))
    options = _dealer_context(ticker, decided_at)
    provider = StructuralLevelProvider(load_config().levels)
    levels = provider.levels(bars=hist, candidate=candidate, options=options, features=feats)
    spot = hist[-1].close
    atr = feats.get("atr") or 0.0
    try:
        extra = _enricher().enrich(ticker, decided_at.to_pydatetime())
    except Exception as exc:  # noqa: BLE001 - context is an annotation
        extra = {"error": str(exc)}
    dealer_note = ("Dealer levels rebuilt from the archived ladder as of the bar's close."
                   if options.source != "none" else
                   "No dealer levels: " + ", ".join(options.warnings or ("none archived",)) + ".")
    return {
        "source": "rebuilt", "at": hist[-1].timestamp.isoformat(), "decided_at": decided_at.isoformat(),
        "spot": spot, "prior_session": prior,
        "features": {k: _num(v) for k, v in feats.items()},
        "levels": [_view_level(compact_level(x, spot=spot, atr=atr)) for x in levels],
        "options": _options_view(options),
        "daily_regime": extra.get("daily_regime"), "news": extra.get("news"),
        "market_news": extra.get("market_news"), "instrument_class": extra.get("instrument_class"),
        "cluster_threshold": round(max(atr * provider.policy.cluster_atr, spot * provider.policy.cluster_pct), 4),
        "note": "Rebuilt, not recorded: the engine made no decision on this bar, or it predates recording "
                "(2026-11-24). " + dealer_note,
    }


_FEATURE_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,80}$")


def feature_series(module: str, ticker: str, start: Any, end: Any, name: str) -> dict:
    """One archived feature for ``ticker`` across the archive runs in [start, end]."""
    am = ARCHIVE_MODULES.get(module)
    lo, hi = _utc(start), _utc(end)
    ticker = (ticker or "").strip().upper()
    if not am or lo is None or hi is None or not _FEATURE_NAME_RE.match(name or "") or not _TICKER_RE.match(ticker):
        return {"error": "bad module, ticker, window or feature name"}
    import pyarrow.parquet as pq
    col = dsa.FEATURE_PREFIX + name
    rows = []
    d0 = (lo.tz_convert(ET) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    d1 = (hi.tz_convert(ET) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for path in dsa.list_partitions(module=am, start=d0, end=d1):
        try:
            if col not in pq.ParquetFile(path).schema_arrow.names:
                continue
            t = pq.read_table(path, columns=["run_id", "decision_ts", "bar_ts", "ticker", col],
                              filters=[("ticker", "==", ticker)]).to_pandas()
        except Exception as exc:  # noqa: BLE001 - one bad file must not hide the rest
            logger.warning("decision_replay: feature read failed %s: %s", path, exc)
            continue
        for _, r in t.iterrows():
            rows.append({"run_id": r["run_id"], "decision_ts": _iso(r["decision_ts"]),
                         "bar_ts": _iso(r["bar_ts"]), "value": _num(r[col])})
    return {"name": name, "rows": rows}


def archive_feature_names(module: str, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    am = ARCHIVE_MODULES.get(module)
    if not am:
        return []
    import pyarrow.parquet as pq
    parts = dsa.list_partitions(module=am, start=start.tz_convert(ET).strftime("%Y-%m-%d"),
                                end=end.tz_convert(ET).strftime("%Y-%m-%d"))
    if not parts:
        return []
    names = pq.ParquetFile(parts[-1]).schema_arrow.names
    return [c[len(dsa.FEATURE_PREFIX):] for c in names if c.startswith(dsa.FEATURE_PREFIX)]


# --------------------------------------------------------------------------- replay

def _position_window(pos: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    stamps = [_utc(pos.get(k)) for k in ("entry_bar", "entry_filled_at", "entry_submitted_at", "first_sell")]
    stamps = [s for s in stamps if s is not None]
    start = min(stamps) if stamps else pd.Timestamp.now(tz="UTC")
    end = _utc(pos.get("last_sell")) if pos.get("status") == "closed" else None
    return start, (end or pd.Timestamp.now(tz="UTC"))


def open_unsold_positions() -> list[dict]:
    """Positions a module still manages that have no sell leg yet.

    The ledgers hold sells only, so trade_library never sees these. They are
    built from the module's live ``managed`` state in the same shape, with no legs.
    """
    have = {(p["module"], p["ticker"]) for p in trade_library.build_positions() if p["status"] == "open"}
    out = []
    for (module, ticker), st in trade_library._open_book().items():
        if (module, ticker) in have:
            continue
        entry_bar = _iso(st.get("entry_bar"))
        gain = _num(st.get("unrealized_gain"))
        out.append({
            "id": f"open:{module}:{ticker}:{entry_bar}", "module": module, "ticker": ticker,
            "route": st.get("route") or "equity", "status": "open", "unsold": True,
            "entry_bar": entry_bar, "first_sell": None, "last_sell": None,
            "entry_price": _num(st.get("entry_avg_price")), "exit_price": _num(st.get("last_mark_price")),
            "realized": 0.0, "open_mark": _num(st.get("last_mark_price")),
            "open_ret_pct": round(gain * 100, 2) if gain is not None else None,
            "peak_gain": _num(st.get("peak_gain")), "trims": 0, "reasons": [], "final_reason": "",
            "hold_days": None, "u_entry": _num(st.get("u_entry")), "u_atr": _num(st.get("u_atr")),
            "entry_filled_at": _iso(st.get("entry_filled_at")),
            "entry_submitted_at": _iso(st.get("entry_submitted_at")),
            "entry_fill_price": _num(st.get("entry_fill_price")),
            "entry_anchor": {"u_entry": _num(st.get("u_entry")), "u_atr": _num(st.get("u_atr"))},
            "runs_held": st.get("runs_held"), "setup_id": None, "legs": [], "occ": st.get("occ"),
        })
    return out


def _find_position(position_id: str) -> dict | None:
    pool = open_unsold_positions() if position_id.startswith("open:") else trade_library.build_positions()
    return next((p for p in pool if p["id"] == position_id), None)


def replay(*, position_id: str | None = None, module: str | None = None, ticker: str | None = None,
           start: str | None = None, end: str | None = None, tf: str | None = None) -> dict[str, Any]:
    """Bars, decisions, orders, fills and levels for one position, or for any ticker and window."""
    pos = None
    if position_id:
        pos = _find_position(position_id)
        if pos is None:
            return {"error": f"no such position: {position_id}"}
        module, ticker = pos["module"], pos["ticker"]
        lo, hi = _position_window(pos)
    else:
        ticker = (ticker or "").strip().upper()
        if module not in KNOWN_MODULES:
            return {"error": f"unknown module: {module}"}
        if not _TICKER_RE.match(ticker):
            return {"error": f"bad ticker: {ticker!r}"}
        if not (_DATE_RE.match(start or "") and _DATE_RE.match(end or "")):
            return {"error": "start and end must be YYYY-MM-DD"}
        lo = pd.Timestamp(start, tz=ET).tz_convert("UTC")
        hi = (pd.Timestamp(end, tz=ET) + pd.Timedelta(days=1)).tz_convert("UTC") - pd.Timedelta(seconds=1)
        if hi <= lo:
            return {"error": "end is before start"}
    opts = timeframe_options(module)
    tf = tf if tf in opts else default_timeframe(module)
    if pos is None and (hi - lo).days > MAX_FREE_DAYS[tf]:
        return {"error": f"window too long for {tf} bars (max {MAX_FREE_DAYS[tf]} days)"}

    if tf == "1m":
        # Whole ET sessions, so the setup's path earlier in the day is on the chart.
        load_lo = lo.tz_convert(ET).normalize().tz_convert("UTC")
        load_hi = (hi.tz_convert(ET).normalize() + pd.Timedelta(days=1)).tz_convert("UTC") - pd.Timedelta(seconds=1)
    elif pos is None:
        load_lo, load_hi = lo, hi  # an ad-hoc window is exactly what was asked for
    else:
        before, after = LOAD_DAYS[tf]
        load_lo, load_hi = lo - pd.Timedelta(days=before), hi + pd.Timedelta(days=after)
    bars, bar_error = load_bars(ticker, tf, load_lo, load_hi)
    times = np.asarray([pd.Timestamp(b["t"]).value for b in bars], dtype="int64")
    if pos is not None and len(times) and tf != "1m":
        # Trim to the position's own bars plus fixed context on each side.
        pad_b, pad_a = PAD_BARS[tf]
        i0 = max(0, int(np.searchsorted(times, lo.value, side="left")) - pad_b)
        i1 = min(len(times), int(np.searchsorted(times, hi.value, side="right")) + pad_a)
        bars, times = bars[i0:i1], times[i0:i1]
    span_ns = int(TIMEFRAME_H[tf] * 3600 * 1e9)
    notes: list[str] = []
    out: dict[str, Any] = {
        "subject": {"module": module, "ticker": ticker, "tf": tf, "tf_options": opts,
                    "archive_module": ARCHIVE_MODULES.get(module),
                    "route": (pos or {}).get("route"), "mode": "position" if pos else "free",
                    "start": lo.isoformat(), "end": hi.isoformat()},
        "position": pos, "bars": bars, "error": bar_error, "notes": notes,
    }
    if not bars:
        out.update({"decisions": [], "events": [], "levels": [], "coverage": {}})
        return out
    w0 = pd.Timestamp(int(times[0]), tz="UTC")
    w1 = pd.Timestamp(int(times[-1]), tz="UTC") + pd.Timedelta(hours=TIMEFRAME_H[tf])

    # Decisions: archive first (full vectors), then audit runs the archive does not hold.
    decisions, warnings = archive_decisions(module, ticker, w0, w1)
    archived_ids = {d["run_id"] for d in decisions if d.get("run_id")}
    audit_dec, audit_orders, audit_runs = audit_records(module, ticker, w0, w1, archived_ids)
    decisions += audit_dec
    if module == "intraday_structure":
        decisions += transition_decisions(ticker, w0, w1, (pos or {}).get("setup_id"))
    notes += warnings
    if module not in ARCHIVE_MODULES and module != "intraday_structure":
        notes.append(f"{module} has no decision source.")

    def frame_of(ts: Any) -> int | None:
        return place(times, ts, span_ns)

    # Audit runs have no wall clock: place each where its scored bar ended, and a
    # later run re-scoring the same bar on the first bar to open after that.
    decided_after = pd.Timedelta(hours=DECISION_SPAN_H.get(module, 4.0))
    audit_frame: dict[str, int | None] = {}
    seen_bars: set[str] = set()
    for r in audit_runs:
        ended = _utc(r["bar_ts"]) + decided_after
        if r["bar_ts"] in seen_bars:
            j = int(np.searchsorted(times, ended.value, side="right"))
            audit_frame[r["run_key"]] = j if j < len(times) else None
        else:
            audit_frame[r["run_key"]] = frame_of(ended)
        seen_bars.add(r["bar_ts"])

    placed, off_chart = [], 0
    for d in decisions:
        d["scored_frame"] = frame_of(d.get("bar_ts"))
        d["frame"] = (audit_frame.get(d["run_key"]) if d["source"] == "audit"
                      else frame_of(d.get("decision_ts") or d.get("bar_ts")))
        if d["frame"] is None:
            off_chart += 1
            continue
        placed.append(d)
    placed.sort(key=lambda d: (d["frame"], d.get("decision_ts") or d.get("bar_ts") or ""))

    events: list[dict] = []
    for o in audit_orders:
        events.append({"kind": "order_plan", "frame": audit_frame.get(o["run_key"]), "t": o["bar_ts"],
                       "label": f"{o.get('side') or '?'} planned: {o.get('reason') or ''}".strip(),
                       "detail": o})
    entry_frame = exit_frame = None
    entry_decision = None
    contract = setup = None
    levels: list[dict] = []
    ref: dict[str, Any] = {"price": None, "basis": None}
    policy = None
    if pos is not None:
        fill_t = _utc(pos.get("entry_filled_at"))
        approx = None
        if fill_t is None and pos.get("entry_submitted_at"):
            fill_t, approx = _utc(pos["entry_submitted_at"]), "submit time (no fill time recorded)"
        if fill_t is not None:
            entry_frame = frame_of(fill_t)
        elif pos.get("entry_bar"):
            # No fill time: assume the fill came right after the scored entry bar ended.
            entry_frame = frame_of(_utc(pos["entry_bar"]) + pd.Timedelta(hours=DECISION_SPAN_H.get(module, 4.0)))
            approx = "the end of the scored entry bar (no fill time recorded)"
        if entry_frame is not None:
            price = _num(pos.get("entry_fill_price")) or _num(pos.get("entry_price"))
            events.append({"kind": "entry", "frame": entry_frame, "t": _iso(fill_t) or pos.get("entry_bar"),
                           "label": f"entry fill ({pos.get('route')})", "price": price,
                           "on_chart_scale": pos.get("route") == "equity", "approx": approx})
        for leg in pos.get("legs") or []:
            f = frame_of(leg["ts"])
            events.append({"kind": "trim" if leg["is_trim"] else "exit", "frame": f, "t": leg["ts"],
                           "label": leg["reason"], "price": leg["price"], "qty": leg["qty"],
                           "pnl": leg["pnl"], "decision_gain": leg.get("decision_gain"),
                           "fill_gain": leg.get("fill_gain"), "submitted_at": leg.get("submitted_at"),
                           "on_chart_scale": pos.get("route") == "equity"})
            if not leg["is_trim"] and f is not None:
                exit_frame = f
        levels, ref, policy, lnotes = position_levels(pos)
        notes += lnotes
        contract = contract_info(pos)
        if module == "intraday_structure":
            setup = setup_record(pos.get("setup_id"), pos.get("entry_filled_at") or pos.get("entry_submitted_at"))
            if setup:
                levels += setup_levels(setup)
        stop_at = exit_frame if exit_frame is not None else len(bars) - 1
        for lv in levels:
            lv["from_frame"], lv["to_frame"] = entry_frame, stop_at
        # The entry decision: the selected row for the scored entry bar, latest
        # decision made no later than the fill.
        eb = _utc(pos.get("entry_bar"))
        cands = [d for d in placed
                 if (d.get("selected") or d.get("in_targets") or d.get("to_state") == "CONFIRMED")
                 and (eb is None or _utc(d.get("bar_ts")) == eb)
                 and (fill_t is None or (_utc(d.get("decision_ts") or d.get("bar_ts")) or fill_t) <= fill_t)]
        if module == "intraday_structure" and pos.get("setup_id"):
            cands = [d for d in cands if d.get("position_setup")] or cands
        if cands:
            entry_decision = cands[-1]["run_key"]
        if ref["price"] is None and entry_frame is not None:
            ref = {"price": bars[entry_frame]["c"], "basis": "close of the entry-fill bar (approximation)"}

    ent = next((d for d in placed if d["run_key"] == entry_decision), None)
    cov = archive_coverage(module)
    try:
        overlays = engine_series(ticker, bars) if tf == "1m" else technical_series(module, ticker, tf, bars)
    except Exception as exc:  # noqa: BLE001 - overlays are an annotation; the replay stands without them
        logger.exception("decision_replay: overlays failed for %s %s", ticker, tf)
        overlays = {"series": {}, "meta": {}, "error": str(exc)}
    out.update({
        "contract": contract, "setup": setup, "overlays": overlays,
        "archive_features": archive_feature_names(module, w0, w1),
        "decisions": placed,
        "events": sorted(events, key=lambda e: (e["frame"] if e["frame"] is not None else 10**9, e["t"] or "")),
        "levels": levels, "ref": ref, "policy": policy,
        "entry_frame": entry_frame, "exit_frame": exit_frame,
        "entry_decision": entry_decision,
        "entry_run_id": ent.get("run_id") if ent and ent["source"] == "archive" else None,
        "span_s": int(TIMEFRAME_H[tf] * 3600),
        "coverage": {
            "archive": {**cov, "rows_in_window": sum(d["source"] == "archive" for d in placed)},
            "audit": {"available": module in AUDIT_MODULES, "runs_in_window": len(audit_runs),
                      "rows_in_window": sum(d["source"] == "audit" for d in placed)},
            "transitions": sum(d["source"] == "transitions" for d in placed),
            "off_chart": off_chart,
        },
    })
    if not placed:
        sources = []
        if cov["module"]:
            sources.append(f"decision archive ({cov['first']} to {cov['last']})" if cov["first"]
                           else "decision archive (no runs yet)")
        if module in AUDIT_MODULES:
            sources.append("signal audit (picks and order plans only)")
        if module == "intraday_structure":
            sources.append("setup transitions")
        notes.append("No decision for this ticker in this window. Sources checked: "
                     + (", ".join(sources) or "none") + ".")
    return out


def positions(q: dict[str, str]) -> dict[str, Any]:
    """Positions for the picker, each flagged with its evidence.

    Open positions with no sell yet come first, then the ledger's positions,
    newest activity first.
    """
    rows = open_unsold_positions() + trade_library.build_positions()
    modules = sorted({r["module"] for r in rows})
    module, ticker = q.get("module", ""), q.get("ticker", "").upper()
    status, outcome = q.get("status", ""), q.get("outcome", "")
    if module:
        rows = [r for r in rows if r["module"] == module]
    if ticker:
        rows = [r for r in rows if ticker in r["ticker"]]
    if status:
        rows = [r for r in rows if r["status"] == status]
    if outcome == "win":
        rows = [r for r in rows if r["realized"] > 0]
    elif outcome == "loss":
        rows = [r for r in rows if r["realized"] < 0]
    cov = {m: archive_coverage(m) for m in {r["module"] for r in rows}}
    out = []
    for r in rows:
        c = cov[r["module"]]
        last = (r.get("last_sell") or "")[:10] if r["status"] == "closed" else "9999-12-31"
        first = (r.get("entry_bar") or r.get("entry_filled_at") or r.get("first_sell") or "")[:10]
        archived = bool(c["first"]) and last >= c["first"] and first <= (c["last"] or "")
        evidence = ("archive" if archived else
                    "audit" if r["module"] in AUDIT_MODULES else
                    "transitions" if r["module"] == "intraday_structure" else "fills only")
        contract = None
        if r.get("route") == "option":
            occ = next((parse_occ(s) for s in [r.get("occ"), *[leg.get("symbol") for leg in r.get("legs") or []]]
                        if parse_occ(s)), None)
            if occ:
                entry_t = _utc(r.get("entry_filled_at") or r.get("entry_submitted_at") or r.get("entry_bar"))
                if entry_t is not None:
                    occ["dte_at_entry"] = (pd.Timestamp(occ["expiry"]).date() - entry_t.tz_convert(ET).date()).days
                contract = occ
        out.append({k: r.get(k) for k in ("id", "module", "ticker", "route", "status", "entry_bar",
                                          "last_sell", "realized", "final_reason", "hold_days", "trims")}
                   | {"evidence": evidence, "contract": contract})
    return {"positions": out, "modules": modules}


def state() -> dict[str, Any]:
    """Hub status. Cheap: a memoized ledger fold plus directory listings."""
    rows = trade_library.build_positions()
    return {"positions": len(rows),
            "archive": {m: archive_coverage(m)["last"] for m in ARCHIVE_MODULES},
            "generated_at": datetime.now(timezone.utc).isoformat()}
