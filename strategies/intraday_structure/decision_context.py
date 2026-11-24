"""What the intraday structure engine had in hand at each decision, recorded as it decides.

The engine has always computed a full picture on every bar (session, liquidity,
volume-profile, round-number and dealer levels, the dealer snapshot, the tape
features and an intraday regime label) and then thrown it away. Only a summary
survived, in the abstention and closed-setup ledgers. That left nothing to train
a model on, and nothing for the replay to show except levels rebuilt after the
fact without the dealer snapshot.

One JSON line per decision goes to ``<root>/<YYYYMMDD>.jsonl`` (ET session of the
bar):

  confirm  a detector confirmed and every gate passed (the entry decision)
  refuse   a confirmation was declined (throttled: a new reason, or every N bars)
  entry    the bar the position was opened on
  manage   every bar of an open position: hold, or the exit it took

The engine builds the record from its own DetectionContext (``build_decision_context``,
pure, no I/O). The runner-side ``DecisionContextRecorder`` adds what the engine does
not read: the daily market-regime row and the news, both as of the decision time,
and the instrument class. Nothing here changes a decision.

Timing: ``bar_ts`` is the bar's open (how bars are stamped) and ``decided_at`` its
close, the moment the engine decided. Every enrichment is restricted to data
available at ``decided_at``.
"""
from __future__ import annotations

import logging
import math
import time
from bisect import bisect_right
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from strategies.intraday_structure.state_store import append_jsonl

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
SCHEMA_VERSION = "intraday_decision_context_v1"
BAR_SPAN = timedelta(minutes=1)
#: Broad-index ETFs (and index symbols). Every other ETF in the universe is "etf".
INDEX_ETFS = frozenset({"SPY", "QQQ", "IWM", "DIA", "SPX", "NDX", "RUT", "XSP", "QQQM", "IVV", "VOO", "RSP", "MDY"})
_REGIME_FIELDS = (
    "risk_appetite_z", "liquidity_stress_z", "credit_risk_z", "breadth_z", "sector_dispersion_z",
    "spy_rv20_z", "spy_trend_state", "risk_appetite_z_stale_days", "breadth_z_stale_days",
    "credit_risk_z_stale_days", "spy_rv20_z_stale_days",
)


def instrument_class(ticker: str, etfs: frozenset[str] = frozenset()) -> str:
    t = str(ticker).upper()
    if t in INDEX_ETFS:
        return "index_etf"
    return "etf" if t in etfs else "single_stock"


def clean(obj: Any) -> Any:
    """JSON-safe: non-finite floats -> None, floats to 7 significant digits, datetimes -> ISO, tuples -> lists."""
    if isinstance(obj, float):
        # 7 significant digits: 4 decimals on a $650 price, and a third off the file size.
        return float(f"{obj:.7g}") if math.isfinite(obj) else None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return clean(asdict(obj))
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "item"):
        return clean(obj.item())  # numpy scalars
    return obj


def compact_level(level: Any, *, spot: float, atr: float) -> dict[str, Any]:
    meta = level.metadata or {}
    out = {
        "price": round(float(level.price), 4), "type": level.level_type, "strength": round(float(level.strength), 3),
        "dir": level.directionality, "dist_atr": round((float(level.price) - spot) / atr, 3) if atr else None,
    }
    for key, attr in (("touch", "touch_count"), ("reject", "rejection_count")):
        if getattr(level, attr, 0):
            out[key] = int(getattr(level, attr))
    if level.break_status != "unbroken":
        out["break"] = level.break_status
    if level.hold_status != "untested":
        out["hold"] = level.hold_status
    if "low" in meta and int(meta.get("cluster_size", 1)) > 1:  # a zone: several levels merged
        out["low"], out["high"] = round(float(meta["low"]), 4), round(float(meta["high"]), 4)
        out["size"] = int(meta["cluster_size"])
    return out


def build_decision_context(
    *, kind: str, action: str, setup: Any, ctx: Any, engine_version: str,
    regime: Any = None, plan: Any = None, outcome: Any = None, include_raw: bool = True,
) -> dict[str, Any]:
    """One decision record from the engine's own context. Pure: no I/O."""
    bar = ctx.bar
    spot = float(bar.close)
    atr = float(ctx.features.get("atr")) or 0.0
    options = ctx.options
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "engine_version": engine_version,
        "kind": kind, "action": action,
        "bar_ts": bar.timestamp.isoformat(), "decided_at": (bar.timestamp + BAR_SPAN).isoformat(),
        "ticker": setup.ticker, "setup_id": setup.setup_id, "setup_type": setup.setup_type.value,
        "direction": setup.direction.value, "state": setup.state.value, "phase": setup.phase,
        "candidate_sources": list(setup.candidate.sources), "candidate_score": setup.candidate.score,
        "bar": {"o": bar.open, "h": bar.high, "l": bar.low, "c": bar.close, "v": bar.volume},
        "setup": {
            "pivot": setup.pivot, "invalidation": setup.invalidation, "targets": list(setup.targets),
            "active_target": setup.active_target, "entry_price": setup.entry_price,
            "entry_time": setup.entry_time, "bars_alive": setup.bars_alive, "bars_in_state": setup.bars_in_state,
            "confidence": setup.confidence, "runway_score": setup.runway_score,
            "expected_reward_risk": setup.expected_reward_risk, "mfe": setup.max_favorable_excursion,
            "mae": setup.max_adverse_excursion, "extensions": setup.extensions,
        },
        "features": ctx.features.to_dict(),
        "market": ctx.market.to_dict(),
        "intraday_regime": regime.to_dict() if regime is not None else None,
        "options": {
            "source": options.source, "timestamp": options.timestamp, "gamma_regime": options.gamma_regime,
            "gamma_flip": options.gamma_flip, "call_wall": options.call_wall, "put_wall": options.put_wall,
            "local_gamma_exposure": options.local_gamma_exposure, "dealer_imbalance": options.dealer_imbalance,
            "dealer_strength_score": options.dealer_strength_score,
            "strike_congestion_score": options.strike_congestion_score,
            "live_call_flow_acceleration": options.live_call_flow_acceleration,
            "live_put_flow_acceleration": options.live_put_flow_acceleration,
            "net_delta_premium": options.net_delta_premium, "short_dated_flow_ratio": options.short_dated_flow_ratio,
            "warnings": list(options.warnings), "structure": options.structure,
        },
        "levels": [compact_level(x, spot=spot, atr=atr) for x in ctx.levels],
    }
    if include_raw:
        record["raw_levels"] = [compact_level(x, spot=spot, atr=atr) for x in getattr(ctx, "raw_levels", ())]
    if plan is not None:
        record["plan"] = {
            "invalidation": plan.invalidation, "targets": list(plan.targets), "reward_risk": plan.reward_risk,
            "runway_score": plan.runway.runway_score, "runway_components": plan.runway.components,
            "target_level_type": plan.runway.target_level_type,
            "target_level_sources": list(plan.runway.target_level_sources),
            "intermediate_obstacles": list(plan.runway.intermediate_obstacles),
        }
    if outcome is not None:
        record["refusal"] = {
            "proposed_invalidation": outcome.proposed_invalidation,
            "proposed_risk_atr": outcome.proposed_risk_atr,
            "max_invalidation_atr": outcome.max_invalidation_atr,
        }
    if kind == "confirm" and setup.metadata.get("dealer_plate"):
        record["dealer_plate"] = setup.metadata["dealer_plate"]
    return record


class DecisionContextRecorder:
    """Engine context sink: adds regime, news and instrument class, then appends the line.

    Every source is cached (the regime table per file change, the news ledger at
    most every ``news_refresh_seconds``), so a record costs dictionary lookups.
    A failure is logged and swallowed: recording must never stop the engine.
    """

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.root = Path(policy.root)
        self._regime: tuple[tuple, pd.DataFrame | None] = ((), None)
        self._news_sig: tuple | None = None
        self._news_checked = 0.0
        self._news: dict[str, tuple[list[datetime], list[dict]]] = {}
        self._market_news_cache: tuple[str, list[dict]] | None = None
        self._qqq_sma: dict[date, dict | None] = {}
        self._etfs = self._load_etfs(policy.universe_path)

    # -- sink -------------------------------------------------------------------
    def enrich(self, ticker: str, decided_at: datetime) -> dict[str, Any]:
        """Instrument class, daily regime and news for ``ticker`` as of ``decided_at``."""
        return clean({
            "instrument_class": instrument_class(ticker, self._etfs),
            "daily_regime": self._daily_regime(decided_at),
            "news": self._ticker_news(ticker, decided_at),
            "market_news": self._market_news(decided_at),
        })

    def __call__(self, record: dict[str, Any]) -> None:
        try:
            record.update(self.enrich(record["ticker"], datetime.fromisoformat(record["decided_at"])))
            session = datetime.fromisoformat(record["bar_ts"]).astimezone(ET).strftime("%Y%m%d")
            append_jsonl(self.root / f"{session}.jsonl", clean(record))
        except Exception:  # noqa: BLE001 - the record is evidence, not a dependency
            logger.exception("intraday decision context write failed for %s", record.get("setup_id"))

    # -- enrichment -------------------------------------------------------------
    @staticmethod
    def _load_etfs(path: str) -> frozenset[str]:
        try:
            u = pd.read_csv(path, usecols=lambda c: c in {"ticker", "asset_type"})
            return frozenset(u.loc[u["asset_type"].astype(str).str.lower() == "etf", "ticker"].astype(str).str.upper())
        except Exception:  # noqa: BLE001
            return frozenset()

    @staticmethod
    def _signature(path: Path) -> tuple:
        try:
            st = path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return ()

    def _daily_regime(self, decided_at: datetime) -> dict | None:
        """The market-regime row published before ``decided_at`` (rows land after each close)."""
        path = Path(self.policy.regime_path)
        sig = self._signature(path)
        if sig != self._regime[0]:
            try:
                frame = pd.read_parquet(path)
                frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
                self._regime = (sig, frame.sort_values("available_at").reset_index(drop=True))
            except Exception:  # noqa: BLE001
                self._regime = (sig, None)
        frame = self._regime[1]
        if frame is None or frame.empty:
            return None
        i = int(frame["available_at"].searchsorted(pd.Timestamp(decided_at), side="right")) - 1
        if i < 0:
            return None
        row = frame.iloc[i]
        out = {k: row.get(k) for k in _REGIME_FIELDS if k in frame.columns}
        out["date"] = pd.Timestamp(row["date"]).date().isoformat()
        out["available_at"] = row["available_at"].isoformat()
        # The five market-table votes of research/regime_switch_2026-11-20/05 (BEAR needs 4 of 7 there;
        # its two QQQ-trend votes are not in this table, see qqq_below_sma50 below).
        votes = [row.get("spy_trend_state") == -1, (row.get("breadth_z") or 0) < 0,
                 (row.get("risk_appetite_z") or 0) < 0, (row.get("credit_risk_z") or 0) < 0,
                 (row.get("spy_rv20_z") or 0) > 1]
        out["bear_votes_of_5"] = int(sum(bool(v) for v in votes))
        out.update(self._qqq_trend(decided_at.astimezone(ET).date()))
        return out

    def _qqq_trend(self, session: date) -> dict:
        """QQQ's prior close against its 50-day average, from the 1d cache (bars before ``session``)."""
        if session not in self._qqq_sma:
            try:
                b = pd.read_parquet(Path(self.policy.daily_bars_dir) / "QQQ.parquet", columns=["timestamp", "close"])
                days = pd.to_datetime(b["timestamp"], utc=True).dt.tz_convert(ET).dt.date
                close = b.loc[(days < session).to_numpy(), "close"].astype(float)
                self._qqq_sma[session] = ({"qqq_prior_close": float(close.iloc[-1]),
                                           "qqq_sma50": float(close.tail(50).mean()),
                                           "qqq_below_sma50": bool(close.iloc[-1] < close.tail(50).mean())}
                                          if len(close) >= 50 else None)
            except Exception:  # noqa: BLE001
                self._qqq_sma[session] = None
        return self._qqq_sma[session] or {}

    def _load_news(self) -> None:
        now = time.monotonic()
        if now - self._news_checked < float(self.policy.news_refresh_seconds) and self._news_sig is not None:
            return
        self._news_checked = now
        path = Path(self.policy.news_path)
        sig = self._signature(path)
        if sig == self._news_sig:
            return
        self._news_sig = sig
        try:
            f = pd.read_parquet(path, columns=["ticker", "timestamp", "scored_at", "headline", "catalyst_family",
                                                "catalyst_subtype", "catalyst_score", "information_direction"])
        except Exception:  # noqa: BLE001
            self._news = {}
            return
        f["scored_at"] = pd.to_datetime(f["scored_at"], utc=True)
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        # Available once BOTH published and scored.
        f["available_at"] = f[["scored_at", "timestamp"]].max(axis=1)
        f = f.sort_values("available_at")
        news: dict[str, tuple[list[datetime], list[dict]]] = {}
        for ticker, g in f.groupby(f["ticker"].astype(str).str.upper()):
            rows = g.to_dict("records")
            news[ticker] = ([r["available_at"].to_pydatetime() for r in rows], rows)
        self._news = news

    def _window(self, ticker: str, decided_at: datetime, hours: float) -> list[dict]:
        stamps, rows = self._news.get(ticker, ([], []))
        hi = bisect_right(stamps, decided_at)
        lo = bisect_right(stamps, decided_at - timedelta(hours=hours))
        return rows[lo:hi]

    @staticmethod
    def _item(r: dict, decided_at: datetime, width: int) -> dict:
        return {"ticker": r["ticker"], "score": round(float(r["catalyst_score"]), 3),
                "family": r["catalyst_family"], "subtype": r["catalyst_subtype"],
                "direction": r["information_direction"],
                "age_min": round((decided_at - r["available_at"].to_pydatetime()).total_seconds() / 60, 1),
                "headline": str(r["headline"] or "")[:width]}

    def _ticker_news(self, ticker: str, decided_at: datetime) -> dict:
        self._load_news()
        rows = self._window(ticker.upper(), decided_at, float(self.policy.news_lookback_hours))
        if not rows:
            return {"count": 0}
        scores = [float(r["catalyst_score"]) for r in rows]
        return {"count": len(rows), "max_score": max(scores), "min_score": min(scores),
                "latest": self._item(rows[-1], decided_at, 160)}

    def _market_news(self, decided_at: datetime) -> list[dict]:
        """Strongest recent catalysts on the names that move the indices (one computation per minute)."""
        key = decided_at.strftime("%Y%m%d%H%M")
        if self._market_news_cache and self._market_news_cache[0] == key:
            return self._market_news_cache[1]
        self._load_news()
        rows = [r for t in self.policy.market_news_tickers
                for r in self._window(t.upper(), decided_at, float(self.policy.market_news_lookback_hours))]
        rows.sort(key=lambda r: abs(float(r["catalyst_score"]) - 0.5), reverse=True)
        out = [self._item(r, decided_at, 120) for r in rows[:3]]
        self._market_news_cache = (key, out)
        return out


def read_session(root: str | Path, session: date | str, *, ticker: str | None = None) -> list[dict]:
    """All records of one ET session (``YYYY-MM-DD`` or date), optionally for one ticker."""
    day = session.strftime("%Y%m%d") if isinstance(session, date) else str(session).replace("-", "")
    path = Path(root) / f"{day}.jsonl"
    if not path.exists():
        return []
    import json

    needle = None if ticker is None else (f'"ticker":"{ticker.upper()}"')
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if needle is not None and needle not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            # The needle also matches news items that name the ticker.
            if ticker is None or row.get("ticker") == ticker.upper():
                out.append(row)
    return out


def nearest_before(records: Sequence[dict], at: datetime) -> dict | None:
    """The latest record decided at or before ``at`` (records in file order)."""
    best = None
    for r in records:
        t = datetime.fromisoformat(r["decided_at"])
        if t <= at and (best is None or t >= datetime.fromisoformat(best["decided_at"])):
            best = r
    return best


__all__ = [
    "BAR_SPAN", "DecisionContextRecorder", "INDEX_ETFS", "SCHEMA_VERSION", "build_decision_context",
    "clean", "compact_level", "instrument_class", "nearest_before", "read_session",
]
