from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from signals.news.ticker_relevance import AMBIGUOUS_TICKERS, classify as classify_ticker_relevance
from strategies.intraday_structure.config import CatalystDiscoveryPolicy, OpeningDiscoveryPolicy
from strategies.intraday_structure.models import Bar, Candidate, Direction, utc_datetime

_ET = ZoneInfo("America/New_York")


DEFAULT_AUDIT_SOURCES = {
    "meta_ranker": Path("Data/inference/meta_ranker/live_signal_audit.jsonl"),
    "momentum_expansion": Path("Data/inference/momentum_expansion/live_signal_audit.jsonl"),
    "4h_swing": Path("Data/inference/multi_ticker_swing_htf/live_signal_audit.jsonl"),
    "dealer_ranker": Path("Data/inference/dealer_ranker/live_signal_audit.jsonl"),
}


class AuditCandidateFeed:
    """Read only the latest signal decisions from existing append-only audits.

    ``poll()`` is called from the runner's tight bar-consumer loop (once per
    bar pulled off the shared stream queue, i.e. many times per second during
    RTH), but each call does a tail-read of every audit source plus a glob +
    stat of the swing-audit directory. The 4H-family audits only gain a new
    ``signal_decision`` event every ~4 hours and the swing audit only a few
    times per minute, so re-scanning disk on every single bar is pure waste
    that can make this feed the bottleneck behind the consumer (observed
    2026-07-20: the shared bar queue backed up and dropped ~27% of bars for
    the whole session once Intraday Structure was enabled). Throttle to a
    floor interval between real scans; between scans, return no candidates
    (matches the steady-state "nothing new" result these polls almost always
    return anyway).
    """

    def __init__(
        self,
        sources: dict[str, Path] | None = None,
        swing_audit_root: str | Path = "UI/swing_audit",
        *,
        min_poll_interval_seconds: float = 5.0,
    ) -> None:
        self.sources = sources or DEFAULT_AUDIT_SOURCES
        self.swing_audit_root = Path(swing_audit_root)
        self._seen: set[tuple[str, str, str, str]] = set()
        self.min_poll_interval_seconds = max(0.0, float(min_poll_interval_seconds))
        self._last_poll_monotonic: float | None = None

    def poll(self) -> list[Candidate]:
        now_monotonic = time.monotonic()
        if (
            self._last_poll_monotonic is not None
            and now_monotonic - self._last_poll_monotonic < self.min_poll_interval_seconds
        ):
            return []
        self._last_poll_monotonic = now_monotonic
        candidates: list[Candidate] = []
        for source, path in self.sources.items():
            event = _latest_json_event(path, event_names={"signal_decision"})
            if event:
                candidates.extend(_ranker_candidates(source, event))
        swing_path = _latest_swing_audit(self.swing_audit_root)
        if swing_path:
            event = _latest_json_event(swing_path, event_names={"signal"}, event_key="type")
            if event:
                candidate = _swing_candidate(event)
                if candidate:
                    candidates.append(candidate)
        fresh: list[Candidate] = []
        for candidate in candidates:
            signature = (candidate.ticker, candidate.direction.value, candidate.timestamp.isoformat(), "+".join(candidate.sources))
            if signature in self._seen:
                continue
            self._seen.add(signature)
            fresh.append(candidate)
        return fresh


class DealerRankingCandidateFeed:
    """Seeds a bounded, fresh dealer-map watchlist for intraday confirmation.

    It intentionally keeps structural-potential and change-intensity ranks
    separate: a name such as MU can have a changing map worth watching even
    when it is not a raw top-ten structural score.  Candidate creation is not
    an entry; the one-minute engine must still confirm price structure.
    """

    def __init__(
        self,
        ranking_path: str | Path,
        *,
        top_structural: int = 40,
        top_change: int = 40,
        max_age_hours: float = 30.0,
    ) -> None:
        self.ranking_path = Path(ranking_path)
        self.top_structural = max(1, int(top_structural))
        self.top_change = max(1, int(top_change))
        self.max_age = timedelta(hours=max(0.0, float(max_age_hours)))
        self._seen: set[tuple[str, str, str]] = set()
        self._last_signature: tuple[int, int] | None = None

    def poll(self, *, now: datetime | None = None) -> list[Candidate]:
        if not self.ranking_path.exists():
            return []
        try:
            stat = self.ranking_path.stat()
        except OSError:
            return []
        # The ranking parquet only refreshes once/day near the close, but this
        # feed is polled on every bar the runner pulls off the shared stream
        # (many times per second during RTH). Re-parsing an unchanged parquet
        # that often is wasted I/O that can starve the consumer loop and back
        # up the shared bar queue (see AuditCandidateFeed docstring for the
        # 2026-07-20 incident this mirrors); skip straight to [] when nothing
        # on disk has changed, matching LiquidityCandidateFeed's pattern.
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
        if signature == self._last_signature:
            return []
        self._last_signature = signature
        try:
            frame = pd.read_parquet(self.ranking_path)
        except (OSError, ValueError, ImportError):
            return []
        required = {"symbol", "captured_at", "dealer_swing_rank", "dealer_change_intensity_rank"}
        if not required.issubset(frame.columns):
            return []
        frame = frame.copy()
        frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True, errors="coerce")
        frame["dealer_swing_rank"] = pd.to_numeric(frame["dealer_swing_rank"], errors="coerce")
        frame["dealer_change_intensity_rank"] = pd.to_numeric(frame["dealer_change_intensity_rank"], errors="coerce")
        frame = frame.dropna(subset=["captured_at", "dealer_swing_rank", "dealer_change_intensity_rank"])
        if frame.empty:
            return []
        captured = frame["captured_at"].max().to_pydatetime()
        reference = now or datetime.now(timezone.utc)
        age = utc_datetime(reference) - utc_datetime(captured)
        if age < timedelta(0) or (self.max_age and age > self.max_age):
            return []
        latest = frame[frame["captured_at"] == pd.Timestamp(captured)].copy()
        selected = latest[
            (latest["dealer_swing_rank"] <= self.top_structural)
            | (latest["dealer_change_intensity_rank"] <= self.top_change)
        ].copy()
        out: list[Candidate] = []
        for row in selected.itertuples(index=False):
            raw = row._asdict()
            symbol = str(raw.get("symbol") or "").upper()
            if not symbol:
                continue
            structural_rank = float(raw["dealer_swing_rank"])
            change_rank = float(raw["dealer_change_intensity_rank"])
            structural_score = max(0.0, 1.0 - (structural_rank - 1.0) / self.top_structural)
            change_score = max(0.0, 1.0 - (change_rank - 1.0) / self.top_change)
            direction_label = str(raw.get("dealer_change_direction") or raw.get("dealer_direction") or "").lower()
            direction = Direction.SHORT if direction_label == "bearish" else Direction.LONG
            signature = (symbol, direction.value, captured.isoformat())
            if signature in self._seen:
                continue
            self._seen.add(signature)
            out.append(Candidate(
                ticker=symbol, timestamp=captured, available_at=captured, direction=direction,
                sources=("dealer_level_map",), score=max(structural_score, change_score),
                average_dollar_volume=raw.get("avg_dollar_volume_20d"),
                metadata={
                    "dealer_captured_at": captured.isoformat(),
                    "dealer_swing_rank": int(structural_rank),
                    "dealer_change_intensity_rank": int(change_rank),
                    "dealer_direction": raw.get("dealer_direction"),
                    "dealer_change_direction": raw.get("dealer_change_direction"),
                    "dealer_swing_potential_score": _as_optional_float(raw.get("dealer_swing_potential_score")),
                    "dealer_change_intensity_score": _as_optional_float(raw.get("dealer_change_intensity_score")),
                },
            ))
        return out


def load_liquidity_universe(
    universe_path: str | Path,
    *,
    top_n: int,
) -> list[tuple[str, float]]:
    """Return the highest-ADV eligible symbols from the shared live universe.

    The CSV is maintained outside this module; invalid or incomplete rows are
    excluded instead of being silently treated as liquid.  The return order is
    deterministic, so the same source file always yields the same watchlist.
    """
    path = Path(universe_path)
    if not path.exists() or top_n <= 0:
        return []
    try:
        frame = pd.read_csv(path, usecols=lambda name: name in {"ticker", "is_eligible", "avg_dollar_volume_20d"})
    except (OSError, ValueError, pd.errors.ParserError):
        return []
    required = {"ticker", "avg_dollar_volume_20d"}
    if not required.issubset(frame.columns):
        return []
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["avg_dollar_volume_20d"] = pd.to_numeric(frame["avg_dollar_volume_20d"], errors="coerce")
    if "is_eligible" in frame.columns:
        eligible = frame["is_eligible"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        frame = frame[eligible]
    frame = frame[(frame["ticker"] != "") & (frame["avg_dollar_volume_20d"] > 0)]
    frame = frame.sort_values(["avg_dollar_volume_20d", "ticker"], ascending=[False, True])
    frame = frame.drop_duplicates("ticker", keep="first").head(max(1, int(top_n)))
    return [(str(row.ticker), float(row.avg_dollar_volume_20d)) for row in frame.itertuples(index=False)]


class LiquidityCandidateFeed:
    """Seeds the configured highest-liquidity universe once per ET session.

    A new seed is emitted after a restart, a shared-universe update, or a new
    session so the engine's candidate TTL cannot leave a long-running server
    with an empty broad watchlist.
    """

    def __init__(self, universe_path: str | Path, *, top_n: int = 50) -> None:
        self.universe_path = Path(universe_path)
        self.top_n = max(1, int(top_n))
        self._last_signature: tuple[int, int] | None = None
        self._last_session_date = None

    def symbols(self) -> list[str]:
        return [symbol for symbol, _adv in load_liquidity_universe(self.universe_path, top_n=self.top_n)]

    def poll(self, *, now: datetime | None = None) -> list[Candidate]:
        reference = utc_datetime(now or datetime.now(timezone.utc))
        try:
            stat = self.universe_path.stat()
            signature = (int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            return []
        session_date = reference.astimezone(_ET).date()
        if signature == self._last_signature and session_date == self._last_session_date:
            return []
        rows = load_liquidity_universe(self.universe_path, top_n=self.top_n)
        self._last_signature = signature
        self._last_session_date = session_date
        return [
            Candidate(
                ticker=symbol,
                timestamp=reference,
                available_at=reference,
                direction=Direction.LONG,
                sources=("high_liquidity_universe",),
                score=0.5,
                average_dollar_volume=adv,
                metadata={
                    "liquidity_universe_path": str(self.universe_path),
                    "liquidity_universe_top_n": self.top_n,
                    "avg_dollar_volume_20d": adv,
                },
            )
            for symbol, adv in rows
        ]


@dataclass(frozen=True)
class DiscoveryBaseline:
    ticker: str
    prior_close: float
    average_dollar_volume: float


def load_discovery_baselines(
    universe_path: str | Path,
    *,
    top_n: int | None,
) -> dict[str, DiscoveryBaseline]:
    """Load the causal-at-startup price/ADV baseline used by discovery.

    ``shared_universe.csv`` is rebuilt by the guarded data jobs.  Its file mtime
    is recorded on every emitted candidate; no current-session bar is joined
    backwards into this baseline.
    """
    path = Path(universe_path)
    if not path.exists() or (top_n is not None and top_n <= 0):
        return {}
    try:
        frame = pd.read_csv(
            path,
            usecols=lambda name: name in {"ticker", "is_eligible", "px", "avg_dollar_volume_20d"},
        )
    except (OSError, ValueError, pd.errors.ParserError):
        return {}
    required = {"ticker", "px", "avg_dollar_volume_20d"}
    if not required.issubset(frame.columns):
        return {}
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["px"] = pd.to_numeric(frame["px"], errors="coerce")
    frame["avg_dollar_volume_20d"] = pd.to_numeric(frame["avg_dollar_volume_20d"], errors="coerce")
    if "is_eligible" in frame.columns:
        eligible = frame["is_eligible"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        frame = frame[eligible]
    frame = frame[
        (frame["ticker"] != "")
        & (frame["px"] > 0)
        & (frame["avg_dollar_volume_20d"] > 0)
    ]
    frame = frame.sort_values(
        ["avg_dollar_volume_20d", "ticker"], ascending=[False, True]
    ).drop_duplicates("ticker", keep="first")
    if top_n is not None:
        frame = frame.head(int(top_n))
    return {
        str(row.ticker): DiscoveryBaseline(
            ticker=str(row.ticker),
            prior_close=float(row.px),
            average_dollar_volume=float(row.avg_dollar_volume_20d),
        )
        for row in frame.itertuples(index=False)
    }


@dataclass
class _OpeningTape:
    session_high: float = -math.inf
    session_low: float = math.inf
    session_dollar_volume: float = 0.0
    rth_open: float | None = None
    closes: list[float] = field(default_factory=list)


class OpeningMomentumCandidateFeed:
    """Promote opening gap/volume/acceleration leaders into structure.

    The feed observes only bars delivered by the existing shared stream and is
    intentionally rules-first.  It creates a watch candidate, not a trade.  All
    calculations use the bar at or before the decision and the startup baseline.
    """

    def __init__(self, policy: OpeningDiscoveryPolicy) -> None:
        self.policy = policy
        self.baselines = load_discovery_baselines(policy.universe_path, top_n=policy.top_n)
        try:
            self._baseline_mtime = Path(policy.universe_path).stat().st_mtime
        except OSError:
            self._baseline_mtime = None
        self._session_date = None
        self._tapes: dict[str, _OpeningTape] = {}
        self._bars: dict[str, list[Bar]] = {}
        self._emitted: set[tuple[str, Direction]] = set()

    def symbols(self) -> list[str]:
        return list(self.baselines)

    def observe(self, bar: Bar, *, observed_at: datetime | None = None) -> list[Candidate]:
        baseline = self.baselines.get(bar.symbol)
        if baseline is None:
            return []
        et = bar.timestamp.astimezone(_ET)
        minute = et.hour * 60 + et.minute
        if et.weekday() >= 5:
            return []
        if et.date() != self._session_date:
            self._session_date = et.date()
            self._tapes.clear()
            self._bars.clear()
            self._emitted.clear()
        # Retain a bounded, already-arrived warmup for any later opening,
        # catalyst or ranker promotion. The engine receives only bars strictly
        # before the signal bar, so this cannot include the decision bar twice.
        if 4 * 60 <= minute < 16 * 60:
            history = self._bars.setdefault(bar.symbol, [])
            if history and history[-1].timestamp == bar.timestamp:
                history[-1] = bar
            elif not history or history[-1].timestamp < bar.timestamp:
                history.append(bar)
                if len(history) > 120:
                    del history[:-120]
        start = self.policy.scan_start_hour_et * 60
        end = self.policy.scan_end_hour_et * 60
        if minute < start or minute >= end:
            return []

        tape = self._tapes.setdefault(bar.symbol, _OpeningTape())
        tape.session_high = max(tape.session_high, bar.high)
        tape.session_low = min(tape.session_low, bar.low)
        tape.session_dollar_volume += max(0.0, bar.volume) * bar.close
        if minute >= 9 * 60 + 30 and tape.rth_open is None:
            tape.rth_open = bar.open
        tape.closes.append(bar.close)
        if len(tape.closes) > 8:
            del tape.closes[:-8]

        current_return = bar.close / baseline.prior_close - 1.0
        if abs(current_return) < 1e-12:
            return []
        direction = Direction.LONG if current_return > 0 else Direction.SHORT
        if (bar.symbol, direction) in self._emitted:
            return []

        gap = bar.open / baseline.prior_close - 1.0
        rth_return = (
            bar.close / tape.rth_open - 1.0
            if tape.rth_open is not None and tape.rth_open > 0
            else 0.0
        )
        three_bar_return = (
            tape.closes[-1] / tape.closes[-4] - 1.0 if len(tape.closes) >= 4 else 0.0
        )
        previous_three_bar_return = (
            tape.closes[-4] / tape.closes[-7] - 1.0 if len(tape.closes) >= 7 else 0.0
        )
        acceleration = three_bar_return - previous_three_bar_return
        session_range = (tape.session_high - tape.session_low) / baseline.prior_close
        expected_fraction = _expected_volume_fraction(minute)
        actual_fraction = tape.session_dollar_volume / baseline.average_dollar_volume
        volume_pace = actual_fraction / expected_fraction if expected_fraction > 0 else 0.0
        market_return = self._market_return()
        relative_strength = current_return - market_return
        sign = 1.0 if direction == Direction.LONG else -1.0

        gap_leader = (
            sign * gap >= self.policy.min_gap_pct
            and sign * current_return >= self.policy.min_gap_pct * 0.75
            and volume_pace >= self.policy.min_volume_pace
        )
        acceleration_leader = (
            minute >= 9 * 60 + 30
            and sign * rth_return >= self.policy.min_rth_return_pct
            and sign * three_bar_return >= self.policy.min_three_bar_return_pct
            and volume_pace >= self.policy.min_volume_pace
            and (
                session_range >= self.policy.min_session_range_pct
                or sign * relative_strength >= self.policy.min_relative_strength_pct
            )
        )
        if not (gap_leader or acceleration_leader):
            return []
        if len(self._emitted) >= self.policy.max_candidates_per_session:
            return []

        score = _opening_score(
            gap=gap,
            rth_return=rth_return,
            three_bar_return=three_bar_return,
            session_range=session_range,
            volume_pace=volume_pace,
            relative_strength=relative_strength,
        )
        available_at = utc_datetime(observed_at or datetime.now(timezone.utc))
        self._emitted.add((bar.symbol, direction))
        trigger = "gap_volume_leader" if gap_leader else "rth_acceleration_leader"
        return [Candidate(
            ticker=bar.symbol,
            timestamp=bar.timestamp,
            available_at=available_at,
            direction=direction,
            sources=("opening_momentum",),
            score=score,
            average_dollar_volume=baseline.average_dollar_volume,
            metadata={
                "discovery_trigger": trigger,
                "prior_close": baseline.prior_close,
                "gap_pct": gap,
                "return_from_prior_close": current_return,
                "rth_return_pct": rth_return,
                "three_bar_return_pct": three_bar_return,
                "acceleration_pct": acceleration,
                "session_range_pct": session_range,
                "session_dollar_volume_fraction": actual_fraction,
                "volume_pace": volume_pace,
                "market_return_pct": market_return,
                "relative_strength_pct": relative_strength,
                "baseline_path": str(self.policy.universe_path),
                "baseline_file_mtime": self._baseline_mtime,
                "market_data_lag_seconds": max(0.0, (available_at - bar.timestamp).total_seconds()),
            },
        )]

    def history(self, symbol: str, *, before: datetime | None = None) -> tuple[Bar, ...]:
        bars = self._bars.get(str(symbol).strip().upper(), [])
        if before is None:
            return tuple(bars)
        cutoff = utc_datetime(before)
        return tuple(bar for bar in bars if bar.timestamp < cutoff)

    def _market_return(self) -> float:
        values: list[float] = []
        for symbol in ("SPY", "QQQ"):
            baseline = self.baselines.get(symbol)
            tape = self._tapes.get(symbol)
            if baseline is not None and tape is not None and tape.closes:
                values.append(tape.closes[-1] / baseline.prior_close - 1.0)
        return sum(values) / len(values) if values else 0.0


_MATERIAL_CATALYST = re.compile(
    r"\b(earnings?.*(?:beat|miss|result)|beats? estimates|raises? (?:guidance|outlook)|"
    r"cuts? (?:guidance|outlook)|fda|pdufa|phase [123]|trial (?:result|update)|approval|"
    r"contract|partnership|acquisition|merger|tender offer|price target|upgrade|downgrade|"
    r"offering|registered direct|buyback|repurchase)\b",
    re.IGNORECASE,
)
_MATERIAL_FAMILIES = {
    "earnings_guidance", "earnings_result", "biotech_fda", "contract_partnership",
    "analyst_action", "ma_proxy_tender", "financing_dilution", "activist_ownership",
}
_AMBIGUOUS_COMPANY_TERMS = {"NOW": ("servicenow",)}


class CatalystCandidateFeed:
    """Promote only fresh, material, relevant records from the live ledger."""

    def __init__(
        self,
        policy: CatalystDiscoveryPolicy,
        *,
        allowed_symbols: Iterable[str] | None = None,
    ) -> None:
        self.policy = policy
        self.path = Path(policy.ledger_path)
        self.allowed_symbols = (
            {str(symbol).strip().upper() for symbol in allowed_symbols if str(symbol).strip()}
            if allowed_symbols is not None else None
        )
        self.baselines = load_discovery_baselines(policy.universe_path, top_n=None)
        self._last_poll_monotonic: float | None = None
        self._last_signature: tuple[int, int] | None = None
        self._seen: set[str] = set()
        self.last_decisions: list[dict] = []

    def poll(self, *, now: datetime | None = None) -> list[Candidate]:
        self.last_decisions = []
        monotonic = time.monotonic()
        if (
            self._last_poll_monotonic is not None
            and monotonic - self._last_poll_monotonic < self.policy.refresh_seconds
        ):
            return []
        self._last_poll_monotonic = monotonic
        try:
            stat = self.path.stat()
            signature = (int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            return []
        if signature == self._last_signature:
            return []
        self._last_signature = signature
        try:
            frame = pd.read_parquet(self.path)
        except (OSError, ValueError, ImportError):
            return []
        required = {"ticker", "timestamp", "headline", "catalyst_score"}
        if not required.issubset(frame.columns):
            return []
        frame = frame.copy()
        frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["scored_at"] = pd.to_datetime(
            frame.get("scored_at", frame["timestamp"]), utc=True, errors="coerce"
        )
        frame["catalyst_score"] = pd.to_numeric(frame["catalyst_score"], errors="coerce")
        frame = frame.dropna(subset=["ticker", "timestamp", "scored_at", "catalyst_score"])
        reference = utc_datetime(now or datetime.now(timezone.utc))
        cutoff = pd.Timestamp(reference - timedelta(minutes=self.policy.max_age_minutes))
        frame["available_at"] = frame[["timestamp", "scored_at"]].max(axis=1)
        frame = frame[(frame["available_at"] >= cutoff) & (frame["available_at"] <= pd.Timestamp(reference))]
        frame = frame.sort_values(["available_at", "catalyst_score"], ascending=[True, False])

        out: list[Candidate] = []
        for row in frame.itertuples(index=False):
            raw = row._asdict()
            identity = str(raw.get("content_hash") or raw.get("record_id") or f"{raw['ticker']}:{raw['timestamp']}:{raw['headline']}")
            if identity in self._seen:
                continue
            self._seen.add(identity)
            ticker = str(raw["ticker"])
            reason = self._rejection_reason(raw)
            if reason is not None:
                self.last_decisions.append({
                    "event_type": "catalyst_rejected",
                    "ticker": ticker,
                    "event_time": raw["timestamp"].isoformat(),
                    "available_at": raw["available_at"].isoformat(),
                    "reason": reason,
                    "content_hash": identity,
                    "headline": str(raw.get("headline") or ""),
                })
                continue
            score = float(raw["catalyst_score"])
            direction = Direction.LONG if score >= self.policy.min_bullish_score else Direction.SHORT
            baseline = self.baselines[ticker]
            candidate_score = max(score, 1.0 - score)
            candidate = Candidate(
                ticker=ticker,
                timestamp=raw["timestamp"].to_pydatetime(),
                available_at=raw["available_at"].to_pydatetime(),
                direction=direction,
                sources=("validated_catalyst",),
                score=candidate_score,
                average_dollar_volume=baseline.average_dollar_volume,
                metadata={
                    "content_hash": identity,
                    "record_id": _clean_scalar(raw.get("record_id")),
                    "headline": str(raw.get("headline") or ""),
                    "news_source": _clean_scalar(raw.get("source")),
                    "catalyst_family": _clean_scalar(raw.get("catalyst_family")),
                    "catalyst_subtype": _clean_scalar(raw.get("catalyst_subtype")),
                    "catalyst_score": score,
                    "information_direction": _clean_scalar(raw.get("information_direction")),
                    "scored_at": raw["scored_at"].isoformat(),
                    "relevance_policy": "strict_material_subject",
                },
            )
            out.append(candidate)
            self.last_decisions.append({
                "event_type": "catalyst_promoted",
                "ticker": ticker,
                "event_time": candidate.timestamp.isoformat(),
                "available_at": candidate.available_at.isoformat(),
                "content_hash": identity,
                "direction": direction.value,
                "score": candidate_score,
            })
        out.sort(key=lambda candidate: candidate.score, reverse=True)
        selected = out[: self.policy.max_candidates_per_poll]
        selected_hashes = {str(candidate.metadata.get("content_hash")) for candidate in selected}
        for decision in self.last_decisions:
            if (
                decision.get("event_type") == "catalyst_promoted"
                and str(decision.get("content_hash")) not in selected_hashes
            ):
                decision["event_type"] = "catalyst_rejected"
                decision["reason"] = "per_poll_candidate_cap"
        return selected

    def _rejection_reason(self, raw: dict) -> str | None:
        ticker = str(raw["ticker"])
        headline = str(raw.get("headline") or "")
        if self.allowed_symbols is not None and ticker not in self.allowed_symbols:
            return "symbol_not_in_streamed_discovery_universe"
        baseline = self.baselines.get(ticker)
        if baseline is None:
            return "missing_eligible_liquidity_baseline"
        relevant, relevance_reason = classify_ticker_relevance(
            ticker,
            headline,
            strict_foreign_symbols=self.policy.strict_ticker_relevance,
        )
        if not relevant:
            return f"ticker_relevance:{relevance_reason}"
        if ticker in AMBIGUOUS_TICKERS and not _ambiguous_subject_is_explicit(ticker, headline):
            return "ambiguous_ticker_subject_not_explicit"
        information_direction = str(raw.get("information_direction") or "")
        if information_direction == "price_recap":
            return "backward_looking_price_recap"
        family = str(raw.get("catalyst_family") or "")
        if family not in _MATERIAL_FAMILIES and not _MATERIAL_CATALYST.search(headline):
            return "not_material_event_type"
        score = float(raw["catalyst_score"])
        if self.policy.max_bearish_score < score < self.policy.min_bullish_score:
            return "catalyst_score_not_directional"
        return None


def _ambiguous_subject_is_explicit(ticker: str, headline: str) -> bool:
    text = str(headline or "")
    if re.search(rf"\(\s*(?:(?:NYSE|NASDAQ|AMEX)\s*:\s*)?{re.escape(ticker)}\s*\)", text, re.IGNORECASE):
        return True
    lowered = text.lower()
    return any(term in lowered for term in _AMBIGUOUS_COMPANY_TERMS.get(ticker, ()))


def _expected_volume_fraction(minute_et: int) -> float:
    """Conservative progress curve used only to identify an unusual pace."""
    open_minute = 9 * 60 + 30
    if minute_et < open_minute:
        premarket_progress = max(0.0, (minute_et - 4 * 60) / (5.5 * 60))
        return max(0.001, 0.01 * premarket_progress)
    rth_progress = min(1.0, max(0.0, (minute_et - open_minute + 1) / 390.0))
    return max(0.02, 0.02 + 0.98 * rth_progress)


def _opening_score(**metrics: float) -> float:
    score = 0.35
    score += 0.20 * min(1.0, abs(metrics["gap"]) / 0.10)
    score += 0.15 * min(1.0, abs(metrics["rth_return"]) / 0.03)
    score += 0.10 * min(1.0, abs(metrics["three_bar_return"]) / 0.015)
    score += 0.10 * min(1.0, metrics["session_range"] / 0.05)
    score += 0.10 * min(1.0, metrics["volume_pace"] / 3.0)
    score += 0.05 * min(1.0, abs(metrics["relative_strength"]) / 0.03)
    return max(0.0, min(1.0, score))


def _clean_scalar(value):
    if value is None:
        return None
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def manual_candidates(
    symbols: Iterable[str],
    *,
    timestamp: datetime | None = None,
    directions: Iterable[Direction] = (Direction.LONG, Direction.SHORT),
) -> list[Candidate]:
    """Seed a hand-picked symbol on both sides by default.

    Long-only was the original behaviour and it quietly excluded every
    rejection setup: price turning DOWN at resistance is a short, so a
    long-only watchlist can never produce the setup type a call-wall study
    is about.
    """
    now = timestamp or datetime.now(timezone.utc)
    return [
        Candidate(ticker=symbol, timestamp=now, direction=direction,
                  sources=("manual_watchlist",), score=0.5)
        for symbol in symbols if str(symbol).strip()
        for direction in directions
    ]


class ManualCandidateFeed:
    """Re-seed the manual watchlist once per ET session.

    The runner used to register these exactly once per process, behind a
    ``_manual_registered`` flag. Combined with the 1,440-minute candidate TTL
    that meant a hand-picked symbol expired after a day and never came back, so
    on any server running longer than that the watchlist was empty in practice.
    Mirrors ``LiquidityCandidateFeed``'s session cadence.
    """

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        directions: Iterable[Direction] = (Direction.LONG, Direction.SHORT),
    ) -> None:
        self.symbols = tuple(str(s).strip().upper() for s in symbols if str(s).strip())
        self.directions = tuple(directions)
        self._last_session_date = None

    def poll(self, *, now: datetime | None = None) -> list[Candidate]:
        if not self.symbols:
            return []
        reference = utc_datetime(now or datetime.now(timezone.utc))
        session_date = reference.astimezone(_ET).date()
        if session_date == self._last_session_date:
            return []
        self._last_session_date = session_date
        return manual_candidates(self.symbols, timestamp=reference, directions=self.directions)


def _ranker_candidates(source: str, event: dict) -> list[Candidate]:
    timestamp_raw = event.get("bar") or event.get("timestamp")
    if not timestamp_raw:
        return []
    audits = event.get("signal_audits") or {}
    targets = event.get("targets") or ([event.get("ticker")] if event.get("ticker") else [])
    result: list[Candidate] = []
    observed_at = datetime.now(timezone.utc)
    for ticker in targets:
        if not ticker:
            continue
        audit = audits.get(ticker, {}) if isinstance(audits, dict) else {}
        side = str(audit.get("side") or "long").lower()
        direction = Direction.SHORT if side in {"short", "put", "sell", "-1"} else Direction.LONG
        score = float(audit.get("score") or audit.get("rank_pct") or event.get("score") or 0.5)
        extra = dict(audit.get("extra") or {})
        result.append(Candidate(
            ticker=str(ticker), timestamp=utc_datetime(timestamp_raw), direction=direction,
            sources=(source,), score=max(0.0, min(1.0, score)),
            average_dollar_volume=extra.get("average_dollar_volume"), available_at=observed_at,
            metadata={"signal_audit": audit, "signal_bar": str(timestamp_raw)},
        ))
    return result


def _swing_candidate(event: dict) -> Candidate | None:
    payload = event.get("payload") or {}
    ticker = payload.get("ticker")
    if not ticker:
        return None
    direction = Direction.LONG if float(payload.get("direction", 1)) >= 0 else Direction.SHORT
    pivot = payload.get("ref_high") if direction == Direction.LONG else payload.get("ref_low")
    return Candidate(
        ticker=str(ticker), timestamp=utc_datetime(event.get("ts")), direction=direction,
        sources=("30m_swing",), score=float(payload.get("p_dir", 0.5)),
        pivot=float(pivot) if pivot is not None else None,
        available_at=utc_datetime(event.get("ts")), metadata={"swing_signal": payload},
    )


def _latest_json_event(path: Path, *, event_names: set[str], event_key: str = "event") -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    for line in reversed(_tail_lines(path)):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get(event_key)) in event_names:
            return row
    return None


def _tail_lines(path: Path, max_bytes: int = 2_000_000) -> list[str]:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    if size > max_bytes:
        data = data.split(b"\n", 1)[-1]
    return data.decode("utf-8", errors="replace").splitlines()


def _latest_swing_audit(root: Path) -> Path | None:
    candidates = sorted(root.glob("swing_session_*.jsonl"))
    candidates.extend(sorted((root / "paper").glob("swing_session_*.jsonl")))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _as_optional_float(value) -> float | None:
    try:
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None
