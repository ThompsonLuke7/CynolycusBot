from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from strategies.intraday_structure.models import Candidate, Direction, utc_datetime

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


def manual_candidates(symbols: Iterable[str], *, timestamp: datetime | None = None) -> list[Candidate]:
    now = timestamp or datetime.now(timezone.utc)
    return [
        Candidate(ticker=symbol, timestamp=now, direction=Direction.LONG, sources=("manual_watchlist",), score=0.5)
        for symbol in symbols if str(symbol).strip()
    ]


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
