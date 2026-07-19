from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from strategies.intraday_structure.models import Candidate, Direction, utc_datetime


DEFAULT_AUDIT_SOURCES = {
    "meta_ranker": Path("Data/inference/meta_ranker/live_signal_audit.jsonl"),
    "momentum_expansion": Path("Data/inference/momentum_expansion/live_signal_audit.jsonl"),
    "4h_swing": Path("Data/inference/multi_ticker_swing_htf/live_signal_audit.jsonl"),
    "dealer_ranker": Path("Data/inference/dealer_ranker/live_signal_audit.jsonl"),
}


class AuditCandidateFeed:
    """Read only the latest signal decisions from existing append-only audits."""

    def __init__(self, sources: dict[str, Path] | None = None, swing_audit_root: str | Path = "UI/swing_audit") -> None:
        self.sources = sources or DEFAULT_AUDIT_SOURCES
        self.swing_audit_root = Path(swing_audit_root)
        self._seen: set[tuple[str, str, str, str]] = set()

    def poll(self) -> list[Candidate]:
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
