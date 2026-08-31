"""One append-only decision funnel plus candidate-level opportunity outcomes.

The existing transition, abstention and closed-setup ledgers remain canonical for
their individual jobs.  This file gives analysis one ordered stream spanning
candidate discovery, capacity decisions, setup decisions and modelled outcomes.
It is evidence only: no method can submit or modify an order.
"""

from __future__ import annotations

import threading
import uuid
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from strategies.intraday_structure.models import Bar, Candidate, Direction, utc_datetime
from strategies.intraday_structure.state_store import append_jsonl


EVIDENCE_SCHEMA_VERSION = "intraday_structure_decision_event_v1"


class DecisionEventLedger:
    """Thread-safe JSONL writer for the complete paper decision funnel."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(
        self,
        event_type: str,
        *,
        event_time: datetime | str | None,
        ticker: str | None,
        payload: dict[str, Any],
        available_at: datetime | str | None = None,
    ) -> None:
        recorded_at = datetime.now(timezone.utc)
        row = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            "event_type": str(event_type),
            "event_time": _iso(event_time),
            "available_at": _iso(available_at),
            "recorded_at": recorded_at.isoformat(),
            "ticker": str(ticker).strip().upper() if ticker else None,
            "payload": payload,
        }
        with self._lock:
            append_jsonl(self.path, row)

    def candidate_event(self, row: dict[str, Any]) -> None:
        candidate = dict(row.get("candidate") or {})
        self.write(
            str(row.get("event_type") or "candidate_event"),
            event_time=candidate.get("timestamp"),
            available_at=candidate.get("available_at"),
            ticker=candidate.get("ticker"),
            payload=row,
        )

    def transition(self, record: Any) -> None:
        payload = record.to_dict()
        self.write(
            "setup_transition",
            event_time=payload.get("timestamp"),
            ticker=payload.get("ticker"),
            payload=payload,
        )

    def abstention(self, record: Any) -> None:
        payload = record.to_dict()
        self.write(
            "setup_abstention",
            event_time=payload.get("timestamp"),
            ticker=payload.get("ticker"),
            payload=payload,
        )

    def closed_setup(self, record: Any) -> None:
        payload = record.to_dict()
        self.write(
            "modeled_setup_close",
            event_time=payload.get("exit_time"),
            ticker=payload.get("ticker"),
            payload=payload,
        )

    def feed_decision(self, row: dict[str, Any]) -> None:
        payload = dict(row)
        self.write(
            str(payload.pop("event_type", "feed_decision")),
            event_time=payload.pop("event_time", None),
            available_at=payload.pop("available_at", None),
            ticker=payload.pop("ticker", None),
            payload=payload,
        )


def composite_sink(*sinks: Callable[[Any], None] | None) -> Callable[[Any], None]:
    active = tuple(sink for sink in sinks if sink is not None)

    def emit(record: Any) -> None:
        for sink in active:
            sink(record)

    return emit


@dataclass
class _CandidateEpisode:
    candidate: Candidate
    entry_time: datetime | None = None
    entry_price: float | None = None
    last_close: float | None = None
    bars_observed: int = 0
    mfe_return: float = 0.0
    mae_return: float = 0.0
    emitted_horizons: set[int] = field(default_factory=set)


class CandidateOutcomeTracker:
    """Measure every accepted candidate even when no setup confirms.

    Entry is the next observed bar's open, never the signal bar that created the
    candidate. Outcomes are model/event-time measurements, not broker fills;
    the candidate's event-vs-availability lag is carried on every row.
    """

    def __init__(
        self,
        ledger: DecisionEventLedger,
        *,
        horizons_minutes: Iterable[int] = (5, 15, 30, 60),
    ) -> None:
        horizons = sorted({int(value) for value in horizons_minutes if int(value) > 0})
        if not horizons:
            raise ValueError("candidate outcome tracker needs at least one positive horizon")
        self.ledger = ledger
        self.horizons = tuple(horizons)
        self._episodes: dict[str, _CandidateEpisode] = {}

    def track(self, candidate: Candidate) -> None:
        self._episodes.setdefault(candidate.candidate_id, _CandidateEpisode(candidate=candidate))

    def on_bar(self, bar: Bar) -> None:
        completed: list[str] = []
        for candidate_id, episode in tuple(self._episodes.items()):
            candidate = episode.candidate
            causal_start = max(candidate.timestamp, candidate.available_at or candidate.timestamp)
            if candidate.ticker != bar.symbol or bar.timestamp <= causal_start:
                continue
            if episode.entry_price is None:
                episode.entry_time = bar.timestamp
                episode.entry_price = bar.open
            entry = episode.entry_price
            entry_time = episode.entry_time
            if entry is None or entry_time is None:
                continue
            episode.bars_observed += 1
            episode.last_close = bar.close
            if candidate.direction == Direction.LONG:
                favorable = bar.high / entry - 1.0
                adverse = bar.low / entry - 1.0
                current_return = bar.close / entry - 1.0
            else:
                favorable = entry / bar.low - 1.0
                adverse = entry / bar.high - 1.0
                current_return = entry / bar.close - 1.0
            episode.mfe_return = max(episode.mfe_return, favorable)
            episode.mae_return = min(episode.mae_return, adverse)
            elapsed = max(0.0, (bar.timestamp - entry_time).total_seconds() / 60.0)
            for horizon in self.horizons:
                if horizon in episode.emitted_horizons or elapsed < horizon:
                    continue
                episode.emitted_horizons.add(horizon)
                availability_lag = max(
                    0.0,
                    ((candidate.available_at or candidate.timestamp) - candidate.timestamp).total_seconds(),
                )
                self.ledger.write(
                    "candidate_fixed_horizon_outcome",
                    event_time=bar.timestamp,
                    available_at=datetime.now(timezone.utc),
                    ticker=candidate.ticker,
                    payload={
                        "candidate_id": candidate_id,
                        "candidate_timestamp": candidate.timestamp.isoformat(),
                        "candidate_available_at": (candidate.available_at or candidate.timestamp).isoformat(),
                        "candidate_sources": list(candidate.sources),
                        "candidate_score": candidate.score,
                        "direction": candidate.direction.value,
                        "entry_time": entry_time.isoformat(),
                        "entry_price": entry,
                        "horizon_minutes": horizon,
                        "evaluated_at": bar.timestamp.isoformat(),
                        "evaluation_delay_minutes": elapsed - horizon,
                        "last_close": episode.last_close,
                        "signed_return": current_return,
                        "mfe_return": episode.mfe_return,
                        "mae_return": episode.mae_return,
                        "bars_observed": episode.bars_observed,
                        "model_event_time_only": True,
                        "candidate_availability_lag_seconds": availability_lag,
                    },
                )
            if len(episode.emitted_horizons) == len(self.horizons):
                completed.append(candidate_id)
        for candidate_id in completed:
            self._episodes.pop(candidate_id, None)

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_candidate_outcomes": len(self._episodes),
            "horizons_minutes": list(self.horizons),
        }


def _iso(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return utc_datetime(value).isoformat()
    return str(value)


def read_decision_events(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_decision_events(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Produce the daily-review facts directly from the unified event stream."""
    events = list(rows)
    event_counts = Counter(str(row.get("event_type") or "unknown") for row in events)
    registration_sources: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    outcome_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    confirmed = 0
    for row in events:
        event_type = str(row.get("event_type") or "")
        payload = row.get("payload") or {}
        if event_type == "candidate_registered":
            candidate = payload.get("candidate") or {}
            for source in candidate.get("sources") or ("unknown",):
                registration_sources[str(source)] += 1
        elif event_type in {"candidate_rejected", "catalyst_rejected"}:
            rejection_reasons[str(payload.get("reason") or "unknown")] += 1
        elif event_type == "setup_transition" and str(payload.get("to_state")) == "CONFIRMED":
            confirmed += 1
        elif event_type == "candidate_fixed_horizon_outcome":
            horizon = int(payload.get("horizon_minutes") or 0)
            sources = payload.get("candidate_sources") or ("unknown",)
            for source in sources:
                outcome_groups[(str(source), horizon)].append(payload)

    outcomes: list[dict[str, Any]] = []
    for (source, horizon), group in sorted(outcome_groups.items()):
        returns = [_finite(row.get("signed_return")) for row in group]
        returns = [value for value in returns if value is not None]
        mfes = [_finite(row.get("mfe_return")) for row in group]
        maes = [_finite(row.get("mae_return")) for row in group]
        outcomes.append({
            "source": source,
            "horizon_minutes": horizon,
            "n": len(returns),
            "mean_signed_return": sum(returns) / len(returns) if returns else None,
            "win_rate": sum(value > 0 for value in returns) / len(returns) if returns else None,
            "mean_mfe_return": _mean(mfes),
            "mean_mae_return": _mean(maes),
            "underpowered": len(returns) < 30,
        })
    return {
        "event_counts": dict(event_counts.most_common()),
        "candidate_registrations_by_source": dict(registration_sources.most_common()),
        "candidate_rejections_by_reason": dict(rejection_reasons.most_common()),
        "funnel": {
            "candidate_registered": event_counts.get("candidate_registered", 0),
            "setup_confirmed": confirmed,
            "setup_abstention": event_counts.get("setup_abstention", 0),
            "modeled_setup_close": event_counts.get("modeled_setup_close", 0),
        },
        "fixed_horizon_outcomes": outcomes,
        "outcomes_are_model_event_time_not_broker_fills": True,
    }


def render_decision_summary(summary: dict[str, Any]) -> str:
    funnel = summary["funnel"]
    lines = [
        "INTRADAY STRUCTURE — DISCOVERY / DECISION FUNNEL",
        (
            f"registered={funnel['candidate_registered']}  confirmed={funnel['setup_confirmed']}  "
            f"abstained={funnel['setup_abstention']}  modeled_closes={funnel['modeled_setup_close']}"
        ),
        "candidate sources:",
    ]
    for source, count in summary["candidate_registrations_by_source"].items():
        lines.append(f"  {count:6d}  {source}")
    if summary["candidate_rejections_by_reason"]:
        lines.append("rejections:")
        for reason, count in summary["candidate_rejections_by_reason"].items():
            lines.append(f"  {count:6d}  {reason}")
    if summary["fixed_horizon_outcomes"]:
        lines.append("fixed-horizon model outcomes (* n<30):")
        for row in summary["fixed_horizon_outcomes"]:
            flag = "*" if row["underpowered"] else " "
            lines.append(
                f"  {row['source']:<26} {row['horizon_minutes']:>3}m {flag} n={row['n']:<5d} "
                f"mean={_pct(row['mean_signed_return'])} win={_pct(row['win_rate'])} "
                f"MFE={_pct(row['mean_mfe_return'])} MAE={_pct(row['mean_mae_return'])}"
            )
    lines.append("Outcomes are event-time model measurements, not broker fills.")
    return "\n".join(lines)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.3f}%"
