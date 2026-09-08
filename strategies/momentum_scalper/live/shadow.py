"""Paper-safe live shadow runner; it records intents and never submits orders."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from strategies.momentum_scalper.configs.v1 import MomentumScalperConfigV1
from strategies.momentum_scalper.contracts import OrderIntent
from strategies.momentum_scalper.execution.entry_policy import build_paper_order_intent
from strategies.momentum_scalper.patterns.state_machine import PatternStateMachine
from strategies.momentum_scalper.portfolio.risk import MomentumRiskBook
from strategies.momentum_scalper.replay.event_engine import _quote_at
from strategies.momentum_scalper.scanners.premarket import reconstruct_premarket_snapshot
from strategies.momentum_scalper.utils.io import add_session_columns, normalize_timestamp_column


@dataclass
class ShadowResult:
    timestamp: datetime
    intents: list[OrderIntent] = field(default_factory=list)
    decisions: list[dict[str, object]] = field(default_factory=list)
    transitions: list[dict[str, object]] = field(default_factory=list)


class MomentumShadowRunner:
    """Evaluate current inputs without any broker write capability."""

    def __init__(self, *, config: MomentumScalperConfigV1, risk_budget_dollars: float) -> None:
        self.config = config.validate()
        self.risk_budget_dollars = risk_budget_dollars
        self.risk = MomentumRiskBook(self.config.risk)
        self.machines: dict[str, PatternStateMachine] = {}
        self.claimed_tickers: set[str] = set()

    def evaluate(
        self,
        *,
        timestamp: datetime,
        bars: pd.DataFrame,
        metadata: pd.DataFrame,
        news: pd.DataFrame,
        quotes: pd.DataFrame,
        halts: pd.DataFrame | None = None,
    ) -> ShadowResult:
        if not self.config.paper_only:
            raise ValueError("shadow runner requires paper-only configuration")
        decision_at = pd.Timestamp(timestamp)
        if decision_at.tzinfo is None:
            raise ValueError("shadow timestamp must be timezone-aware")
        decision_at = decision_at.tz_convert("UTC")
        local = decision_at.tz_convert(ZoneInfo("America/New_York"))
        cutoff_h, cutoff_m = (int(value) for value in self.config.new_entry_cutoff_et.split(":"))
        result = ShadowResult(timestamp=decision_at.to_pydatetime())
        if (local.hour, local.minute) > (cutoff_h, cutoff_m):
            result.decisions.append({"timestamp": decision_at, "decision": "reject", "reason": "new_entry_cutoff"})
            return result
        ordered = add_session_columns(normalize_timestamp_column(bars))
        snapshot = reconstruct_premarket_snapshot(
            ordered, decision_at=decision_at, metadata=metadata, news=news,
            quotes=quotes, halts=halts, config=self.config.scanner,
        )
        for row in snapshot[snapshot["eligible"]].sort_values("scanner_rank").itertuples(index=False):
            ticker = str(row.ticker)
            if ticker in self.claimed_tickers:
                continue
            session_date = decision_at.tz_convert(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            history = ordered[
                (ordered["ticker"] == ticker)
                & (ordered["timestamp"] <= decision_at)
                & (ordered["date"] == session_date)
            ]
            machine = self.machines.setdefault(ticker, PatternStateMachine(ticker=ticker, config=self.config.patterns))
            signals, transitions = machine.observe(history, halted=bool(row.halted), data_stale=float(row.quote_age_seconds) > self.config.scanner.max_quote_age_seconds)
            result.transitions.extend({
                "timestamp": event.timestamp, "ticker": event.ticker, "pattern": event.kind.value,
                "previous": event.previous.value, "current": event.current.value, "reason": event.reason,
            } for event in transitions)
            if not signals:
                continue
            quote = _quote_at(quotes, ticker, decision_at)
            if quote is None:
                result.decisions.append({"timestamp": decision_at, "ticker": ticker, "decision": "reject", "reason": "missing_quote"})
                continue
            intent, reason = build_paper_order_intent(
                signal=signals[0], quote=quote, config=self.config, risk_book=self.risk,
                risk_budget_dollars=self.risk_budget_dollars,
            )
            result.decisions.append({"timestamp": decision_at, "ticker": ticker, "pattern": signals[0].kind.value, "decision": "intent" if intent else "reject", "reason": reason})
            if intent is not None:
                self.claimed_tickers.add(ticker)
                result.intents.append(intent)
        return result


__all__ = ["MomentumShadowRunner", "ShadowResult"]
