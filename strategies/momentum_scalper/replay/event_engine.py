"""One chronological event clock for scanner, setups, orders, and positions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import count

import pandas as pd

from strategies.momentum_scalper.configs.v1 import MomentumScalperConfigV1
from strategies.momentum_scalper.contracts import OrderIntent, QuoteSnapshot
from strategies.momentum_scalper.execution.entry_policy import build_paper_order_intent
from strategies.momentum_scalper.patterns.state_machine import PatternStateMachine
from strategies.momentum_scalper.portfolio.positions import MomentumPosition
from strategies.momentum_scalper.portfolio.risk import MomentumRiskBook
from strategies.momentum_scalper.scanners.premarket import reconstruct_premarket_snapshot
from strategies.momentum_scalper.utils.io import add_session_columns, normalize_timestamp_column


@dataclass
class ReplayResult:
    decisions: list[dict[str, object]] = field(default_factory=list)
    orders: list[dict[str, object]] = field(default_factory=list)
    fills: list[dict[str, object]] = field(default_factory=list)
    transitions: list[dict[str, object]] = field(default_factory=list)

    def frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(getattr(self, name))


def _quote_at(quotes: pd.DataFrame, ticker: str, timestamp: pd.Timestamp) -> QuoteSnapshot | None:
    if quotes.empty:
        return None
    out = quotes.copy()
    required = {"ticker", "timestamp", "bid", "ask"}
    if not required.issubset(out.columns):
        return None
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    received = pd.to_datetime(out.get("received_at", out["timestamp"]), utc=True, errors="coerce")
    out["_received_at"] = received
    out = out[(out["ticker"] == ticker) & (out["timestamp"] <= timestamp) & (out["_received_at"] <= timestamp)]
    if out.empty:
        return None
    row = out.sort_values("_received_at").iloc[-1]
    return QuoteSnapshot(
        ticker=ticker,
        event_at=row["timestamp"].to_pydatetime(),
        received_at=row["_received_at"].to_pydatetime(),
        bid=float(row["bid"]), ask=float(row["ask"]),
        bid_size=float(row.get("bid_size", 0.0) or 0.0),
        ask_size=float(row.get("ask_size", 0.0) or 0.0),
        feed=str(row.get("feed", "UNKNOWN")),
    )


class MomentumReplayEngine:
    """Replay one frozen configuration using conservative next-observation fills."""

    def __init__(
        self,
        *,
        config: MomentumScalperConfigV1,
        risk_budget_dollars: float,
    ) -> None:
        self.config = config.validate()
        if risk_budget_dollars <= 0:
            raise ValueError("risk_budget_dollars must be positive")
        self.risk_budget_dollars = risk_budget_dollars

    def run(
        self,
        *,
        bars: pd.DataFrame,
        metadata: pd.DataFrame,
        news: pd.DataFrame,
        quotes: pd.DataFrame,
        halts: pd.DataFrame | None = None,
    ) -> ReplayResult:
        ordered = add_session_columns(normalize_timestamp_column(bars))
        if ordered.empty:
            return ReplayResult()
        result = ReplayResult()
        machines: dict[str, PatternStateMachine] = {}
        pending: dict[str, OrderIntent] = {}
        positions: dict[str, MomentumPosition] = {}
        risk = MomentumRiskBook(self.config.risk)
        order_counter = count(1)
        traded_tickers: set[str] = set()

        for timestamp, batch in ordered.groupby("timestamp", sort=True):
            # First resolve prior intents; a signal cannot fill on the event that made it.
            for ticker, intent in list(pending.items()):
                quote = _quote_at(quotes, ticker, timestamp)
                if timestamp.to_pydatetime() >= intent.expires_at:
                    result.orders.append({"timestamp": timestamp, "ticker": ticker, "event": "expired", "reason": "entry_expired"})
                    del pending[ticker]
                elif quote is not None and quote.event_at > intent.created_at and quote.ask <= intent.limit_price:
                    order_id = f"replay-{next(order_counter)}"
                    positions[ticker] = MomentumPosition.from_fill(
                        ticker=ticker, opened_at=quote.event_at, entry_price=quote.ask,
                        stop_price=intent.stop_price, quantity=intent.quantity, config=self.config.risk,
                    )
                    risk.record_entry()
                    traded_tickers.add(ticker)
                    result.fills.append({"timestamp": quote.event_at, "ticker": ticker, "order_id": order_id, "side": "buy", "quantity": intent.quantity, "price": quote.ask, "reason": "next_quote_fill"})
                    del pending[ticker]

            # Existing positions use the current bar and bid; a missing quote does not invent an exit fill.
            for ticker, position in list(positions.items()):
                ticker_bars = batch[batch["ticker"] == ticker]
                if ticker_bars.empty:
                    continue
                bar = ticker_bars.iloc[-1]
                quote = _quote_at(quotes, ticker, timestamp)
                instructions = position.process_bar(
                    timestamp=timestamp.to_pydatetime(), open_price=float(bar["open"]),
                    high=float(bar["high"]), low=float(bar["low"]), close=float(bar["close"]),
                    config=self.config.risk,
                )
                for instruction in instructions:
                    if quote is None or quote.bid <= 0:
                        result.orders.append({"timestamp": timestamp, "ticker": ticker, "event": "exit_unfilled", "reason": f"{instruction.reason}:missing_quote"})
                        continue
                    if instruction.reason == "partial_at_target" and quote.bid < instruction.limit_price:
                        result.orders.append({"timestamp": timestamp, "ticker": ticker, "event": "exit_unfilled", "reason": "target_not_marketable"})
                        continue
                    price = instruction.limit_price if instruction.reason == "partial_at_target" else min(instruction.limit_price, quote.bid)
                    position.apply_exit(quantity=instruction.quantity, price=price)
                    result.fills.append({"timestamp": quote.event_at, "ticker": ticker, "order_id": f"replay-{next(order_counter)}", "side": "sell", "quantity": instruction.quantity, "price": price, "reason": instruction.reason})
                if position.remaining_quantity == 0:
                    denominator = (position.entry_price - position.initial_stop) * position.quantity
                    realized_r = position.realized_pnl / denominator if denominator > 0 else 0.0
                    risk.record_close(realized_r)
                    del positions[ticker]

            local_time = timestamp.tz_convert("America/New_York").time()
            cutoff_hour, cutoff_minute = (int(part) for part in self.config.new_entry_cutoff_et.split(":"))
            if (local_time.hour, local_time.minute) > (cutoff_hour, cutoff_minute):
                continue
            snapshot = reconstruct_premarket_snapshot(
                ordered, decision_at=timestamp, metadata=metadata, news=news, quotes=quotes,
                halts=halts, config=self.config.scanner,
            )
            for row in snapshot[snapshot["eligible"]].sort_values("scanner_rank").itertuples(index=False):
                ticker = str(row.ticker)
                if ticker in positions or ticker in pending or ticker in traded_tickers:
                    continue
                session_date = timestamp.tz_convert("America/New_York").strftime("%Y-%m-%d")
                history = ordered[
                    (ordered["ticker"] == ticker)
                    & (ordered["timestamp"] <= timestamp)
                    & (ordered["date"] == session_date)
                ]
                machine = machines.setdefault(ticker, PatternStateMachine(ticker=ticker, config=self.config.patterns))
                signals, transitions = machine.observe(history, halted=bool(row.halted), data_stale=float(row.quote_age_seconds) > self.config.scanner.max_quote_age_seconds)
                result.transitions.extend({
                    "timestamp": t.timestamp, "ticker": t.ticker, "pattern": t.kind.value,
                    "previous": t.previous.value, "current": t.current.value, "reason": t.reason,
                } for t in transitions)
                if not signals:
                    continue
                quote = _quote_at(quotes, ticker, timestamp)
                if quote is None:
                    continue
                # Prefer the first declared detector result so replay and live ordering are deterministic.
                intent, reason = build_paper_order_intent(
                    signal=signals[0], quote=quote, config=self.config, risk_book=risk,
                    risk_budget_dollars=self.risk_budget_dollars,
                )
                result.decisions.append({"timestamp": timestamp, "ticker": ticker, "pattern": signals[0].kind.value, "decision": "enter" if intent else "reject", "reason": reason})
                if intent is not None:
                    pending[ticker] = intent
                    result.orders.append({"timestamp": timestamp, "ticker": ticker, "event": "intent", "limit_price": intent.limit_price, "quantity": intent.quantity, "extended_hours": intent.extended_hours})
        return result


__all__ = ["MomentumReplayEngine", "ReplayResult"]
