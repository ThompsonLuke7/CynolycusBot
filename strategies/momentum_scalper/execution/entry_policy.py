"""Paper-only entry policy and legacy dataframe compatibility helper."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategies.momentum_scalper.configs.settings import EntryConfig
from strategies.momentum_scalper.configs.v1 import MomentumScalperConfigV1
from strategies.momentum_scalper.contracts import OrderIntent, PatternSignal, QuoteSnapshot
from strategies.momentum_scalper.portfolio.risk import MomentumRiskBook, size_shares


@dataclass(frozen=True)
class EntrySignal:
    should_enter: bool
    pattern: str
    trigger_price: float
    reason: str = ""


def build_paper_order_intent(
    *,
    signal: PatternSignal,
    quote: QuoteSnapshot,
    config: MomentumScalperConfigV1,
    risk_book: MomentumRiskBook,
    risk_budget_dollars: float,
) -> tuple[OrderIntent | None, str]:
    """Build a fill-independent paper order intent from a fresh ask quote.

    This does not submit to a broker.  It is deliberately separate from the
    legacy dataframe helper below so live code cannot turn a minute close into
    an execution price.
    """

    if not config.paper_only:
        raise ValueError("momentum scalper v1 must remain paper-only")
    allowed, reason = risk_book.allow_entry()
    if not allowed:
        return None, reason
    quote_delay = (quote.received_at - quote.event_at).total_seconds()
    signal_to_quote = (quote.event_at - signal.timestamp).total_seconds()
    if quote_delay < 0 or signal_to_quote < 0 or signal_to_quote > config.scanner.max_quote_age_seconds:
        return None, "stale_quote"
    if quote.spread_pct > config.scanner.max_spread_pct:
        return None, "spread_above_limit"
    max_entry = signal.trigger_price * (1.0 + config.patterns.max_chase_pct / 100.0)
    if quote.ask > max_entry:
        return None, "chase_limit"
    quantity = size_shares(
        entry_price=quote.ask,
        stop_price=signal.invalidation_price,
        risk_budget_dollars=risk_budget_dollars,
        ask_size=quote.ask_size,
        config=config.risk,
    )
    if quantity <= 0:
        return None, "size_zero"
    local = quote.event_at.astimezone(ZoneInfo("America/New_York")).time()
    # The strategy's runner has the authoritative session calendar. This local
    # test intentionally only marks obvious premarket intents for the broker
    # adapter; a regular-hours intent remains a standard DAY limit order.
    extended_hours = local.hour < 9 or (local.hour == 9 and local.minute < 30)
    intent = OrderIntent(
        ticker=signal.ticker,
        created_at=quote.received_at,
        expires_at=quote.received_at + timedelta(seconds=config.risk.entry_expiry_seconds),
        quantity=quantity,
        limit_price=quote.ask,
        stop_price=signal.invalidation_price,
        pattern=signal.kind,
        reason_codes=signal.reason_codes,
        extended_hours=extended_hours,
    )
    return intent, "ok"


def _passes_filters(row: pd.Series, config: EntryConfig) -> tuple[bool, str]:
    if float(row.get("spread_pct", 0.0) or 0.0) > config.max_spread_pct:
        return False, "spread"
    if float(row.get("liquidity_score", 0.0) or 0.0) < config.min_liquidity_score:
        return False, "liquidity"
    if float(row.get("halt_count", 0.0) or 0.0) > 0:
        return False, "halt_risk"
    if float(row.get("dist_to_hod", 0.0) or 0.0) < -config.max_chase_pct_above_trigger:
        return False, "anti_chase"
    return True, ""


def evaluate_entry(row: pd.Series, config: EntryConfig = EntryConfig()) -> EntrySignal:
    raise RuntimeError(
        "Legacy dataframe entry evaluation is retired. Use causal PatternSignal plus "
        "build_paper_order_intent with a fresh two-sided quote."
    )
    passed, reason = _passes_filters(row, config)
    price = float(row.get("close", np.nan) if "close" in row else np.nan)
    if not passed:
        return EntrySignal(False, "none", price, reason)
    patterns = [
        ("HOD breakout", row.get("premarket_high_break", 0)),
        ("first pullback", row.get("micro_pullback", 0)),
        ("bull flag breakout", (row.get("bull_flag_tightness", 1) < 0.015) and (row.get("breakout_velocity", 0) > 0)),
        ("flat-top breakout", row.get("flat_top_breakout", 0)),
        ("VWAP reclaim", row.get("dist_to_vwap", -1) > 0),
        ("opening range breakout", row.get("opening_range_break", 0)),
    ]
    for pattern, ok in patterns:
        if bool(ok):
            return EntrySignal(True, pattern, price, "ok")
    return EntrySignal(False, "none", price, "no_pattern")
