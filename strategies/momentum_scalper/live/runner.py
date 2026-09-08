"""Session coordinator for paper-safe momentum shadow operation."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from strategies.momentum_scalper.configs.v1 import MomentumScalperConfigV1
from strategies.momentum_scalper.live.ledger import ShadowLedger
from strategies.momentum_scalper.live.shadow import MomentumShadowRunner, ShadowResult


class MomentumShadowSession:
    """Persist shadow decisions; deliberately has no broker or order-submit dependency."""

    def __init__(
        self,
        *,
        config: MomentumScalperConfigV1,
        risk_budget_dollars: float,
        ledger: ShadowLedger,
    ) -> None:
        self.runner = MomentumShadowRunner(config=config, risk_budget_dollars=risk_budget_dollars)
        self.ledger = ledger

    def evaluate_and_record(
        self,
        *,
        timestamp: datetime,
        bars: pd.DataFrame,
        metadata: pd.DataFrame,
        news: pd.DataFrame,
        quotes: pd.DataFrame,
        halts: pd.DataFrame | None = None,
    ) -> ShadowResult:
        result = self.runner.evaluate(
            timestamp=timestamp, bars=bars, metadata=metadata, news=news,
            quotes=quotes, halts=halts,
        )
        for row in result.transitions:
            self.ledger.append("transitions", row)
        for row in result.decisions:
            self.ledger.append("decisions", row)
        for intent in result.intents:
            self.ledger.append("intents", {
                "ticker": intent.ticker, "created_at": intent.created_at,
                "expires_at": intent.expires_at, "quantity": intent.quantity,
                "limit_price": intent.limit_price, "stop_price": intent.stop_price,
                "pattern": intent.pattern.value, "reason_codes": intent.reason_codes,
                "extended_hours": intent.extended_hours, "paper_only": intent.paper_only,
            })
        return result


__all__ = ["MomentumShadowSession"]
