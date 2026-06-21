from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import pytest

from strategies.dealer_positioning.config import DealerPositioningConfig
from strategies.dealer_positioning.execution import DealerTradeExecutor
from strategies.dealer_positioning.models import DealerSignal


class _FakeAlpacaClient:
    def get_option_contracts(self, **params):
        return {
            "option_contracts": [
                {
                    "symbol": "SPY260612C00510000",
                    "expiration_date": "2026-06-12",
                    "strike_price": 510,
                    "tradable": True,
                    "size": 100,
                },
                {
                    "symbol": "SPY260612C00515000",
                    "expiration_date": "2026-06-12",
                    "strike_price": 515,
                    "tradable": True,
                    "size": 100,
                },
            ]
        }

    def get_option_quotes(self, **params):
        symbol = params.get("symbols")
        return {
            symbol: {
                "bid_price": 2.0,
                "ask_price": 2.2,
                "last_price": 2.1,
            }
        }

    def submit_option_order(self, **params):
        return {"id": "paper-order-1", "status": "accepted"}


def test_execution_policy_uses_target_level_for_contract_choice() -> None:
    config = DealerPositioningConfig(submit_orders=False, alpaca_env_file=".env#paper")
    engine = DealerTradeExecutor.__new__(DealerTradeExecutor)
    engine._config = config
    engine._open = defaultdict(list)
    engine._closed = []
    engine._counter = 0
    engine._client = _FakeAlpacaClient()

    signal = DealerSignal(
        timestamp=datetime(2026, 6, 12, 14, 35, tzinfo=timezone.utc).isoformat(),
        symbol="SPY",
        signal_type="bullish_magnet",
        direction="long",
        spot=507.5,
        level=505.0,
        target=510.0,
        stop=505.0,
        reason="held above broken magnet",
        confidence=0.8,
    )
    trade = engine.on_signal(signal)
    assert trade is not None
    assert trade.contract_symbol == "SPY260612C00510000"
    assert trade.contract_strike == 510
    assert trade.entry_contract_price == 2.2
    assert trade.entry_contract_bid == 2.0
    assert trade.entry_contract_ask == 2.2
    assert round(float(trade.entry_contract_spread_pct), 4) == 0.0952
    assert trade.order_mode == "alpaca_simulated"


def test_execution_policy_finalizes_expired_contracts_at_intrinsic_value() -> None:
    config = DealerPositioningConfig(submit_orders=False, option_order_qty=1)
    engine = DealerTradeExecutor.__new__(DealerTradeExecutor)
    engine._config = config
    engine._open = defaultdict(list)
    engine._closed = []
    engine._counter = 0
    engine._client = _FakeAlpacaClient()

    signal = DealerSignal(
        timestamp=datetime(2026, 6, 18, 14, 35, tzinfo=timezone.utc).isoformat(),
        symbol="SPY",
        signal_type="bullish_magnet",
        direction="long",
        spot=507.5,
        level=505.0,
        target=510.0,
        stop=505.0,
        reason="held above broken magnet",
        confidence=0.8,
    )
    trade = engine.on_signal(signal)
    assert trade is not None
    trade.contract_expiration = "2026-06-18"
    trade.contract_strike = 510.0
    trade.entry_contract_price = 2.2

    closed = engine.finalize_expired(
        datetime(2026, 6, 18, 20, 1, tzinfo=timezone.utc),
        {"SPY": 511.0},
    )

    assert len(closed) == 1
    assert closed[0].status == "closed"
    assert closed[0].exit_reason == "expiration"
    assert closed[0].exit_contract_price == 1.0
    assert closed[0].pnl_dollars == pytest.approx(-120.0)
    assert engine.open_trades == []
