from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pandas as pd
import pytest

from core.nervous_system.contracts.enums import RuntimeEnvironment
from core.nervous_system.contracts.orders import OrderRequest
from strategies.momentum_scalper.configs.v1 import MomentumScalperConfigV1, PatternConfigV1, RiskConfigV1, ScannerConfigV1
from strategies.momentum_scalper.contracts import OrderIntent, SetupKind
from strategies.momentum_scalper.execution.paper_bridge import to_paper_order_request
from strategies.momentum_scalper.replay.event_engine import MomentumReplayEngine


def _ts(day: str, time: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {time}", tz="America/New_York").tz_convert("UTC")


def _config() -> MomentumScalperConfigV1:
    return MomentumScalperConfigV1(
        scanner=ScannerConfigV1(min_premarket_volume=1, min_rvol=1, min_rvol_history_sessions=5, min_daily_resistance_distance_pct=1, max_quote_age_seconds=5),
        patterns=PatternConfigV1(min_breakout_volume_ratio=1.5, max_chase_pct=20.0),
        risk=RiskConfigV1(max_position_notional=10_000, max_quote_participation=1.0, partial_at_r=2.0, partial_fraction=0.5, max_hold_minutes=30, entry_expiry_seconds=90),
    )


def _bars() -> pd.DataFrame:
    rows = []
    for day in ("2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"):
        rows += [
            {"timestamp": _ts(day, "04:00"), "ticker": "RUNR", "open": 8, "high": 8.2, "low": 7.9, "close": 8, "volume": 100},
            {"timestamp": _ts(day, "16:00"), "ticker": "RUNR", "open": 8, "high": 8.5, "low": 7.8, "close": 8, "volume": 1000},
        ]
    rows += [
        {"timestamp": _ts("2026-08-10", "04:00"), "ticker": "RUNR", "open": 9, "high": 9, "low": 9, "close": 9, "volume": 100000},
        {"timestamp": _ts("2026-08-10", "04:01"), "ticker": "RUNR", "open": 9, "high": 10, "low": 9, "close": 10, "volume": 200000},
        {"timestamp": _ts("2026-08-10", "04:02"), "ticker": "RUNR", "open": 10, "high": 10.1, "low": 9.9, "close": 10, "volume": 100000},
        {"timestamp": _ts("2026-08-10", "04:03"), "ticker": "RUNR", "open": 10, "high": 12.1, "low": 10, "close": 12, "volume": 200000},
        {"timestamp": _ts("2026-08-10", "04:04"), "ticker": "RUNR", "open": 12, "high": 12.1, "low": 10.5, "close": 11, "volume": 100000},
    ]
    return pd.DataFrame(rows)


def test_replay_fills_only_on_later_quote_and_records_partial_exit() -> None:
    metadata = pd.DataFrame([{"ticker": "RUNR", "float": 5_000_000, "available_at": _ts("2026-08-10", "03:59")}])
    news = pd.DataFrame([{"ticker": "RUNR", "material": True, "available_at": _ts("2026-08-10", "03:59")}])
    quotes = pd.DataFrame([
        {"ticker": "RUNR", "timestamp": _ts("2026-08-10", time), "received_at": _ts("2026-08-10", time), "bid": bid, "ask": ask, "bid_size": 10000, "ask_size": 10000, "feed": "SIP"}
        for time, bid, ask in (("04:01", 9.99, 10.0), ("04:02", 9.99, 10.0), ("04:03", 12.0, 12.01), ("04:04", 11.0, 11.01))
    ])
    result = MomentumReplayEngine(config=_config(), risk_budget_dollars=100).run(bars=_bars(), metadata=metadata, news=news, quotes=quotes)
    fills = result.frame("fills")

    assert list(fills["side"]) == ["buy", "sell", "sell"]
    assert fills.iloc[0]["timestamp"] == _ts("2026-08-10", "04:02").to_pydatetime()
    assert "partial_at_target" in set(fills["reason"])


def test_paper_bridge_hashes_extended_hours_and_rejects_invalid_market_order() -> None:
    created = _ts("2026-08-10", "04:01").to_pydatetime()
    intent = OrderIntent("RUNR", created, created + timedelta(seconds=10), 10, 10.0, 9.0, SetupKind.HOD_BREAKOUT, ("test",), True)
    request = to_paper_order_request(intent=intent, decision_id=uuid4(), policy_decision_id=uuid4(), account_alias="paper")
    assert request.environment is RuntimeEnvironment.QA_PAPER
    assert request.extended_hours is True
    with pytest.raises(ValueError, match="extended-hours"):
        OrderRequest.create(
            decision_id=uuid4(), policy_decision_id=uuid4(), environment=RuntimeEnvironment.QA_PAPER,
            account_alias="paper", decision_kind=request.decision_kind, risk_reducing=False,
            instrument_family=request.instrument_family, equity_symbol="RUNR", equity_side=request.equity_side,
            parent_quantity=request.parent_quantity, debit_credit=request.debit_credit, net_limit_price=None,
            maximum_loss=request.maximum_loss, buying_power_required=request.buying_power_required,
            time_in_force="day", order_type="market", extended_hours=True, idempotency_key="a" * 64,
            created_at=created, expires_at=created + timedelta(seconds=10),
        )
