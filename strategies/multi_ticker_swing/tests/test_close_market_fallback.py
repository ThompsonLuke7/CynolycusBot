"""Exit robustness: when the limit ladder can't get a verified fill during RTH,
_submit_close_order must fall back to a MARKET sell that closes the position."""
from __future__ import annotations

from datetime import datetime

import pytest

import strategies.multi_ticker_swing.live.position_manager as pm
from strategies.multi_ticker_swing.live.position_manager import (
    SwingPosition,
    SwingPositionManager,
)
from strategies.multi_ticker_swing.live.universe import TickerConfig

SYMBOL = "AXTI260717C00070000"


class _FakeClient:
    """Limit sells never fill (stay 'new'); a market sell fills immediately."""

    def __init__(self, *, held_qty: int = 10):
        self.held_qty = held_qty
        self.market_orders: list[dict] = []
        self.limit_orders: list[dict] = []
        self.canceled: list[str] = []

    def get_option_quotes(self, symbols=None, limit=None):
        return {"quotes": {SYMBOL: {"bp": 5.00, "ap": 5.40}}}

    def submit_option_order(self, *, symbol, qty, side, order_type="market",
                            time_in_force="day", limit_price=None, stop_price=None,
                            position_intent=None):
        if order_type == "market":
            self.market_orders.append({"symbol": symbol, "qty": qty})
            self.held_qty = 0  # market fill flattens the position
            return {"id": "mkt-1", "status": "filled"}
        self.limit_orders.append({"symbol": symbol, "qty": qty, "limit_price": limit_price})
        return {"id": f"lim-{len(self.limit_orders)}", "status": "new"}

    def get_order(self, order_id):
        return {"id": order_id, "status": "new"}  # never progresses -> verify times out

    def cancel_order(self, order_id):
        self.canceled.append(order_id)
        return {"id": order_id, "status": "canceled"}

    def get_positions(self):
        if self.held_qty <= 0:
            return []
        return [{"symbol": SYMBOL, "side": "long", "qty": str(self.held_qty)}]


def _make_position() -> SwingPosition:
    cfg = TickerConfig(
        ticker="AXTI", tier=1, entry_threshold=0.5, sl_atr=2.0, np_n_bars=None,
        np_mfe_atr=None, avg_win_pct=0.1, avg_loss_pct=-0.05, profit_factor=1.5, sharpe=1.0,
    )
    return SwingPosition(
        ticker="AXTI", direction=1, entry_price=5.0, entry_time=datetime.now(),
        atr_at_entry=1.0, option_symbol=SYMBOL, qty=10, config=cfg,
    )


def test_market_fallback_closes_when_limit_never_fills(monkeypatch):
    monkeypatch.setattr(pm, "_market_is_open", lambda now=None: True)
    monkeypatch.setattr(pm, "_ORDER_VERIFY_TIMEOUT_SECS", 0.0)  # don't block the test
    client = _FakeClient(held_qty=10)
    mgr = SwingPositionManager(client, dry_run=False)

    result = mgr._submit_close_order(_make_position(), reason="tp")

    assert result["verified"] is True
    assert result.get("via_market_fallback") is True
    assert client.market_orders == [{"symbol": SYMBOL, "qty": 10}]
    assert client.limit_orders, "limit ladder should have been tried first"


def test_no_market_fallback_when_market_closed(monkeypatch):
    monkeypatch.setattr(pm, "_market_is_open", lambda now=None: False)
    monkeypatch.setattr(pm, "_ORDER_VERIFY_TIMEOUT_SECS", 0.0)
    client = _FakeClient(held_qty=10)
    mgr = SwingPositionManager(client, dry_run=False)

    result = mgr._submit_close_order(_make_position(), reason="tp")

    assert result.get("via_market_fallback") is None
    assert client.market_orders == []  # limits only, no market outside RTH


def test_market_fallback_skips_when_already_flat(monkeypatch):
    # Limit "times out" but the position is actually gone (limit filled) -> no market sell.
    monkeypatch.setattr(pm, "_market_is_open", lambda now=None: True)
    monkeypatch.setattr(pm, "_ORDER_VERIFY_TIMEOUT_SECS", 0.0)
    client = _FakeClient(held_qty=0)
    mgr = SwingPositionManager(client, dry_run=False)

    result = mgr._submit_close_order(_make_position(), reason="tp")

    assert result["verified"] is True
    assert client.market_orders == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
