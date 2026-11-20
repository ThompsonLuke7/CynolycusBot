"""The SPY daytrader under the account-wide entry freeze (core/trading_controls.py).

A refused open must leave the policy flat and the loop running: the 2026-09-03
no-quote 403 showed that an HTTPError escaping on_decision kills the SPY loop
for the rest of the session, exits included.
"""
from __future__ import annotations

import urllib.error
from datetime import date, datetime
from zoneinfo import ZoneInfo

from core.trading_controls import controls_path, write_controls
from strategies.spy_intraday.Policy.order_policy import (
    OptionOrderPolicy,
    OptionOrderPolicyConfig,
)

ET = ZoneInfo("America/New_York")
SYMBOL = "SPY261103C00755000"


def _freeze() -> None:
    write_controls(entries="frozen", reason="test", set_at_utc="2026-11-03T07:00:00Z", path=controls_path())


def test_frozen_entries_are_refused_before_any_order_with_an_honest_reason() -> None:
    pol = OptionOrderPolicy(OptionOrderPolicyConfig(submit_orders=True))
    assert pol._new_entries_allowed(local_ts=None)
    _freeze()
    assert not pol._new_entries_allowed(local_ts=None)
    assert pol._entry_block_label() == "entries_frozen"


def test_simulation_ignores_the_freeze() -> None:
    _freeze()
    pol = OptionOrderPolicy(OptionOrderPolicyConfig(submit_orders=False))
    assert pol._new_entries_allowed(local_ts=None)


def test_a_refused_open_holds_flat_instead_of_raising(monkeypatch) -> None:
    pol = OptionOrderPolicy(OptionOrderPolicyConfig(submit_orders=True))
    monkeypatch.setattr(
        pol, "_target_contracts_for_side", lambda *, side, **_: 1 if side == "long" else 0
    )
    monkeypatch.setattr(pol, "_get_buying_power", lambda: 1_000_000.0)
    monkeypatch.setattr(
        pol, "_select_side_contract",
        lambda **_: (SYMBOL, "call", date(2026, 11, 3), 755.0, 755.0),
    )
    monkeypatch.setattr(pol, "_contracts_max_for_symbol", lambda **_: (5, "long", 1.0))

    def refused(**_):
        raise urllib.error.HTTPError(
            "https://paper-api.alpaca.markets/v2/orders", 403,
            'Forbidden: {"code": "entries_frozen", "message": "frozen"}', None, None,
        )

    monkeypatch.setattr(pol, "_submit_order", refused)
    logs: list[str] = []
    result = pol._on_independent_meta_decision(
        closed_bar={"high": 756.0, "low": 754.0},
        logger=logs.append,
        close=755.0,
        atr=1.0,
        local_ts=datetime(2026, 11, 3, 10, 0, tzinfo=ET),
    )
    assert isinstance(result, dict)
    assert pol._long_contracts == 0 and pol._long_symbol is None
    assert any("OPEN REFUSED" in line for line in logs)
    assert pol._meta_side_reason["long"] == "open_refused_by_broker"
