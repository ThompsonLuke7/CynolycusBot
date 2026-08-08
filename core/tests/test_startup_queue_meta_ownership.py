"""A queued close must say when it touched strategy-owned inventory.

`startup_queue` is a human escape hatch: an operator queues "flatten SNDK" and
the server executes it at boot. It is symbol-based and knows nothing about which
strategy owns the position.

Blocking it when a strategy owns the symbol would be the wrong trade. This is a
risk-*reducing*, explicitly authored, audited action, and an operator reaching
for it in an emergency must not be told no. What it must not do is act
invisibly: if it flattens inventory a strategy still believes it manages, the
strategy's own state and the ownership reconciliation need to be able to see it.
"""

from __future__ import annotations

import json
from typing import Any

from core.startup_queue import _close_position, meta_owned_symbols


class _FakeClient:
    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []

    def get_positions(self):
        return [{"symbol": "SNDK", "qty": "40", "side": "long"}]

    def submit_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"id": "ord-1", "status": "accepted"}


def _entry(symbol: str = "SNDK") -> dict[str, Any]:
    return {"params": {"symbol": symbol, "qty": "all"}}


def _write_state(tmp_path, managed: dict) -> Any:
    path = tmp_path / "live_state.json"
    path.write_text(json.dumps({"managed": managed}))
    return path


def test_a_close_still_happens_when_a_strategy_owns_the_symbol() -> None:
    """The emergency exit is never blocked; that is the whole point of it."""

    client = _FakeClient()
    result = _close_position(client, _entry(), owned_symbols={"SNDK"})

    assert result["closed"] is True
    assert len(client.orders) == 1


def test_a_close_on_owned_inventory_is_flagged_in_its_record() -> None:
    """Otherwise the strategy keeps managing a position that no longer exists,
    and nothing in the audit trail explains where it went.
    """

    result = _close_position(_FakeClient(), _entry(), owned_symbols={"SNDK"})

    assert result["strategy_owned"] is True


def test_a_close_on_unowned_inventory_is_not_flagged() -> None:
    result = _close_position(_FakeClient(), _entry(), owned_symbols=set())

    assert result["strategy_owned"] is False


def test_meta_ownership_reads_both_equity_and_option_symbols(tmp_path) -> None:
    path = _write_state(
        tmp_path,
        {
            "AMD": {"route": "option", "occ": "AMD260821C00200000"},
            "MSFT": {"route": "equity", "symbol": "MSFT"},
        },
    )

    owned = meta_owned_symbols(state_path=path)

    assert owned == {"AMD260821C00200000", "MSFT"}


def test_missing_meta_state_is_not_an_error(tmp_path) -> None:
    """A module that has never run owns nothing; that must not break the queue."""

    assert meta_owned_symbols(state_path=tmp_path / "absent.json") == set()


def test_unreadable_meta_state_owns_nothing_rather_than_raising(tmp_path) -> None:
    path = tmp_path / "live_state.json"
    path.write_text("{not json")

    assert meta_owned_symbols(state_path=path) == set()
