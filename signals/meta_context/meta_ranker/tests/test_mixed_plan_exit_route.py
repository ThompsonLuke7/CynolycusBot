"""Meta's options-mode pass submits ONE plan holding option and share rows.

`_submit_via_gateway` labelled every sell in that plan with the plan-level
`is_option` flag, so a share exit on the 14:20 pass was handed to the exit
ledger as an option and booked at the x100 contract multiplier (EQPT
2026-09-24: -$90,611 recorded for a -$906.11 loss). Exits flushed at the open
were unaffected, because the deferred queue carries each row's own route.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import signals.meta_context.meta_ranker.live_runner as lr
from core.nervous_system.execution.gateway import ExecutionOutcome

_OCC = "AAA260717C00050000"
_BAR = pd.Timestamp("2026-09-24 18:00:00+00:00")


def _accepted(symbol, qty, broker_id):
    result = SimpleNamespace(outcome=ExecutionOutcome.SUBMITTED, reason_code=None,
                             detail=None, broker_order_id=broker_id)
    return SimpleNamespace(symbol=symbol, side="sell", quantity=qty, refusal=None,
                           policy_vetoes=(), outcome=SimpleNamespace(execution_result=result),
                           intent=SimpleNamespace(reason_codes=("horizon",)))


class _Router:
    def __init__(self, rows):
        self._rows = rows

    def route(self, plan, *, on_row, **_kwargs):
        for row in self._rows:
            on_row(row)


def _submit(monkeypatch, plan, rows, *, is_option):
    tracked: list[tuple] = []
    monkeypatch.setattr(lr, "build_router", lambda **_k: _Router(rows))
    monkeypatch.setattr(lr, "intent_config", lambda args, **_k: None)
    monkeypatch.setattr(lr, "_save_state", lambda state: None)
    monkeypatch.setattr(lr, "track_exit_submission",
                        lambda client, **k: tracked.append(k["item"]))
    lr._submit_via_gateway(
        SimpleNamespace(), plan, {}, {}, _BAR, is_option=is_option, limits={},
        exit_context={}, module="meta_ranker", pos_lookup={}, client=None)
    return tracked


def test_a_share_exit_in_an_options_mode_plan_is_tracked_as_equity(monkeypatch):
    plan = [(_OCC, "sell", 5, "take_profit_+30%", "option"),
            ("EQPT", "sell", 251, "horizon", "equity")]
    rows = [_accepted(_OCC, 5, "o-1"), _accepted("EQPT", 251, "o-2")]

    tracked = _submit(monkeypatch, plan, rows, is_option=True)

    assert [(item[0], item[4]) for item in tracked] == [(_OCC, "option"), ("EQPT", "equity")]


def test_a_plan_without_row_routes_still_follows_the_mode_flag(monkeypatch):
    # The equity-mode plan is 4-field; the flag is the only label it has.
    tracked = _submit(monkeypatch, [("EQPT", "sell", 251, "horizon")],
                      [_accepted("EQPT", 251, "o-2")], is_option=False)

    assert [(item[0], item[4]) for item in tracked] == [("EQPT", "equity")]
