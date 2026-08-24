"""The risk pass clears an entry's unconfirmed flag once the broker reports it.

mark_entry_unconfirmed is deliberate: an accepted-but-unfilled entry stays
CLAIMED rather than dropped, because a limit that fills later would otherwise
become a position nobody owns — Swing force-sold Dealer Ranker's
IOT260724C00031500 for -$4,945 on 2026-07-23 for exactly that reason.

What was wrong is the latency, not the design. Only build_mixed_plan cleared the
flag, and for a 4H module "the next pass" can be twenty hours away:
dealer_ranker's MRNA260828C00140000 and NEM260828C00130000 sat flagged from
2026-08-20 15:52 to 2026-08-21 15:52. The risk pass already reads broker
positions every ~5 minutes.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from core.live_4h_exec import ExecPolicy
from core.live_risk_pass import RiskPassConfig, evaluate_risk_exits

_ET = ZoneInfo("America/New_York")
NOW = dt.datetime(2026, 8, 20, 16, 30, tzinfo=_ET)
CDE = "CDE260828C00022000"


def _policy():
    return ExecPolicy(target_notional=5000.0)


def test_a_confirmed_entry_loses_its_unconfirmed_flag():
    managed = {"CDE": {"route": "option", "occ": CDE, "contracts": 94,
                       "pending_fill": True, "entry_order_id": "cd670d83",
                       "u_entry": 21.5, "u_atr": 0.8}}
    res = evaluate_risk_exits(
        client=None, module="dealer_ranker", managed=managed,
        pos_info={CDE: {"qty": 94, "avg_entry": 0.50, "current": 0.35}},
        policy=_policy(), now_et=NOW, cfg=RiskPassConfig(hard_stop=False, expiry_flatten=False),
        underlying_fn=lambda _t, at=None: (21.5, 0.8),
    )

    assert res.confirmed_entries == {"CDE": {"symbol": CDE, "route": "option", "qty": 94}}
    assert "pending_fill" not in res.new_managed["CDE"]
    assert "entry_order_id" not in res.new_managed["CDE"]


def test_an_unfilled_entry_keeps_its_flag_and_its_claim():
    """The broker reports nothing, so the order may still be resting. Dropping
    it here is what the flag exists to prevent."""
    managed = {"CDE": {"route": "option", "occ": CDE, "contracts": 94,
                       "pending_fill": True, "entry_order_id": "cd670d83"}}
    res = evaluate_risk_exits(
        client=None, module="dealer_ranker", managed=managed, pos_info={},
        policy=_policy(), now_et=NOW, cfg=RiskPassConfig(hard_stop=False, expiry_flatten=False),
        underlying_fn=lambda _t, at=None: (21.5, 0.8),
    )

    assert res.confirmed_entries == {}
    assert res.new_managed["CDE"]["pending_fill"] is True
    assert "CDE" in res.new_managed          # still claimed


def test_a_position_that_was_never_flagged_is_untouched():
    managed = {"CDE": {"route": "option", "occ": CDE, "contracts": 94,
                       "u_entry": 21.5, "u_atr": 0.8}}
    res = evaluate_risk_exits(
        client=None, module="dealer_ranker", managed=managed,
        pos_info={CDE: {"qty": 94, "avg_entry": 0.50, "current": 0.35}},
        policy=_policy(), now_et=NOW, cfg=RiskPassConfig(hard_stop=False, expiry_flatten=False),
        underlying_fn=lambda _t, at=None: (21.5, 0.8),
    )
    assert res.confirmed_entries == {}
