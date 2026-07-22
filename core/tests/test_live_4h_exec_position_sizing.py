"""Dollar-notional position sizing in the shared 4H engine.

Fixed 100-share / N-contract entries meant wildly different risk per name (a
$5 stock was $500 of exposure, a $200 stock was $20,000; a $0.20 option
premium was $20/contract, a $50 premium was $5,000/contract), which also made
%/$ gain figures incomparable across the book. `build_mixed_plan` now sizes
every new entry off `ExecPolicy.target_notional` (default $5,000) instead.
"""
from __future__ import annotations

from core.live_4h_exec import (
    ExecPolicy,
    build_mixed_plan,
    contracts_for_notional,
    shares_for_notional,
)


class _FillClient:
    pass


def test_shares_for_notional_rounds_to_nearest_whole_share():
    assert shares_for_notional(50.0, 5000.0) == 100
    assert shares_for_notional(3.0, 5000.0) == 1667  # round(1666.67)
    assert shares_for_notional(200.0, 5000.0) == 25


def test_shares_for_notional_floors_at_one_share():
    assert shares_for_notional(50_000.0, 5000.0) == 1  # would round to 0 otherwise


def test_shares_for_notional_handles_missing_or_bad_price():
    assert shares_for_notional(None, 5000.0) == 1
    assert shares_for_notional(0.0, 5000.0) == 1
    assert shares_for_notional(-5.0, 5000.0) == 1


def test_contracts_for_notional_uses_100x_multiplier():
    assert contracts_for_notional(50.0, 5000.0) == 1     # $50 * 100 = $5,000/contract
    assert contracts_for_notional(0.20, 5000.0) == 250   # $0.20 * 100 = $20/contract
    assert contracts_for_notional(5.0, 5000.0) == 10      # $5 * 100 = $500/contract


def test_contracts_for_notional_floors_at_one_contract():
    assert contracts_for_notional(500.0, 5000.0) == 1  # $50,000/contract, would round to 0


def test_contracts_for_notional_handles_missing_or_bad_premium():
    assert contracts_for_notional(None, 5000.0) == 1
    assert contracts_for_notional(0.0, 5000.0) == 1


def test_build_mixed_plan_sizes_equity_entry_by_notional():
    def _equity_route(client, ticker, px, **kwargs):
        return "equity", None, "not_optionable"

    res = build_mixed_plan(
        _FillClient(), targets=["AAA"], managed={}, pos_info={}, bar="2026-07-21T18:00:00Z",
        signal_audits={}, policy=ExecPolicy(target_notional=5000.0), route_fn=_equity_route,
        ref_price_fn=lambda ticker: 25.0, verbose=False,
    )
    assert res.plan == [("AAA", "buy", 200, "entry", "equity")]  # 5000/25 = 200 shares
    assert res.new_managed["AAA"]["shares"] == 200


def test_build_mixed_plan_sizes_option_entry_by_notional():
    def _option_route(client, ticker, px, **kwargs):
        return "option", {"occ": "AAA260814C00025000", "limit": 2.0, "mid": 1.9,
                           "delta": 0.5, "strike": 25.0, "expiry": "2026-08-14",
                           "open_interest": 10, "volume": 5, "spread": 0.05}, "ok"

    res = build_mixed_plan(
        _FillClient(), targets=["AAA"], managed={}, pos_info={}, bar="2026-07-21T18:00:00Z",
        signal_audits={}, policy=ExecPolicy(target_notional=5000.0), route_fn=_option_route,
        ref_price_fn=lambda ticker: 25.0, verbose=False,
    )
    # premium=$2.00 -> $200/contract -> 5000/200 = 25 contracts
    assert res.plan == [("AAA260814C00025000", "buy", 25, "entry", "option")]
    assert res.new_managed["AAA"]["contracts"] == 25


def test_build_mixed_plan_respects_a_custom_target_notional():
    def _equity_route(client, ticker, px, **kwargs):
        return "equity", None, "not_optionable"

    res = build_mixed_plan(
        _FillClient(), targets=["AAA"], managed={}, pos_info={}, bar="2026-07-21T18:00:00Z",
        signal_audits={}, policy=ExecPolicy(target_notional=1000.0), route_fn=_equity_route,
        ref_price_fn=lambda ticker: 10.0, verbose=False,
    )
    assert res.plan == [("AAA", "buy", 100, "entry", "equity")]  # 1000/10 = 100 shares
