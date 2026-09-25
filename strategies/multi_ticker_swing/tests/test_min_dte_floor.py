"""Regression test for the option DTE floor.

The floor was 0 ("nearest listed expiry, including 0DTE/1DTE weeklies"), which put
the median entry on a 2-DTE contract. Measured on this module's own 575 real fills,
a 2-day floor captures only 19% of the +10% underlying moves that actually occur,
versus 74% at 21 days -- and even losing trades went on to a 9.2% median favorable
move by 30d. See research/options_experiment/13_dte_floor_and_regime_rules.md.

These tests pin the floor so it cannot silently regress to a near-dated default.
"""
from __future__ import annotations

from datetime import date

import pytest

from strategies.multi_ticker_swing.live import runner


def test_min_dte_floor_never_returns_to_same_week_expiry():
    """The floor may be short, but never 0-2 DTE again.

    The original regression was a 0 floor putting the median entry on a 2-DTE
    contract. The 2026-09 revision shortens the floor deliberately, but a
    same-week expiry re-introduces the assignment and OTM-liquidity problems the
    floor exists to prevent -- and 0-2 DTE is where the 8c fixed half-spread is
    most punitive (58% round-trip on a $0.28 premium).
    """
    assert runner._MIN_DTE_DAYS >= 3, (
        "DTE floor regressed to same-week expiry; that is the 2-DTE default the "
        "floor was introduced to remove"
    )


def test_min_dte_floor_is_the_current_policy_value():
    assert runner._MIN_DTE_DAYS == 7, (
        "7d is the current policy: this module holds a median 0 days (p75 1d) against "
        "a formerly 24-day median DTE, using 0% of the contract's life and producing "
        "0.0% of trades above +100%. Changing it is a policy decision -- update this "
        "test and the rationale comment in runner.py together. "
        "See research/regime_coverage_2026-09-21/exp_d_ci.py."
    )


def test_min_dte_floor_matches_the_modules_actual_holding_period():
    """The floor must stay in the same order of magnitude as the hold.

    This is the failure the 21d floor represented: a floor chosen from how long
    the MOVE takes, on a module that exits within a day. Whatever the floor is,
    it should be days-not-weeks while the p75 hold is 1 day.
    """
    assert runner._MIN_DTE_DAYS <= 14, (
        "floor drifted back toward a multi-week contract on a module whose p75 "
        "holding period is 1 calendar day"
    )


def test_min_dte_floor_is_not_absurdly_long():
    """Guard the other direction: an over-long floor drifts away from the signal horizon."""
    assert runner._MIN_DTE_DAYS <= 60


def test_live_selection_floor_is_the_constant_not_the_monthly_helper():
    """The LIVE path floors on `_MIN_DTE_DAYS` against broker-listed expiries.

    `_next_monthly_expiry` returns third Fridays only and is NOT on the live path
    (it has no non-test caller). Live selection queries the contracts endpoint from
    `ref_date + _MIN_DTE_DAYS` and takes the nearest listed expiry, so with weeklies
    the floor is what actually determines DTE. Pinning that here because lowering
    the constant only produces short-dated contracts if this remains true.
    """
    import inspect

    src = inspect.getsource(runner)
    assert "_next_monthly_expiry(" not in src.split("def _next_monthly_expiry")[0], (
        "_next_monthly_expiry gained a caller before its definition; the live "
        "expiry path may have changed"
    )
    assert "ref_date + timedelta(days=_MIN_DTE_DAYS)" in src, (
        "live contract discovery no longer starts at ref_date + _MIN_DTE_DAYS; "
        "the floor may no longer bind the expiry actually chosen"
    )


def test_next_monthly_expiry_respects_the_floor():
    """The monthly-expiry helper must skip an expiry that is inside the floor."""
    ref = date(2026, 5, 11)          # May monthly expiry (3rd Fri) = 2026-05-15, only 4d away
    exp = runner._next_monthly_expiry(ref)
    assert (exp - ref).days >= runner._MIN_DTE_DAYS, (
        f"_next_monthly_expiry returned {exp} which is inside the {runner._MIN_DTE_DAYS}d floor"
    )


@pytest.mark.parametrize("ref", [
    date(2026, 1, 2), date(2026, 3, 16), date(2026, 6, 30), date(2026, 11, 20),
])
def test_next_monthly_expiry_respects_floor_across_the_year(ref):
    exp = runner._next_monthly_expiry(ref)
    assert (exp - ref).days >= runner._MIN_DTE_DAYS
    assert exp.weekday() == 4, "monthly expiry must be a Friday"


# ---------------------------------------------------------------------------
# Long-only gate
# ---------------------------------------------------------------------------

def test_short_entries_are_disabled():
    """Puts lost -$32,928 across 299 real fills vs +$5,827 on 258 calls, in every
    regime. The module is options-only, so a short signal means buying puts."""
    assert runner._ALLOW_SHORT_ENTRIES is False, (
        "long-only gate was re-enabled for shorts; see 12_dte_and_put_call_study.md"
    )


def test_long_only_gate_is_independent_of_challenger_policy():
    """The gate must hold even with the challenger policy off, which is its whole
    point -- enabling that policy would also activate unrelated blocked-ticker and
    blocked-time-bucket rules."""
    assert runner._LIVE_OPTION_FILTER_POLICY != runner._CHALLENGER_OPTION_FILTER_POLICY, (
        "test premise changed: challenger policy is now on, so this no longer proves independence"
    )
    assert not runner._challenger_policy_enabled()
    assert runner._ALLOW_SHORT_ENTRIES is False
