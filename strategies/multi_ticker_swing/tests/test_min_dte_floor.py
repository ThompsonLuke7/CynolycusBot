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


def test_min_dte_floor_is_at_least_two_weeks():
    """A floor under 14 days systematically buys less time than the move needs."""
    assert runner._MIN_DTE_DAYS >= 14, (
        "DTE floor regressed below 14 days; 2-DTE contracts captured only 19% of "
        "achievable +10% moves in the live-fill study"
    )


def test_min_dte_floor_is_the_validated_value():
    assert runner._MIN_DTE_DAYS == 21, (
        "21d is the validated efficiency knee (74% move capture). Changing it is a "
        "policy decision -- update this test and the rationale comment together."
    )


def test_min_dte_floor_is_not_absurdly_long():
    """Guard the other direction: an over-long floor drifts away from the signal horizon."""
    assert runner._MIN_DTE_DAYS <= 60


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
