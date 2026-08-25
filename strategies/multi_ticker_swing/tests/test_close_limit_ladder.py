"""Coverage for the forced-liquidation close ladder.

2026-08-05: swing sold 159 VALE260828C00015000 at a $0.01 limit into a
0.01 x 0.74 market on `restored_unknown_loss_cut`. The ladder anchored on the
BID, mid was only consulted when the bid was *missing*, so the bid ticking up
from 0.00 to 0.01 became the first rung and filled instantly. The broker marked
the position at $3,021; it realized $159.

The ladder now anchors on the mid and relaxes its floor across ~60s passes.
"""
from __future__ import annotations

import pytest

from strategies.multi_ticker_swing.live.position_manager import (
    _LIQUIDATION_FLOOR_BY_PASS,
    _close_limit_ladder,
)

# The live VALE quote at 09:50 ET: bid a penny, ask 0.74, mid 0.375.
VALE_QUOTE = {"bid": 0.01, "ask": 0.74, "mid": 0.375, "spread": 0.73,
              "spread_pct_mid": 1.95}


def _ladder(*, reason, base, bid, close_pass=1, quote=None, attempts=5):
    return _close_limit_ladder(
        base_limit=base, close_bid=bid, quote_meta=quote or VALE_QUOTE,
        reason=reason, attempts=attempts, close_pass=close_pass,
    )


def test_penny_bid_no_longer_sets_the_first_rung():
    """The regression itself: a one-cent bid must not price the exit."""
    rungs = _ladder(reason="restored_unknown_loss_cut", base=0.37, bid=0.01)
    assert rungs[0] == 0.37
    assert min(rungs) > 0.01


def test_first_pass_floors_at_85_percent_of_mid():
    rungs = _ladder(reason="restored_unknown_loss_cut", base=0.37, bid=0.01)
    assert min(rungs) == pytest.approx(0.37 * _LIQUIDATION_FLOOR_BY_PASS[0], abs=0.005)
    assert rungs == sorted(rungs, reverse=True)


@pytest.mark.parametrize("close_pass, floor_frac", list(enumerate(_LIQUIDATION_FLOOR_BY_PASS, start=1)))
def test_floor_relaxes_one_step_per_pass(close_pass, floor_frac):
    rungs = _ladder(reason="restored_unknown_loss_cut", base=0.37, bid=0.01,
                    close_pass=close_pass)
    # Within one tick: the floor is rounded to a submittable option price.
    assert min(rungs) == pytest.approx(0.37 * floor_frac, abs=0.011)


def test_broker_mark_is_reachable_before_the_schedule_runs_out():
    """VALE's real mark was 0.19. Some rung must reach it while still patient."""
    reachable = [
        p for pass_no in (1, 2, 3)
        for p in _ladder(reason="restored_unknown_loss_cut", base=0.37, bid=0.01,
                         close_pass=pass_no)
        if p <= 0.19
    ]
    assert reachable, "ladder never offers the position at its broker mark"


def test_bid_only_reached_after_the_schedule_is_exhausted():
    patient = [
        _ladder(reason="restored_unknown_loss_cut", base=0.37, bid=0.01, close_pass=p)
        for p in range(1, len(_LIQUIDATION_FLOOR_BY_PASS) + 1)
    ]
    assert all(min(r) > 0.01 for r in patient)
    capitulated = _ladder(reason="restored_unknown_loss_cut", base=0.37, bid=0.01,
                          close_pass=len(_LIQUIDATION_FLOOR_BY_PASS) + 1)
    assert min(capitulated) == 0.01


def test_a_bid_above_the_scheduled_floor_is_taken():
    """A healthy bid is a real price; patience must not skip past it."""
    quote = {"bid": 0.34, "ask": 0.40, "mid": 0.37, "spread": 0.06,
             "spread_pct_mid": 0.16}
    rungs = _ladder(reason="restored_unknown_loss_cut", base=0.37, bid=0.34, quote=quote)
    assert min(rungs) == pytest.approx(0.34, abs=0.005)


def test_penny_rung_needs_a_wide_spread_not_just_a_late_pass():
    """A 0.02 bid on a tight book is a real price — do not dump at a penny."""
    tight = {"bid": 0.02, "ask": 0.04, "mid": 0.03, "spread": 0.02,
             "spread_pct_mid": 0.67}
    rungs = _ladder(reason="restored_unknown_loss_cut", base=0.03, bid=0.02,
                    close_pass=99, quote=tight)
    assert min(rungs) == 0.02
    wide = _ladder(reason="restored_unknown_loss_cut", base=0.37, bid=0.01,
                   close_pass=99)
    assert min(wide) == 0.01


def test_empty_book_still_capitulates_to_a_penny():
    """No bid at all (the IOT-style stuck contract) must still reach 0.01."""
    quote = {"bid": None, "ask": 0.74, "mid": 0.37, "spread": None,
             "spread_pct_mid": 2.0}
    rungs = _ladder(reason="restored_unknown_loss_cut", base=0.37, bid=float("nan"),
                    close_pass=99, quote=quote)
    assert min(rungs) == 0.01


def test_ordinary_exits_still_rest_at_the_bid():
    """trail/take_profit are not liquidations — resting at the bid is how they fill."""
    quote = {"bid": 22.79, "ask": 24.10, "mid": 23.445, "spread": 1.31,
             "spread_pct_mid": 0.056}
    rungs = _close_limit_ladder(base_limit=22.79, close_bid=22.79, quote_meta=quote,
                                reason="trail", attempts=5, close_pass=1)
    assert rungs[0] == pytest.approx(22.79)
    assert rungs == sorted(rungs, reverse=True)


def test_ladder_respects_the_attempt_budget():
    for pass_no in (1, 3, 5, 99):
        rungs = _ladder(reason="restored_unknown_loss_cut", base=0.37, bid=0.01,
                        close_pass=pass_no, attempts=5)
        assert 1 <= len(rungs) <= 5
        assert len(rungs) == len(set(rungs))


# --- a ladder that cannot reach the book cannot fill (2026-08-18) -------------
#
# TGT260911C00157500 quoted 3.65 x 5.92 on a restored_unknown_loss_cut. The
# pass-1 floor of 0.85 * 4.79 = 4.07 sat 42 cents ABOVE the best bid, so all
# five rungs were unfillable by construction; the position left on the market
# fallback at 3.70 after burning ~23 seconds and five cancel/resubmit round
# trips. The floor schedule exists to avoid dumping into an ABSENT bid (VALE:
# 0.01 against a 0.375 mid), not a merely WIDE one.

TGT_QUOTE = {"bid": 3.65, "ask": 5.92, "mid": 4.785, "spread": 2.27,
             "spread_pct_mid": 0.4744}


def test_a_wide_book_with_a_credible_bid_is_reachable_on_the_first_pass():
    rungs = _ladder(reason="restored_unknown_loss_cut", base=4.79, bid=3.65,
                    quote=TGT_QUOTE)
    assert min(rungs) == pytest.approx(3.65, abs=0.005), "lowest rung must touch the bid"
    assert rungs[0] == pytest.approx(4.79), "still starts at the mid"
    assert rungs == sorted(rungs, reverse=True)


def test_an_absent_bid_is_still_treated_as_patiently_as_before():
    """VALE's 0.01 bid is 2.7% of the mid — not credible, keep the schedule."""
    rungs = _ladder(reason="restored_unknown_loss_cut", base=0.37, bid=0.01)
    assert min(rungs) == pytest.approx(0.37 * _LIQUIDATION_FLOOR_BY_PASS[0], abs=0.005)
    assert min(rungs) > 0.01


@pytest.mark.parametrize("bid,base,reachable", [
    (3.65, 4.79, True),    # 76% of mid — credible
    (2.40, 4.79, True),    # 50% of mid — exactly at the threshold
    (2.30, 4.79, False),   # 48% of mid — below it, stay patient
    (0.01, 0.37, False),   # the VALE book
])
def test_bid_credibility_threshold(bid, base, reachable):
    quote = {"bid": bid, "ask": base * 2 - bid, "mid": base,
             "spread": (base - bid) * 2, "spread_pct_mid": (base - bid) * 2 / base}
    rungs = _ladder(reason="restored_unknown_loss_cut", base=base, bid=bid, quote=quote)
    assert (min(rungs) == pytest.approx(bid, abs=0.005)) is reachable
