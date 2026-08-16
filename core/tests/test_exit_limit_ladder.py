"""Coverage for the priced exit fallback.

2026-07-24: Dealer Ranker's -50% stop on IOT260724C00036000 was submitted as a
market sell and rejected HTTP 403 "order has been rejected due to no available
quote for symbol. please reenter with a limit". The position was restored to
managed state, never retried, and the contract expired that afternoon. An exit
has to be priced when the book is empty.
"""
from __future__ import annotations

import pytest

from core.live_4h_exec import (
    _exit_limit_ladder,
    submit_option_exit_with_ladder,
)


class _Client:
    def __init__(self, *, bid=None, accept_market=False, accept_limit_at=None):
        self.bid = bid
        self.accept_market = accept_market
        self.accept_limit_at = accept_limit_at
        self.attempts: list[tuple[str, float | None]] = []

    def get_option_quotes(self, symbols):  # noqa: ARG002
        if self.bid is None:
            return {"quotes": {}}
        return {"quotes": {symbols: {"bp": self.bid, "ap": self.bid + 0.1}}}

    def submit_option_order(self, *, symbol, qty, side, order_type, time_in_force,  # noqa: ARG002
                            limit_price=None):
        self.attempts.append((order_type, limit_price))
        if order_type == "market":
            if self.accept_market:
                return {"id": "market-ok"}
            raise RuntimeError(
                'HTTP Error 403: {"code":40310000,"message":"order has been rejected '
                'due to no available quote for symbol. please reenter with a limit"}'
            )
        if self.accept_limit_at is not None and limit_price == self.accept_limit_at:
            return {"id": f"limit-ok@{limit_price}"}
        raise RuntimeError("limit rejected")


def _no_sleep(_seconds):
    return None


def test_market_exit_is_still_preferred_when_it_works():
    client = _Client(accept_market=True)
    resp = submit_option_exit_with_ladder(client, symbol="IOT260724C00036000", qty=1,
                                          sleep_fn=_no_sleep)
    assert resp == {"id": "market-ok"}
    assert client.attempts == [("market", None)]


def test_rejected_market_exit_reprices_as_a_limit():
    """The client quotes 0.40 x 0.50, so the ladder opens at the 0.45 mid."""
    client = _Client(bid=0.40, accept_limit_at=0.40)
    resp = submit_option_exit_with_ladder(client, symbol="IOT260724C00036000", qty=1,
                                          sleep_fn=_no_sleep)
    assert resp == {"id": "limit-ok@0.4"}
    assert client.attempts[0] == ("market", None)
    assert client.attempts[1] == ("limit", 0.45)


def test_ladder_starts_at_the_mid_and_pauses_between_rungs():
    client = _Client(bid=1.00, accept_limit_at=1.00)
    calls: list[float] = []
    submit_option_exit_with_ladder(client, symbol="X", qty=2, sleep_fn=calls.append)
    limits = [limit for kind, limit in client.attempts if kind == "limit"]
    assert limits == [1.05, 1.02, 1.00]  # mid -> halfway -> bid
    assert limits == sorted(limits, reverse=True)
    assert calls == [pytest.approx(1.5), pytest.approx(1.5)]


def test_ladder_never_opens_below_the_bid():
    """2026-08-05 VALE: a bid-anchored ladder made a penny bid the first rung.

    The exit must open at the mid and treat the bid as a floor, so a collapsed
    bid costs at most the last rung instead of pricing the whole exit.
    """
    rungs = _exit_limit_ladder(0.01, 0.375)  # bid 0.01, ask 0.74
    assert rungs[0] == pytest.approx(0.38, abs=0.011)
    assert rungs[0] > 0.01
    assert rungs == sorted(rungs, reverse=True)


def test_empty_book_falls_back_to_the_one_cent_rung():
    """The IOT case exactly: no quote at all, so there is nothing to price off."""
    client = _Client(bid=None, accept_limit_at=0.01)
    resp = submit_option_exit_with_ladder(client, symbol="IOT260724C00036000", qty=1,
                                          sleep_fn=_no_sleep)
    assert resp == {"id": "limit-ok@0.01"}
    assert [limit for kind, limit in client.attempts if kind == "limit"] == [0.01]


def test_a_genuinely_stuck_exit_still_raises():
    client = _Client(bid=0.50)  # accepts nothing
    with pytest.raises(RuntimeError):
        submit_option_exit_with_ladder(client, symbol="X", qty=1, sleep_fn=_no_sleep)


@pytest.mark.parametrize(
    "bid, mid, expected",
    [
        (None, None, [0.01]),
        (0.01, None, [0.01]),
        (0.005, None, [0.01]),
        # Bid only (no ask to build a mid): stay marketable, bounded 20% under.
        (1.00, None, [1.00, 0.90, 0.80]),
        # Mid known: open at the mid, floor at the bid.
        (1.00, 1.05, [1.05, 1.02, 1.00]),
        # Empty book with a mark: still reaches a penny so nothing gets stuck.
        (None, 0.40, [0.40, 0.21, 0.01]),
    ],
)
def test_ladder_shape(bid, mid, expected):
    assert _exit_limit_ladder(bid, mid) == expected
