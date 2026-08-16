"""Guards against the 2026-08-14 sub-minute churn.

Six SPY round trips that session, all losers, four of them open for under 35
seconds (one for 2s). Three independent defects combined:

  1. The option-value exit bracket ran BEFORE `hold_ready` was computed, so
     `meta_min_hold_bars` — which exists to prevent exactly this — never applied
     to it.
  2. A broker reconcile seeded `_meta_side_bars_held` at `meta_min_hold_bars`
     for any position it saw as new, pre-satisfying the guard even where it did
     apply. A position we opened seconds earlier looks new, because the policy's
     own prev_qty is still 0 until the fill registers.
  3. Exit decisions were marked at the BID while entries fill at the ASK, so
     every position began underwater by the full spread. Every option-value rule
     tests `current_profit_pct <= 0`, so all of them were armed at the fill.

Plus the open ladder chased the limit above the offer (0.74 -> 0.79) to force a
fill on a signal that had already flipped.
"""
from __future__ import annotations

from strategies.spy_intraday.Policy.order_policy import (
    OptionOrderPolicy,
    OptionOrderPolicyConfig,
)

SYMBOL = "SPY260814C00776000"


def _policy(**cfg_kw) -> OptionOrderPolicy:
    cfg = OptionOrderPolicyConfig(submit_orders=True, **cfg_kw)
    return OptionOrderPolicy(cfg)


def test_exit_decision_is_marked_at_the_mid_not_the_bid():
    """A fresh fill must not read as an instant loss just because of the spread."""
    cfg = OptionOrderPolicyConfig()
    assert cfg.option_exit_decision_quote_mode == "mid"
    # Order pricing for a close still has to be marketable.
    assert cfg.option_exit_quote_mode == "bid"

    quote = {"bid_price": 0.66, "ask_price": 0.72}
    bid = OptionOrderPolicy._quote_price(quote, mode="bid")
    mid = OptionOrderPolicy._quote_price(quote, mode="mid")
    # Bought at the 0.69 mid: the bid mark shows -4.3%, the mid mark shows flat.
    assert (bid - 0.69) / 0.69 < -0.04
    assert abs((mid - 0.69) / 0.69) < 1e-9


def test_reconcile_does_not_pre_satisfy_the_min_hold_guard():
    pol = _policy()
    pol._long_contracts = 0
    pol._long_symbol = None

    pol._apply_broker_position_state(
        broker_state={"long_contracts": 1, "short_contracts": 0,
                      "long_symbol": SYMBOL, "short_symbol": None,
                      "avg_entry_price_long": 0.69, "avg_entry_price_short": None},
        preserve_bars_held=True,
        local_ts=None,
    )

    # Was meta_min_hold_bars (2) — instantly exit-eligible. Must start at 0.
    assert pol._meta_side_bars_held["long"] == 0
    assert pol._meta_side_bars_held["long"] < pol.cfg.meta_min_hold_bars


def test_reconcile_still_preserves_bars_held_for_an_existing_position():
    """The counter must keep advancing for a position we were already holding."""
    pol = _policy()
    pol._long_contracts = 1
    pol._long_symbol = SYMBOL
    pol._meta_side_bars_held["long"] = 7

    pol._apply_broker_position_state(
        broker_state={"long_contracts": 1, "short_contracts": 0,
                      "long_symbol": SYMBOL, "short_symbol": None,
                      "avg_entry_price_long": 0.69, "avg_entry_price_short": None},
        preserve_bars_held=True,
        local_ts=None,
    )

    assert pol._meta_side_bars_held["long"] == 7


class _NeverFillsClient:
    """Accepts every order without filling, so the ladder walks every rung."""

    def __init__(self, *, bid: float, ask: float):
        self.quote = {"bid_price": bid, "ask_price": ask}
        self.submitted: list[dict] = []

    def get_option_quotes(self, symbols, limit=1):
        return {"quotes": {symbols: self.quote}}

    def submit_option_order(self, **kwargs):
        self.submitted.append(kwargs)
        return {"id": f"oid-{len(self.submitted)}", "status": "new"}

    def get_order(self, order_id):
        return {"id": order_id, "status": "new", "filled_avg_price": None}

    def cancel_order(self, order_id):
        return {"id": order_id, "status": "canceled"}


def test_open_ladder_never_prices_through_the_ask():
    """Chasing above the offer buys nothing — the ask is the whole book.

    2026-08-14 walked SPY260814C00776000 from 0.74 to 0.79 across five attempts
    against a 0.76 offer, then closed the position 28 seconds later.
    """
    pol = _policy(max_resubmit_attempts=4, price_mode="mid", verify_submitted_orders=False)
    client = _NeverFillsClient(bid=0.72, ask=0.76)
    pol._client = client

    pol._submit_order(symbol=SYMBOL, side="buy", intent="open", qty=1,
                      logger=lambda *_: None)

    limits = [float(o["limit_price"]) for o in client.submitted]
    assert limits, "ladder should have submitted at least once"
    assert max(limits) <= 0.76, f"priced through the ask: {limits}"
    # Without the clamp the mid-anchored ladder reaches 0.78 by the fifth rung.
    assert max(limits) < 0.74 + 4 * 0.01


def test_min_hold_blocks_a_discretionary_option_exit(monkeypatch):
    """A time-decay exit on a position held 0 bars must be refused."""
    pol = _policy(meta_min_hold_bars=2, meta_replay_compatible_mode=True)
    pol._long_contracts = 1
    pol._long_symbol = SYMBOL
    pol._meta_side_bars_held["long"] = 0

    def _fake_hit(*, side, logger, closed_bar=None, local_ts=None):
        pol._meta_side_reason[side] = "option_time_decay"
        return True

    monkeypatch.setattr(pol, "_option_value_exit_hit", _fake_hit)
    monkeypatch.setattr(pol, "_ohlc_exit_bar_is_after_entry", lambda **kw: False)

    qty = pol._target_contracts_for_side(
        side="long", closed_bar={}, close=776.0, high=776.5, low=775.5,
        atr=1.2, local_ts=None, allow_exits=True, logger=lambda *_: None)

    assert qty > 0, "position should be held, not closed, before min-hold is met"
    assert pol._meta_side_reason["long"] == "min_hold_blocks:option_time_decay"


def test_min_hold_never_blocks_a_real_stop(monkeypatch):
    pol = _policy(meta_min_hold_bars=2, meta_replay_compatible_mode=True)
    pol._long_contracts = 1
    pol._long_symbol = SYMBOL
    pol._meta_side_bars_held["long"] = 0

    def _fake_hit(*, side, logger, closed_bar=None, local_ts=None):
        pol._meta_side_reason[side] = "option_stop_loss"
        return True

    monkeypatch.setattr(pol, "_option_value_exit_hit", _fake_hit)
    monkeypatch.setattr(pol, "_ohlc_exit_bar_is_after_entry", lambda **kw: False)

    qty = pol._target_contracts_for_side(
        side="long", closed_bar={}, close=776.0, high=776.5, low=775.5,
        atr=1.2, local_ts=None, allow_exits=True, logger=lambda *_: None)

    assert qty == 0, "a stop must never be delayed by the min-hold guard"


def test_min_hold_allows_the_exit_once_the_position_has_been_held(monkeypatch):
    pol = _policy(meta_min_hold_bars=2, meta_replay_compatible_mode=True)
    pol._long_contracts = 1
    pol._long_symbol = SYMBOL
    pol._meta_side_bars_held["long"] = 2

    def _fake_hit(*, side, logger, closed_bar=None, local_ts=None):
        pol._meta_side_reason[side] = "option_time_decay"
        return True

    monkeypatch.setattr(pol, "_option_value_exit_hit", _fake_hit)
    monkeypatch.setattr(pol, "_ohlc_exit_bar_is_after_entry", lambda **kw: False)

    qty = pol._target_contracts_for_side(
        side="long", closed_bar={}, close=776.0, high=776.5, low=775.5,
        atr=1.2, local_ts=None, allow_exits=True, logger=lambda *_: None)

    assert qty == 0


def test_decision_reason_is_exposed_for_the_audit_stream():
    """Without this the audit records that a position closed, not why."""
    pol = _policy()
    pol._meta_side_reason["long"] = "option_time_decay"
    snap = pol.snapshot_state()
    assert snap["long_decision_reason"] == "option_time_decay"
    assert snap["option_exit_decision_quote_mode"] == "mid"
