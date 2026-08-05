"""select_option must return a usable *mark*, not just a number.

Task 23, increment 4. The governed path needs `bid`, `ask`, and `quote_at` to
build an `OptionLeg`; today `select_option` computes bid/ask and throws them
away, and never records when the quote was observed.

The timestamp is the point. Without it there is no way to tell a live
two-sided market from a stale one, and this project has already published and
retracted an entire options study built on prices that turned out to be stale
trade prints (research/options_experiment/10_RETRACTION_option_pnl_invalid.md).
A quote we cannot date is not a quote we may trade on, and a clock reading is
not a substitute for one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from core.nervous_system.execution.options.quotes import OptionQuote
from core.option_liquidity import ContractLiquidity
from signals.meta_context.meta_ranker import options_exec as ox


OCC = "ABC260717C00010000"
QUOTE_AT = "2026-07-06T19:59:58.123456Z"
NOW_ET = datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc)


class _FakeClient:
    """Serves one call contract with a configurable snapshot quote."""

    def __init__(
        self,
        *,
        bid: float = 1.0,
        ask: float = 1.5,
        delta: float = 0.45,
        quote_at: str | None = QUOTE_AT,
        snapshot_quote: bool = True,
        fallback_quote: dict | None = None,
    ) -> None:
        self.bid, self.ask, self.delta = bid, ask, delta
        self.quote_at = quote_at
        self.snapshot_quote = snapshot_quote
        self.fallback_quote = fallback_quote
        self.quote_calls = 0

    def get_option_contracts(self, **_):
        return {
            "option_contracts": [
                {
                    "symbol": OCC,
                    "strike_price": "10",
                    "underlying_symbol": "ABC",
                    "type": "call",
                }
            ]
        }

    def get_option_snapshots(self, *_a, **_k):
        quote: dict = {}
        if self.snapshot_quote:
            quote = {"bp": self.bid, "ap": self.ask}
            if self.quote_at is not None:
                quote["t"] = self.quote_at
        return {OCC: {"greeks": {"delta": self.delta}, "latestQuote": quote}}

    def get_option_quotes(self, **_):
        self.quote_calls += 1
        return {"quotes": {OCC: self.fallback_quote}} if self.fallback_quote else {}


@pytest.fixture(autouse=True)
def liquidity(monkeypatch):
    monkeypatch.setattr(
        ox,
        "contract_liquidity",
        lambda _u, *, expiry, strike, option_type="C": ContractLiquidity(
            open_interest=1000, volume=500, source="test"
        ),
    )


def _select(client, per_name_usd: float = 5000.0):
    return ox.select_option(
        client, "ABC", 100.0, per_name_usd, roll_trading_days=5, now_et=NOW_ET
    )


# ---------------------------------------------------------------------------
# The mark is returned, and it is dated
# ---------------------------------------------------------------------------


def test_the_selection_returns_a_validated_two_sided_quote() -> None:
    order, reason = _select(_FakeClient())

    assert reason == "ok"
    quote = order["quote"]
    assert isinstance(quote, OptionQuote)
    assert quote.bid == Decimal("1.0")
    assert quote.ask == Decimal("1.5")
    assert quote.symbol == OCC


def test_the_quote_carries_the_time_the_market_was_observed() -> None:
    order, _ = _select(_FakeClient())

    assert order["quote"].quote_at == datetime(
        2026, 7, 6, 19, 59, 58, 123456, tzinfo=timezone.utc
    )


def test_the_quote_time_is_never_the_current_clock() -> None:
    """A clock reading dresses an undated quote up as a fresh one. That is the
    precise failure mode behind the retracted options study.
    """

    order, _ = _select(_FakeClient())

    assert order["quote"].quote_at != NOW_ET
    assert abs((order["quote"].quote_at - datetime.now(timezone.utc)).total_seconds()) > 60


def test_an_undated_quote_is_refused_rather_than_stamped_with_now() -> None:
    order, reason = _select(_FakeClient(quote_at=None))

    assert order is None
    assert reason == "no_quote_timestamp"


def test_the_mid_is_the_two_sided_midpoint_not_a_trade_print() -> None:
    order, _ = _select(_FakeClient(bid=1.0, ask=1.5))

    assert order["quote"].mid == Decimal("1.25")
    assert order["mid"] == pytest.approx(1.25)
    assert order["quote"].last_trade_price is None


def test_the_delta_is_carried_onto_the_quote() -> None:
    order, _ = _select(_FakeClient(delta=0.45))

    assert order["quote"].delta == pytest.approx(Decimal("0.45"))


def test_open_interest_and_volume_reach_the_quote() -> None:
    order, _ = _select(_FakeClient())

    assert order["quote"].open_interest == 1000
    assert order["quote"].volume == 500


# ---------------------------------------------------------------------------
# Degenerate markets
# ---------------------------------------------------------------------------


def test_a_crossed_market_is_refused() -> None:
    order, reason = _select(_FakeClient(bid=2.0, ask=1.0))

    assert order is None
    assert reason == "crossed_quote"


def test_a_one_sided_market_falls_back_and_then_refuses() -> None:
    client = _FakeClient(bid=0.0, ask=1.5, fallback_quote=None)
    order, reason = _select(client)

    assert order is None
    assert reason == "no_quote"
    assert client.quote_calls == 1, "the snapshot quote was unusable, so refetch once"


def test_the_fallback_quote_must_also_be_dated() -> None:
    client = _FakeClient(
        snapshot_quote=False, fallback_quote={"bp": 1.0, "ap": 1.5}
    )
    order, reason = _select(client)

    assert order is None
    assert reason == "no_quote_timestamp"


def test_a_dated_fallback_quote_is_accepted() -> None:
    client = _FakeClient(
        snapshot_quote=False,
        fallback_quote={"bp": 1.0, "ap": 1.5, "t": QUOTE_AT},
    )
    order, reason = _select(client)

    assert reason == "ok"
    assert order["quote"].quote_at == datetime(
        2026, 7, 6, 19, 59, 58, 123456, tzinfo=timezone.utc
    )


# ---------------------------------------------------------------------------
# Existing behaviour is unchanged
# ---------------------------------------------------------------------------


def test_the_legacy_fields_are_preserved_exactly() -> None:
    """The runner and the audit trail already read these; the quote is added
    alongside them, not in place of them.
    """

    order, _ = _select(_FakeClient(bid=1.0, ask=1.5))

    assert order["occ"] == OCC
    assert order["limit"] == 1.5
    assert order["mid"] == pytest.approx(1.25)
    assert order["strike"] == 10.0
    assert order["contracts"] == 40
    assert order["open_interest"] == 1000
    assert order["volume"] == 500


def test_a_budget_below_one_contract_is_still_refused() -> None:
    order, reason = _select(_FakeClient(), per_name_usd=10.0)

    assert order is None
    assert reason == "budget_lt_1_contract"


def test_a_naive_quote_timestamp_is_refused_not_assumed_utc() -> None:
    """A timestamp with no zone has no defined instant. Assuming UTC would
    silently shift a quote by hours and could make a stale market pass a
    freshness check.
    """

    order, reason = _select(_FakeClient(quote_at="2026-07-06T19:59:58.123456"))

    assert order is None
    assert reason == "no_quote_timestamp"


def test_a_non_utc_quote_timestamp_is_normalised_not_dropped() -> None:
    """An offset-bearing timestamp is a real instant and must be kept."""

    order, reason = _select(_FakeClient(quote_at="2026-07-06T15:59:58.123456-04:00"))

    assert reason == "ok"
    assert order["quote"].quote_at == datetime(
        2026, 7, 6, 19, 59, 58, 123456, tzinfo=timezone.utc
    )


def test_an_unparseable_quote_timestamp_is_refused() -> None:
    order, reason = _select(_FakeClient(quote_at="not-a-timestamp"))

    assert order is None
    assert reason == "no_quote_timestamp"
