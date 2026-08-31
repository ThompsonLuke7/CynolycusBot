"""The pre-open flush must produce the same governed intents as the 4H pass.

`governed_submitter` is the submit_fn the shared 4H engine calls during the
pre-open flush, for both queued entries and queued exits. It was the one
governed path with no test, and it showed: it handed the router an empty
`scores_by_ticker`, so every queued ENTRY hit the adapter's "refusing to open a
position that cannot be explained" guard and was recorded as a skip. The flush
looked like it worked -- it reported skips, not errors -- while placing nothing.

Two rules this pins down.

*An entry carries its evidence.* The adapter is right to refuse an unexplained
open; the caller is wrong to withhold the explanation it already has. The
runner scores every name before the flush, so the scores exist.

*A symbol maps to its ticker explicitly.* A queued option record carries its
ticker, so the router must not be left to infer it from an OCC root.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from signals.meta_context.meta_ranker import live_runner


RUNNER_SOURCE = Path(live_runner.__file__)


BAR = datetime(2026, 8, 14, 20, 0, tzinfo=timezone.utc)
SCORES = {"AMD": {"s_combo": 0.94, "s_quality": 0.71, "s_upside": 0.62}}


class _CapturingRouter:
    """Stands in for the assembled gateway; records what it was asked to route."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def route(self, plan, **kwargs):
        self.calls.append({"plan": list(plan), **kwargs})
        return (
            SimpleNamespace(
                refusal=None,
                submitted=True,
                outcome=SimpleNamespace(
                    execution_result=SimpleNamespace(broker_order_id="broker-1")
                ),
            ),
        )


REFERENCE_BARS = {"AMD": 160.0}


@pytest.fixture
def router(monkeypatch) -> _CapturingRouter:
    captured = _CapturingRouter()
    monkeypatch.setattr(live_runner, "build_router", lambda **_: captured)
    monkeypatch.setattr(live_runner, "intent_config", lambda *_a, **_k: object())

    def _fake_ref_price(ticker: str, *, decision_bar):
        try:
            return REFERENCE_BARS[ticker]
        except KeyError:
            raise ValueError(f"no reference-bar Parquet exists for {ticker}") from None

    monkeypatch.setattr(live_runner, "_ref_price", _fake_ref_price)
    return captured


def _args() -> SimpleNamespace:
    return SimpleNamespace(mode="options", quality_floor=0.5, target_notional=5000.0)


# ---------------------------------------------------------------------------
# Entries carry their evidence
# ---------------------------------------------------------------------------


def test_a_queued_entry_reaches_the_router_with_its_scores(router) -> None:
    """Without this the adapter refuses the open and the flush places nothing."""

    submit = live_runner.governed_submitter(_args(), bar=BAR, scores_by_ticker=SCORES)
    submit(symbol="AMD", side="buy", qty=10, route="equity")

    assert router.calls[0]["scores_by_ticker"] == SCORES


def test_an_option_entry_maps_its_occ_symbol_to_its_ticker(router) -> None:
    """The queued record knows the ticker; the router must not guess it."""

    submit = live_runner.governed_submitter(_args(), bar=BAR, scores_by_ticker=SCORES)
    submit(
        symbol="AMD260821C00160000", side="buy", qty=3, route="option", ticker="AMD",
    )

    assert router.calls[0]["ticker_by_symbol"] == {"AMD260821C00160000": "AMD"}


def test_scores_are_passed_for_every_name_not_just_the_one_submitted(router) -> None:
    """Each call routes one order, but the scored frame is shared. Trimming it
    per call would be work with no benefit and one more place to get wrong."""

    submit = live_runner.governed_submitter(_args(), bar=BAR, scores_by_ticker=SCORES)
    submit(symbol="AMD", side="buy", qty=10, route="equity")

    assert set(router.calls[0]["scores_by_ticker"]) == {"AMD"}


def test_an_unscored_name_is_still_offered_to_the_router(router) -> None:
    """The refusal belongs to the adapter, which states the reason precisely.
    Screening it out here would hide an unexplained open behind a generic skip.
    """

    submit = live_runner.governed_submitter(_args(), bar=BAR, scores_by_ticker=SCORES)
    submit(symbol="NVDA", side="buy", qty=4, route="equity")

    assert router.calls[0]["plan"] == [("NVDA", "buy", 4, "pending_open", "equity")]


# ---------------------------------------------------------------------------
# Entries carry a price to be sized from
# ---------------------------------------------------------------------------


def test_a_queued_equity_entry_reaches_the_router_with_a_reference_price(router) -> None:
    """The twin of the scores defect, one layer over.

    ``_route_one`` sizes an ENTRY as shares_for_notional(price, budget) and
    refuses NO_REFERENCE_PRICE without one, so a submitter built with an empty
    price map rejected *every* queued entry unconditionally. On 2026-08-25 that
    refused all four of meta_ranker's queued 8/24 entries (submitted=0
    skipped=10) while HTF and Momentum flushed the same market minutes later.
    """

    submit = live_runner.governed_submitter(_args(), bar=BAR, scores_by_ticker=SCORES)
    submit(symbol="AMD", side="buy", qty=10, route="equity", ticker="AMD")

    assert router.calls[0]["reference_prices"] == {"AMD": 160.0}


def test_an_unpriceable_equity_entry_is_refused_by_the_router_not_guessed(router) -> None:
    """A name with no readable decision bar contributes no price at all.

    Substituting a last print or a stale close would put a fabricated quantity
    on a real order; the correct outcome is the router's NO_REFERENCE_PRICE.
    """

    submit = live_runner.governed_submitter(_args(), bar=BAR, scores_by_ticker=SCORES)
    submit(symbol="NVDA", side="buy", qty=4, route="equity", ticker="NVDA")

    assert router.calls[0]["reference_prices"] == {}


def test_an_option_entry_is_not_gated_on_an_underlying_bar(router) -> None:
    """An option entry is sized in contracts off the quoted ask, so it must not
    be made conditional on a lookup its size never uses.
    """

    submit = live_runner.governed_submitter(_args(), bar=BAR, scores_by_ticker=SCORES)
    submit(
        symbol="ZZZZ260821C00160000", side="buy", qty=3, route="option", ticker="ZZZZ",
    )

    assert router.calls[0]["reference_prices"] == {}


def test_a_reduction_is_never_gated_on_a_reference_price(router) -> None:
    """A sell carries its own exact quantity. Looking a price up for it would
    make leaving a position depend on data only an entry needs.
    """

    submit = live_runner.governed_submitter(_args(), bar=BAR, scores_by_ticker=SCORES)
    submit(
        symbol="NVDA", side="sell", qty=10, route="equity",
        reason="horizon", full_exit=True, ticker="NVDA",
    )

    assert router.calls[0]["reference_prices"] == {}
    assert router.calls[0]["plan"] == [("NVDA", "sell", 10, "horizon", "equity")]


# ---------------------------------------------------------------------------
# Exits are unchanged by the entry fix
# ---------------------------------------------------------------------------


def test_a_full_exit_still_routes_as_an_exit(router) -> None:
    submit = live_runner.governed_submitter(_args(), bar=BAR, scores_by_ticker=SCORES)
    submit(
        symbol="AMD", side="sell", qty=10, route="equity",
        reason="take_profit", full_exit=True,
    )

    call = router.calls[0]
    assert call["plan"] == [("AMD", "sell", 10, "take_profit", "equity")]
    assert "AMD" in call["exit_context"]


def test_a_trim_is_not_recorded_as_an_exit(router) -> None:
    submit = live_runner.governed_submitter(_args(), bar=BAR, scores_by_ticker=SCORES)
    submit(
        symbol="AMD", side="sell", qty=4, route="equity",
        reason="scale_out", full_exit=False,
    )

    assert router.calls[0]["exit_context"] == {}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_scores_default_to_empty_rather_than_failing_to_build(router) -> None:
    """A caller that genuinely has no scores still gets a submitter; the refusal
    then comes from the adapter, naming the ticker."""

    submit = live_runner.governed_submitter(_args(), bar=BAR)
    submit(symbol="AMD", side="sell", qty=10, route="equity", full_exit=True)

    assert router.calls[0]["scores_by_ticker"] == {}


def test_every_submitter_the_runner_builds_is_given_the_scores() -> None:
    """The call site, not just the function.

    A static scan because the flush lives inside main(), behind a broker client
    and an account read. Every test above passed with the runner building its
    submitter as `governed_submitter(args, bar=bar)` -- the original defect --
    because they all construct their own. This is the assertion that fails when
    the wiring is dropped, which is where the bug actually was.
    """

    tree = ast.parse(RUNNER_SOURCE.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "governed_submitter"
    ]

    assert calls, "the runner must build its submitter through governed_submitter"
    for call in calls:
        passed = {keyword.arg for keyword in call.keywords}
        assert "scores_by_ticker" in passed, (
            f"governed_submitter at line {call.lineno} is built without scores; "
            "every queued entry it submits would be refused as unexplained"
        )


def test_a_refusal_raises_so_the_caller_records_a_skip(monkeypatch) -> None:
    """Never a silent fallback to a direct broker call."""

    class _Refusing:
        def route(self, plan, **_):
            return (
                SimpleNamespace(
                    refusal=SimpleNamespace(value="POLICY_BLOCKED"),
                    submitted=False,
                    outcome=None,
                ),
            )

    monkeypatch.setattr(live_runner, "build_router", lambda **_: _Refusing())
    monkeypatch.setattr(live_runner, "intent_config", lambda *_a, **_k: object())

    submit = live_runner.governed_submitter(_args(), bar=BAR, scores_by_ticker=SCORES)

    with pytest.raises(RuntimeError, match="POLICY_BLOCKED"):
        submit(symbol="AMD", side="buy", qty=10, route="equity")
