"""Momentum's governed execution path.

Three properties matter most and each is pinned here:

1. **Off by default.** The direct path is what has been running. Two paths
   submitting the same signal is the worst outcome available.
2. **No direct-broker fallback.** An unreachable governed path QUEUES the plan.
   Falling back to a direct call would reintroduce the exact bypass the
   cutover removes -- and losing the plan is how a 9-order plan vanished on
   2026-08-20 without leaving a record that it existed.
3. **Exits are restored, entries are dropped.** A refused exit means the
   broker never closed the position, so it must stay managed; a refused entry
   means nothing was bought, so it must not.
"""

from __future__ import annotations

import types
from decimal import Decimal

import pytest

from core.governed_plan_execution import gateway_verdict, submit_plan_via_router
from strategies.momentum_expansion.live.gateway_execution import (
    PRIMARY_SCORE,
    SCORE_FIELDS,
    STRATEGY_ID,
    momentum_intent_config,
    momentum_scores_by_ticker,
)


class _Result:
    def __init__(self, outcome=None, reason_code=None, detail=None, broker_order_id=None):
        self.outcome = outcome
        self.reason_code = reason_code
        self.detail = detail
        self.broker_order_id = broker_order_id


class _Row:
    def __init__(self, symbol, side, quantity, *, outcome=None, refusal=None,
                 policy_vetoes=(), reason_codes=("entry",)):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.refusal = refusal
        self.policy_vetoes = policy_vetoes
        self.outcome = types.SimpleNamespace(execution_result=outcome)
        self.intent = types.SimpleNamespace(reason_codes=reason_codes)


class _Router:
    """Captures what route() was asked to do and replays scripted rows."""

    def __init__(self, rows):
        self._rows = rows
        self.calls: list[dict] = []

    def route(self, plan, **kwargs):
        self.calls.append({"plan": list(plan), **kwargs})
        on_row = kwargs.get("on_row")
        for row in self._rows:
            if on_row is not None:
                on_row(row)
        return tuple(self._rows)


# ---------------------------------------------------------------------------
# Intent vocabulary
# ---------------------------------------------------------------------------


def test_momentum_intent_config_uses_its_own_score_vocabulary() -> None:
    config = momentum_intent_config()
    assert config.strategy_id == STRATEGY_ID
    assert config.primary_score == PRIMARY_SCORE == "expansion_score"
    assert config.score_fields == SCORE_FIELDS
    assert config.primary_score in config.score_fields


def test_meta_vocabulary_is_unchanged() -> None:
    """Momentum must not have moved Meta's defaults underneath it."""

    from signals.meta_context.meta_ranker.nervous_system_adapter import MetaIntentConfig

    meta = MetaIntentConfig()
    assert meta.primary_score == "s_combo"
    assert meta.score_fields == ("s_combo", "s_upside", "s_quality")


def test_primary_score_must_be_in_score_fields() -> None:
    from signals.meta_context.meta_ranker.nervous_system_adapter import MetaIntentConfig

    with pytest.raises(ValueError, match="primary_score must appear"):
        MetaIntentConfig(primary_score="nope", score_fields=("s_combo",))


def test_a_name_without_a_score_is_omitted_not_fabricated() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [
            {"ticker": "AMD", "expansion_score": 0.91, "rank": 1},
            {"ticker": "NVDA", "expansion_score": float("nan"), "rank": 2},
        ]
    )
    scores = momentum_scores_by_ticker(frame)
    assert "AMD" in scores and scores["AMD"]["expansion_score"] == 0.91
    # Opening risk we cannot explain is never acceptable: NVDA is absent, and
    # the adapter will refuse it rather than invent a score.
    assert "NVDA" not in scores


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------


def test_refusal_is_no_exposure_and_names_the_vetoes() -> None:
    row = _Row(
        "AMD", "buy", 10,
        refusal=types.SimpleNamespace(value="POLICY_VETO"),
        policy_vetoes=("PORTFOLIO_MAX_THEME_NOTIONAL_BREACH",),
    )
    verdict, broker_id, detail = gateway_verdict(row)
    assert verdict == "NO_EXPOSURE"
    assert broker_id is None
    assert "PORTFOLIO_MAX_THEME_NOTIONAL_BREACH" in detail


def test_an_unreached_gateway_is_no_exposure_not_uncertain() -> None:
    verdict, _broker_id, _detail = gateway_verdict(_Row("AMD", "buy", 10))
    assert verdict == "NO_EXPOSURE"


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def _submitted_row(symbol, side, qty, broker_id="abc"):
    from core.nervous_system.execution.gateway import ExecutionOutcome

    return _Row(
        symbol, side, qty,
        outcome=_Result(outcome=ExecutionOutcome.SUBMITTED, broker_order_id=broker_id),
    )


def test_a_refused_exit_is_restored_to_managed_state(monkeypatch) -> None:
    import core.governed_plan_execution as mod

    monkeypatch.setattr(mod, "mark_entry_unconfirmed", lambda *a, **k: None)
    monkeypatch.setattr(mod, "track_exit_submission", lambda *a, **k: None)
    dropped: list = []
    monkeypatch.setattr(mod, "drop_failed_entry", lambda m, s: dropped.append(s))

    previous = {"qty": 100, "entry": 10.0}
    new_managed: dict = {}
    exit_context = {"AMD": ("AMD", previous)}
    router = _Router([_Row("AMD", "sell", 100, refusal=types.SimpleNamespace(value="REFUSED"))])
    dispositions: dict = {}

    submit_plan_via_router(
        router, [("AMD", "sell", 100, "exit", "equity")],
        module="momentum_expansion", client=None, bar=None,
        new_managed=new_managed, exit_context=exit_context, pos_lookup={},
        scores_by_ticker={}, dispositions=dispositions,
    )
    # The broker never closed it, so the position must stay tracked.
    assert new_managed["AMD"] == previous
    assert dispositions["AMD"] == "refused"
    assert dropped == []


def test_a_refused_entry_is_dropped(monkeypatch) -> None:
    import core.governed_plan_execution as mod

    monkeypatch.setattr(mod, "mark_entry_unconfirmed", lambda *a, **k: None)
    monkeypatch.setattr(mod, "track_exit_submission", lambda *a, **k: None)
    dropped: list = []
    monkeypatch.setattr(mod, "drop_failed_entry", lambda m, s: dropped.append(s))

    router = _Router([_Row("AMD", "buy", 10, refusal=types.SimpleNamespace(value="REFUSED"))])
    dispositions: dict = {}
    submit_plan_via_router(
        router, [("AMD", "buy", 10, "entry", "equity")],
        module="momentum_expansion", client=None, bar=None,
        new_managed={}, exit_context={}, pos_lookup={},
        scores_by_ticker={}, dispositions=dispositions,
    )
    assert dropped == ["AMD"]
    assert dispositions["AMD"] == "refused"


def test_uncertain_keeps_the_claim_and_books_no_fill(monkeypatch) -> None:
    """Releasing the claim is how a sibling module adopts a real position."""

    import core.governed_plan_execution as mod
    from core.nervous_system.execution.gateway import ExecutionOutcome

    monkeypatch.setattr(mod, "mark_entry_unconfirmed", lambda *a, **k: None)
    tracked: list = []
    monkeypatch.setattr(mod, "track_exit_submission",
                        lambda *a, **k: tracked.append(k.get("item")))
    dropped: list = []
    monkeypatch.setattr(mod, "drop_failed_entry", lambda m, s: dropped.append(s))

    previous = {"qty": 100}
    new_managed: dict = {}
    router = _Router([
        _Row("AMD", "sell", 100,
             outcome=_Result(outcome=ExecutionOutcome.RECONCILIATION_REQUIRED))
    ])
    dispositions: dict = {}
    submit_plan_via_router(
        router, [("AMD", "sell", 100, "exit", "equity")],
        module="momentum_expansion", client=None, bar=None,
        new_managed=new_managed, exit_context={"AMD": ("AMD", previous)},
        pos_lookup={}, scores_by_ticker={}, dispositions=dispositions,
    )
    assert new_managed["AMD"] == previous
    assert dispositions["AMD"] == "uncertain"
    assert tracked == [], "no fill may be booked against an unconfirmed order"


def test_state_is_persisted_after_every_order(monkeypatch) -> None:
    """A sibling's reconcile must never find a fresh position missing."""

    import core.governed_plan_execution as mod

    monkeypatch.setattr(mod, "mark_entry_unconfirmed", lambda *a, **k: None)
    monkeypatch.setattr(mod, "track_exit_submission", lambda *a, **k: None)
    monkeypatch.setattr(mod, "drop_failed_entry", lambda *a, **k: None)

    saves: list[int] = []
    rows = [_submitted_row("AMD", "buy", 10), _submitted_row("NVDA", "buy", 5)]
    submit_plan_via_router(
        _Router(rows),
        [("AMD", "buy", 10, "entry", "equity"), ("NVDA", "buy", 5, "entry", "equity")],
        module="momentum_expansion", client=None, bar=None,
        new_managed={}, exit_context={}, pos_lookup={}, scores_by_ticker={},
        persist_managed=lambda: saves.append(1),
    )
    assert len(saves) == 2, "state must be saved per order, not once per plan"


def test_router_receives_position_keys_for_every_row() -> None:
    rows = [_submitted_row("AMD", "buy", 10)]
    router = _Router(rows)
    submit_plan_via_router(
        router, [("AMD", "buy", 10, "entry", "equity")],
        module="momentum_expansion", client=None, bar=None,
        new_managed={}, exit_context={}, pos_lookup={}, scores_by_ticker={"AMD": {}},
    )
    call = router.calls[0]
    assert call["position_keys"] == {"AMD": "paper:AMD"}
    assert call["submit"] is True


# ---------------------------------------------------------------------------
# The runner switch
# ---------------------------------------------------------------------------


def test_runner_is_ungoverned_by_default() -> None:
    from strategies.momentum_expansion.live.runner import MomentumLiveRunner

    assert MomentumLiveRunner().governed is False
    assert MomentumLiveRunner(governed=True).governed is True


def test_governed_is_paper_only() -> None:
    """The router refuses PRODUCTION_LIVE; the CLI says so before building one."""

    import inspect

    from strategies.momentum_expansion.live import runner as mod

    source = inspect.getsource(mod.main)
    assert "--governed is paper-only" in source
    assert "args.governed and args.live" in source


def test_governed_branch_never_calls_execute_plan_directly() -> None:
    """No direct-broker fallback: that is the bypass the cutover removes."""

    import inspect

    from strategies.momentum_expansion.live import runner as mod

    source = inspect.getsource(mod.MomentumLiveRunner._submit_via_gateway)
    assert "execute_plan" not in source
    assert "submit_plan_via_router" in source
    # Ticker states must be published before routing, or every entry is vetoed
    # as SNAPSHOT_REQUIRED_STATE_MISSING.
    assert source.index("publish_momentum_ticker_states") < source.index(
        "submit_plan_via_router"
    )
