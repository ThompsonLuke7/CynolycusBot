"""HTF and Meta hand-roll their own `_execute` instead of calling execute_plan.

That duplication is why the 2026-08-11 ownership fix nearly shipped broken: the
shared `execute_plan` was corrected, but HTF's and Meta's copies still called
`defer_exits_if_opg_unavailable(module, bar, plan, limits)` with no managed
state — and HTF is the module the VSH incident actually happened in.

These tests assert the contract at the call site rather than the behaviour,
because there is no cheap way to drive those runners end to end. They are a
tripwire for the next person who edits one copy and not the others.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Every module that submits a 4H-family order plan.
EXEC_PATHS = [
    REPO / "core/live_4h_exec.py",
    REPO / "strategies/multi_ticker_swing_htf/live/runner.py",
    REPO / "signals/meta_context/meta_ranker/live_runner.py",
]


def _calls(path: Path, func: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == func]


@pytest.mark.parametrize("path", EXEC_PATHS, ids=lambda p: p.name)
def test_every_deferral_call_passes_managed_state(path):
    """A deferred exit must keep the position claimed, in every exec path."""
    calls = _calls(path, "defer_exits_if_opg_unavailable")
    assert calls, f"{path} no longer calls the deferral — update this test"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        missing = {"new_managed", "exit_context"} - kwargs
        assert not missing, (
            f"{path.relative_to(REPO)}:{call.lineno} defers exits without {sorted(missing)}. "
            "The position would be held at the broker and claimed by nobody, which is "
            "how Swing adopted HTF's VSH on 2026-08-11."
        )


@pytest.mark.parametrize("path", EXEC_PATHS, ids=lambda p: p.name)
def test_every_exec_path_flags_unconfirmed_entries(path):
    """An accepted-but-unfilled entry must not read as a held position."""
    assert _calls(path, "mark_entry_unconfirmed"), (
        f"{path.relative_to(REPO)} submits entries without calling "
        "mark_entry_unconfirmed; managed state will claim positions the broker "
        "does not hold (Dealer Ranker, 2026-08-11: 11 claimed vs 7 held)."
    )


@pytest.mark.parametrize("path", EXEC_PATHS, ids=lambda p: p.name)
def test_every_exec_path_writes_the_realized_pnl_ledger(path):
    assert _calls(path, "record_exit_realized_pnl"), (
        f"{path.relative_to(REPO)} submits exits without writing the realized-PnL "
        "ledger; closed trades would go unrecorded."
    )


# Every module that can DEFER an exit into the pending queue. Dealer Ranker
# reaches the queue through the shared `execute_plan` rather than its own
# `_execute`, which is why it is absent from EXEC_PATHS but present here.
QUEUE_OWNING_RUNNERS = [
    REPO / "strategies/multi_ticker_swing_htf/live/runner.py",
    REPO / "signals/meta_context/meta_ranker/live_runner.py",
    REPO / "strategies/momentum_expansion/live/runner.py",
    REPO / "strategies/dealer_positioning/live_ranked_options.py",
]


@pytest.mark.parametrize("path", QUEUE_OWNING_RUNNERS, ids=lambda p: p.name)
def test_a_module_that_defers_exits_also_flushes_them(path):
    """Writing to the queue without draining it strands the exit forever.

    Dealer Ranker deferred exits into `pending_exit_orders.json` but only ever
    called `submit_pending_open_entries`, so a `trail_-35%` sell for
    RBRK260828C00100000 queued on 2026-08-28 was still sitting in the file
    eleven days later — past the contract's own expiry, with nothing retrying
    it and nothing reporting it. The queue is only a deferral if something
    flushes it; otherwise it is a silent drop.
    """

    assert _calls(path, "submit_pending_exit_orders"), (
        f"{path.relative_to(REPO)} can queue a deferred exit but never calls "
        "submit_pending_exit_orders, so anything it defers is stranded."
    )


def test_the_pending_exit_flush_also_writes_the_ledger():
    """The flush is a fourth exit path and had no ledger call at all until
    2026-08-12 — the AMAT/VSH rows that went missing on 08-11."""
    src = (REPO / "core/live_4h_exec.py").read_text()
    tree = ast.parse(src)
    flush = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "submit_pending_exit_orders")
    assert any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "record_exit_realized_pnl"
               for n in ast.walk(flush))
