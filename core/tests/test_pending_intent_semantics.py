"""After-close pending-entry semantics (Task 23).

Two rules, both about failing in the safe direction.

*Unknown market state fails closed for entries.* "I could not tell whether the
market is open" is not the same as "the market is open". Treating it as open
sends after-hours entries that will be rejected, and worse, does so silently.

*A retained entry is never silently deleted.* The queue is the only record that
a decision is still waiting. Dropping an entry that was skipped — for readiness,
for a failed submit — loses the decision entirely, with nothing to show it ever
existed.
"""

from __future__ import annotations

import json

import pytest

from core.live_4h_exec import (
    defer_entries_if_market_closed,
    pending_open_path,
    submit_pending_open_entries,
)


BAR = "2026-08-03T20:00:00Z"


@pytest.fixture
def readiness_ok(monkeypatch):
    """Let entries past the readiness gate so these tests exercise the queue.

    Without this the gate blocks every entry in a bare test environment, and a
    test claiming to check submission would silently be checking readiness.
    """

    import core.live_4h_exec as engine

    monkeypatch.setattr(
        engine, "filter_entry_orders_for_readiness",
        lambda plan, **_: (plan, {}, "ok"),
    )


def _plan() -> list[tuple]:
    return [
        ("AMD", "buy", 10, "entry", "equity"),
        ("MSFT", "sell", 20, "horizon", "equity"),
    ]


def test_an_unknown_market_calendar_defers_entries_rather_than_sending_them(
    tmp_path, monkeypatch
) -> None:
    """Fail closed: an entry we cannot time is queued, not fired blind."""

    import core.calendar as calendar

    def _explode(*_a, **_k):
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr(calendar, "is_market_open_now", _explode)

    kept = defer_entries_if_market_closed(
        "m", BAR, _plan(), {}, {}, ledger_root=str(tmp_path)
    )

    assert [item[0] for item in kept] == ["MSFT"], "the exit still goes"
    queued = json.loads(pending_open_path("m", str(tmp_path)).read_text())["entries"]
    assert [entry["order_symbol"] for entry in queued] == ["AMD"]


def test_an_unknown_calendar_never_blocks_an_exit(tmp_path, monkeypatch) -> None:
    """Failing closed applies to opening risk only. A risk-reducing exit must
    still go, or an unreadable calendar could trap a position.
    """

    import core.calendar as calendar

    monkeypatch.setattr(
        calendar, "is_market_open_now", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError())
    )

    kept = defer_entries_if_market_closed(
        "m", BAR, [("MSFT", "sell", 20, "horizon", "equity")], {}, {},
        ledger_root=str(tmp_path),
    )

    assert len(kept) == 1


class _RefusingClient:
    def submit_order(self, **_):
        raise RuntimeError("broker said no")

    def submit_option_order(self, **_):
        raise RuntimeError("broker said no")


def _queue(tmp_path, entries: list[dict]) -> None:
    path = pending_open_path("m", str(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"updated": "x", "entries": entries}))


def _entry(ticker: str = "AMD") -> dict:
    return {
        "order_symbol": ticker, "side": "buy", "qty": 10, "route": "equity",
        "limit": None, "ticker": ticker, "bar": BAR, "managed": {"qty": 10},
    }


def test_an_entry_that_failed_to_submit_is_retained_not_dropped(tmp_path, readiness_ok) -> None:
    """The decision is still live. Deleting it loses it with no trace."""

    _queue(tmp_path, [_entry()])

    result = submit_pending_open_entries(
        _RefusingClient(), "m", ["AMD"],
        equity_tif_fn=lambda: "day", ledger_root=str(tmp_path),
    )

    assert result["count"] == 0
    retained = json.loads(pending_open_path("m", str(tmp_path)).read_text())["entries"]
    assert [entry["order_symbol"] for entry in retained] == ["AMD"]


def test_an_entry_no_longer_in_the_top_k_is_terminal_and_removed(tmp_path) -> None:
    """This one is genuinely finished: the rank it depended on is gone, so it
    is not waiting for anything and retaining it would resubmit it forever.
    """

    _queue(tmp_path, [_entry("AMD")])

    submit_pending_open_entries(
        _RefusingClient(), "m", ["NVDA"],
        equity_tif_fn=lambda: "day", ledger_root=str(tmp_path),
    )

    path = pending_open_path("m", str(tmp_path))
    remaining = json.loads(path.read_text())["entries"] if path.exists() else []
    assert remaining == []


class _AcceptingClient:
    def submit_order(self, **_):
        return {"id": "ok-1"}

    def submit_option_order(self, **_):
        return {"id": "ok-1"}


def test_a_submitted_entry_leaves_the_queue(tmp_path, readiness_ok) -> None:
    _queue(tmp_path, [_entry()])

    result = submit_pending_open_entries(
        _AcceptingClient(), "m", ["AMD"],
        equity_tif_fn=lambda: "day", ledger_root=str(tmp_path),
    )

    assert result["count"] == 1
    path = pending_open_path("m", str(tmp_path))
    remaining = json.loads(path.read_text())["entries"] if path.exists() else []
    assert remaining == []


def test_a_mixed_flush_retains_only_the_unfinished_entry(tmp_path, readiness_ok) -> None:
    _queue(tmp_path, [_entry("AMD"), _entry("NVDA")])

    class _Selective:
        def submit_order(self, *, symbol, **_):
            if symbol == "NVDA":
                raise RuntimeError("broker said no")
            return {"id": "ok-1"}

        submit_option_order = submit_order

    submit_pending_open_entries(
        _Selective(), "m", ["AMD", "NVDA"],
        equity_tif_fn=lambda: "day", ledger_root=str(tmp_path),
    )

    retained = json.loads(pending_open_path("m", str(tmp_path)).read_text())["entries"]
    assert [entry["order_symbol"] for entry in retained] == ["NVDA"]


def test_a_readiness_failure_retains_the_entry_rather_than_deleting_it(tmp_path) -> None:
    """The decision is still valid; only the data around it is not ready yet.
    Deleting it would discard a decision because of a transient gate.
    """

    _queue(tmp_path, [_entry()])

    result = submit_pending_open_entries(
        _AcceptingClient(), "m", ["AMD"],
        equity_tif_fn=lambda: "day", ledger_root=str(tmp_path),
    )

    assert result["count"] == 0
    retained = json.loads(pending_open_path("m", str(tmp_path)).read_text())["entries"]
    assert [entry["order_symbol"] for entry in retained] == ["AMD"]


# ---------------------------------------------------------------------------
# The injected submitter
# ---------------------------------------------------------------------------


class _ForbiddenClient:
    """Any direct broker call here is the bypass the injection exists to stop."""

    def submit_order(self, **_):
        raise AssertionError("the direct broker path must not be used")

    submit_option_order = submit_order


def test_an_injected_submitter_replaces_the_direct_broker_call(tmp_path, readiness_ok) -> None:
    """Meta routes through the gateway while the other modules on this shared
    engine keep their existing path; injection is what makes both true at once.
    """

    _queue(tmp_path, [_entry()])
    seen: list[dict] = []

    def _submit(**kwargs):
        seen.append(kwargs)
        return {"id": "governed-1"}

    result = submit_pending_open_entries(
        _ForbiddenClient(), "m", ["AMD"],
        equity_tif_fn=lambda: "day", ledger_root=str(tmp_path),
        submit_fn=_submit,
    )

    assert result["count"] == 1
    assert seen == [
        {"symbol": "AMD", "side": "buy", "qty": 10, "route": "equity", "limit": None}
    ]


def test_the_exit_ladder_uses_an_injected_submitter_without_retrying() -> None:
    """The governed path owns its own laddering and market fallback, so running
    the legacy retry loop on top would submit the same close several times.
    """

    from core.live_4h_exec import submit_option_exit_with_ladder

    calls: list[dict] = []

    def _submit(**kwargs):
        calls.append(kwargs)
        return {"id": "governed-exit"}

    resp = submit_option_exit_with_ladder(
        _ForbiddenClient(), symbol="AMD260821C00200000", qty=3, submit_fn=_submit
    )

    assert resp == {"id": "governed-exit"}
    assert len(calls) == 1, "one governed close, not a ladder of duplicates"
