"""Coverage for the next-startup action queue.

The queue exists because the live server is hand-started and is often down when
a decision gets made. Two properties matter most: an order entry must never fire
into a closed market, and a live-account entry must never run on a paper-account
server (or vice versa) by accident.
"""
from __future__ import annotations

import json

import pytest

from core import startup_queue as sq


@pytest.fixture
def qpath(tmp_path):
    return tmp_path / "startup_queue.json"


class _Client:
    def __init__(self, positions=None, fail=False):
        self._positions = positions or []
        self.fail = fail
        self.orders: list[dict] = []

    def get_positions(self):
        return self._positions

    def submit_order(self, **kwargs):
        if self.fail:
            raise RuntimeError("broker rejected")
        self.orders.append(kwargs)
        return {"id": "order-1", "status": "accepted"}


def _run(qpath, client, *, is_open=True, allow_live=False, readiness=None):
    return sq.run_pending(
        client_factory=lambda _a: client,
        readiness_runner=readiness,
        market_is_open=lambda: is_open,
        allow_live=allow_live,
        path=qpath,
    )


def test_enqueue_then_list(qpath):
    sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "SNDK", "qty": "all"}, path=qpath)
    entries = sq.load(qpath)
    assert len(entries) == 1
    assert entries[0]["kind"] == "close_position"
    assert entries[0]["status"] == "pending"
    assert entries[0]["account"] == "paper"


def test_close_all_flattens_the_whole_position(qpath):
    sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "SNDK", "qty": "all"}, path=qpath)
    client = _Client([{"symbol": "SNDK", "qty": "100", "side": "long"}])
    summary = _run(qpath, client)
    assert summary["done"] == 1
    assert client.orders == [{
        "symbol": "SNDK", "qty": 100, "side": "sell",
        "order_type": "market", "time_in_force": "day",
    }]
    assert sq.load(qpath)[0]["status"] == "done"


def test_partial_qty_is_capped_at_the_held_amount(qpath):
    sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "SNDK", "qty": 500}, path=qpath)
    client = _Client([{"symbol": "SNDK", "qty": "100", "side": "long"}])
    _run(qpath, client)
    assert client.orders[0]["qty"] == 100


def test_short_position_closes_with_a_buy(qpath):
    sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "DIA", "qty": "all"}, path=qpath)
    client = _Client([{"symbol": "DIA", "qty": "-40", "side": "short"}])
    _run(qpath, client)
    assert client.orders[0] == {
        "symbol": "DIA", "qty": 40, "side": "buy",
        "order_type": "market", "time_in_force": "day",
    }


def test_orders_do_not_fire_into_a_closed_market(qpath):
    sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "SNDK", "qty": "all"}, path=qpath)
    client = _Client([{"symbol": "SNDK", "qty": "100", "side": "long"}])
    summary = _run(qpath, client, is_open=False)
    assert client.orders == []
    assert summary["deferred"] == 1
    assert sq.load(qpath)[0]["status"] == "pending"  # retried next boot


def test_a_deferred_close_fires_on_a_later_drain_once_the_market_opens(qpath):
    """The boot drain defers when the market is shut; a later drain must fire it.

    This is what the server's market-hours re-drain slots rely on. Before those
    existed the queue was only drained once at boot, so a server started
    overnight deferred a queued liquidate and then never revisited it -- the
    entry sat pending all day with the position still open.
    """
    sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "SNDK", "qty": "all"}, path=qpath)
    client = _Client([{"symbol": "SNDK", "qty": "100", "side": "long"}])

    # Boot, overnight: deferred, untouched, and crucially not counted as an attempt.
    assert _run(qpath, client, is_open=False)["deferred"] == 1
    assert client.orders == []
    assert sq.load(qpath)[0]["attempts"] == 0

    # 09:35 slot the same session: same entry, now actually executed.
    summary = _run(qpath, client, is_open=True)
    assert summary["done"] == 1
    assert [o["symbol"] for o in client.orders] == ["SNDK"]
    assert sq.load(qpath)[0]["status"] == sq.STATUS_DONE


def test_redraining_after_completion_does_not_resubmit(qpath):
    """The 13:00 backstop must be a no-op once the 09:35 slot has closed it."""
    sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "SNDK", "qty": "all"}, path=qpath)
    client = _Client([{"symbol": "SNDK", "qty": "100", "side": "long"}])
    _run(qpath, client, is_open=True)
    assert len(client.orders) == 1

    _run(qpath, client, is_open=True)
    assert len(client.orders) == 1  # not double-submitted


def test_live_entry_is_skipped_on_a_paper_server(qpath):
    sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "SNDK", "qty": "all"},
               account="live", path=qpath)
    client = _Client([{"symbol": "SNDK", "qty": "100", "side": "long"}])
    summary = _run(qpath, client, allow_live=False)
    assert client.orders == []
    assert summary["skipped"] == 1
    assert sq.load(qpath)[0]["status"] == "pending"


def test_already_flat_counts_as_satisfied(qpath):
    sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "SNDK", "qty": "all"}, path=qpath)
    summary = _run(qpath, _Client([]))
    assert summary["done"] == 1
    assert sq.load(qpath)[0]["result"]["reason"] == "not_held"


def test_a_failing_entry_is_recorded_and_does_not_block_the_rest(qpath):
    sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "SNDK", "qty": "all"}, path=qpath)
    sq.enqueue(sq.KIND_NOTE, note="still here", path=qpath)
    summary = _run(qpath, _Client([{"symbol": "SNDK", "qty": "100", "side": "long"}], fail=True))
    entries = sq.load(qpath)
    assert summary["failed"] == 1 and summary["done"] == 1
    assert entries[0]["status"] == "failed"
    assert "broker rejected" in entries[0]["result"]["error"]
    assert entries[1]["status"] == "done"


def test_done_entries_are_not_rerun(qpath):
    sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "SNDK", "qty": "all"}, path=qpath)
    client = _Client([{"symbol": "SNDK", "qty": "100", "side": "long"}])
    _run(qpath, client)
    _run(qpath, client)
    assert len(client.orders) == 1


def test_readiness_entry_calls_the_runner(qpath):
    sq.enqueue(sq.KIND_DATA_READINESS, note="stamp stale", path=qpath)
    calls = []
    summary = _run(qpath, _Client(), readiness=lambda: calls.append(1) or "ok")
    assert summary["done"] == 1
    assert calls == [1]


def test_cancelled_entry_never_runs(qpath):
    entry = sq.enqueue(sq.KIND_CLOSE_POSITION, params={"symbol": "SNDK"}, path=qpath)
    entries = sq.load(qpath)
    entries[0]["status"] = sq.STATUS_CANCELLED
    sq.save(entries, qpath)
    client = _Client([{"symbol": "SNDK", "qty": "100", "side": "long"}])
    _run(qpath, client)
    assert client.orders == []
    assert sq.load(qpath)[0]["id"] == entry["id"]


def test_unreadable_queue_does_not_raise(qpath):
    qpath.parent.mkdir(parents=True, exist_ok=True)
    qpath.write_text("{not json")
    assert sq.load(qpath) == []


def test_save_is_atomic_and_readable(qpath):
    sq.enqueue(sq.KIND_NOTE, note="hello", path=qpath)
    payload = json.loads(qpath.read_text())
    assert payload["entries"][0]["note"] == "hello"
    assert "updated" in payload


def test_unknown_kind_is_rejected_at_enqueue(qpath):
    with pytest.raises(ValueError):
        sq.enqueue("rm_rf", path=qpath)


def test_unknown_account_is_rejected_at_enqueue(qpath):
    with pytest.raises(ValueError):
        sq.enqueue(sq.KIND_NOTE, account="margin", path=qpath)
