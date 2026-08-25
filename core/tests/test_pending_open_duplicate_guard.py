"""A queued entry must not be sent twice when an order is already working.

``pos_lookup`` only sees FILLED positions. Two modules flushing their deferred
queues minutes apart can therefore both look at an unheld symbol and both send
an order for it.

That is not hypothetical. On 2026-08-24 the Meta Ranker's 09:35 pre-open flush
correctly found PSIG, RZLT and PATH260918C00018000 unheld; HTF filled PSIG and
RZLT at 09:37 and momentum filled the PATH call at 09:45; and because a
POLICY_VETO had made Meta's submits fail rather than succeed, all three stayed
queued at identical quantities — sizing is deterministic given a shared target
notional, so two modules picking the same name produce the same order. Only the
veto stopped the account buying them twice. PLUG x2273 was queued for the same
bar in both Meta and HTF at the close that day.
"""

from __future__ import annotations

import json

import pytest

from core.live_4h_exec import (
    pending_open_path,
    submit_pending_open_entries,
    working_order_symbols,
)


def _last_session_bar() -> str:
    from datetime import datetime, timezone

    from core.calendar import prev_trading_day

    return f"{prev_trading_day(datetime.now(timezone.utc).date()).isoformat()}T20:00:00Z"


BAR = _last_session_bar()


@pytest.fixture
def readiness_ok(monkeypatch):
    """Let entries past the readiness gate so these tests exercise the queue."""

    import core.live_4h_exec as engine

    monkeypatch.setattr(
        engine, "filter_entry_orders_for_readiness",
        lambda plan, **_: (plan, {}, "ok"),
    )


class _Client:
    """Records what actually reached the broker."""

    def __init__(self, open_orders=(), orders_raise=False):
        # Passed through as-is, not coerced: one test hands this a non-list on
        # purpose, and list()-ing it here would silently repair the bad shape.
        self._open_orders = open_orders
        self._orders_raise = orders_raise
        self.sent: list[str] = []

    def get_orders(self, **_):
        if self._orders_raise:
            raise RuntimeError("orders endpoint down")
        return self._open_orders

    def submit_order(self, **kw):
        self.sent.append(kw["symbol"])
        return {"id": f"ok-{len(self.sent)}"}

    def submit_option_order(self, **kw):
        self.sent.append(kw["symbol"])
        return {"id": f"ok-{len(self.sent)}"}


class _NoOrdersApiClient:
    """A client predating get_orders — the guard must degrade, not crash."""

    def __init__(self):
        self.sent: list[str] = []

    def submit_order(self, **kw):
        self.sent.append(kw["symbol"])
        return {"id": "ok-1"}

    def submit_option_order(self, **kw):
        self.sent.append(kw["symbol"])
        return {"id": "ok-1"}


def _queue(tmp_path, entries: list[dict]) -> None:
    path = pending_open_path("m", str(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"updated": "x", "entries": entries}))


def _entry(symbol: str = "PLUG", ticker: str | None = None, qty: int = 2273) -> dict:
    return {
        "order_symbol": symbol, "side": "buy", "qty": qty, "route": "equity",
        "limit": None, "ticker": ticker or symbol, "bar": BAR,
        "managed": {"qty": qty},
    }


def _remaining(tmp_path) -> list[str]:
    path = pending_open_path("m", str(tmp_path))
    if not path.exists():
        return []
    return [e["order_symbol"] for e in json.loads(path.read_text())["entries"]]


# --- the guard itself ------------------------------------------------------

def test_working_order_symbols_reads_the_symbols_off_open_orders() -> None:
    client = _Client(open_orders=[{"id": "1", "symbol": "PLUG"},
                                  {"id": "2", "symbol": "PSIG"}])

    assert working_order_symbols(client) == {"PLUG", "PSIG"}


def test_an_unreadable_orders_endpoint_is_none_not_an_empty_set() -> None:
    """None means "could not ask". An empty set would claim nothing is working,
    which is the one answer that must never be guessed.
    """

    assert working_order_symbols(_Client(orders_raise=True)) is None
    assert working_order_symbols(_NoOrdersApiClient()) is None
    assert working_order_symbols(_Client(open_orders="not-a-list")) is None


# --- the flush -------------------------------------------------------------

def test_an_entry_with_a_working_order_is_not_sent_again(tmp_path, readiness_ok) -> None:
    """The 2026-08-24 case: unheld, but an order for it is already in flight."""

    _queue(tmp_path, [_entry("PLUG")])
    client = _Client(open_orders=[{"id": "1", "symbol": "PLUG"}])

    result = submit_pending_open_entries(
        client, "m", ["PLUG"],
        equity_tif_fn=lambda: "day", pos_lookup={}, ledger_root=str(tmp_path),
    )

    assert client.sent == [], "no duplicate reached the broker"
    assert result["count"] == 0
    assert [s["skip"] for s in result["skipped"]] == ["already_working"]


def test_a_working_order_retains_the_entry_rather_than_dropping_it(
    tmp_path, readiness_ok
) -> None:
    """Not terminal, unlike already_held. A working order can still be
    cancelled unfilled, and the queue is the only record the decision was made.
    Tomorrow it resolves to already_held or ages out on the bar guard.
    """

    _queue(tmp_path, [_entry("PLUG")])

    submit_pending_open_entries(
        _Client(open_orders=[{"id": "1", "symbol": "PLUG"}]), "m", ["PLUG"],
        equity_tif_fn=lambda: "day", pos_lookup={}, ledger_root=str(tmp_path),
    )

    assert _remaining(tmp_path) == ["PLUG"]


def test_the_guard_matches_on_the_order_symbol_not_the_ticker(
    tmp_path, readiness_ok
) -> None:
    """An option contract is the thing that gets bought twice, so the OCC
    symbol is what has to match — PATH the ticker is not PATH260918C00018000.
    """

    _queue(tmp_path, [_entry("PATH260918C00018000", ticker="PATH", qty=49)])
    client = _Client(open_orders=[{"id": "1", "symbol": "PATH260918C00018000"}])

    submit_pending_open_entries(
        client, "m", ["PATH"],
        equity_tif_fn=lambda: "day", pos_lookup={}, ledger_root=str(tmp_path),
    )

    assert client.sent == []


def test_an_unrelated_working_order_does_not_block_the_entry(
    tmp_path, readiness_ok
) -> None:
    """The guard must be narrow: a working order somewhere else in the account
    is not a reason to skip this name.
    """

    _queue(tmp_path, [_entry("PLUG")])
    client = _Client(open_orders=[{"id": "1", "symbol": "NVDA"}])

    submit_pending_open_entries(
        client, "m", ["PLUG"],
        equity_tif_fn=lambda: "day", pos_lookup={}, ledger_root=str(tmp_path),
    )

    assert client.sent == ["PLUG"]
    assert _remaining(tmp_path) == []


def test_an_unreadable_orders_endpoint_still_lets_the_entry_through(
    tmp_path, readiness_ok
) -> None:
    """Fail open, deliberately. The pre-existing already_held check still
    applies, and making a transient orders-endpoint blip block every deferred
    entry would trade a rare double-buy for a common silent outage.
    """

    _queue(tmp_path, [_entry("PLUG")])
    client = _Client(orders_raise=True)

    result = submit_pending_open_entries(
        client, "m", ["PLUG"],
        equity_tif_fn=lambda: "day", pos_lookup={}, ledger_root=str(tmp_path),
    )

    assert client.sent == ["PLUG"]
    assert result["count"] == 1


def test_a_client_without_get_orders_is_unaffected(tmp_path, readiness_ok) -> None:
    _queue(tmp_path, [_entry("PLUG")])
    client = _NoOrdersApiClient()

    submit_pending_open_entries(
        client, "m", ["PLUG"],
        equity_tif_fn=lambda: "day", pos_lookup={}, ledger_root=str(tmp_path),
    )

    assert client.sent == ["PLUG"]


def test_already_held_still_wins_over_already_working(tmp_path, readiness_ok) -> None:
    """A filled position is terminal; it must not be downgraded to a retained
    already_working just because an unrelated order for it is also open.
    """

    _queue(tmp_path, [_entry("PLUG")])
    client = _Client(open_orders=[{"id": "1", "symbol": "PLUG"}])

    result = submit_pending_open_entries(
        client, "m", ["PLUG"],
        equity_tif_fn=lambda: "day",
        pos_lookup={"PLUG": {"qty": 2273}}, ledger_root=str(tmp_path),
    )

    assert [s["skip"] for s in result["skipped"]] == ["already_held"]
    assert _remaining(tmp_path) == []


def test_a_stale_entry_is_dropped_even_when_an_order_is_working(
    tmp_path, readiness_ok
) -> None:
    """Staleness wins. A decision too old to act on is finished regardless of
    what is working at the broker; retaining it as already_working would keep a
    dead signal alive another session.
    """

    stale = _entry("PLUG")
    stale["bar"] = "2026-01-02T20:00:00Z"
    _queue(tmp_path, [stale])

    result = submit_pending_open_entries(
        _Client(open_orders=[{"id": "1", "symbol": "PLUG"}]), "m", ["PLUG"],
        equity_tif_fn=lambda: "day", pos_lookup={}, ledger_root=str(tmp_path),
    )

    assert [s["skip"] for s in result["skipped"]] == ["stale_bar"]
    assert _remaining(tmp_path) == []
