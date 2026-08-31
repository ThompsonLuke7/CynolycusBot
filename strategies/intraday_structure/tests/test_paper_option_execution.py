"""Real paper option orders for confirmed setups.

This engine had no broker code at all: every closed setup was a MODELLED fill,
which is honest but means its results are not comparable with the other modules'
and the large 2026-08 upgrade to it can be neither credited nor blamed.

The properties that matter here are mostly SAFETY ones. A rules engine that fires
on hundreds of setups a day must not be able to express that as hundreds of open
contracts, and a broker outage must degrade it back to the simulation it already
was rather than stop setups being detected.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from strategies.intraday_structure.config import ExecutionPolicy, load_config
from strategies.intraday_structure.execution import IntradayOptionExecutor


class _Client:
    def __init__(self, fill=1.25):
        self.fill = fill
        self.orders = []

    def submit_option_order(self, *, symbol, qty, side, **k):
        self.orders.append((side, symbol, qty))
        return {"id": f"oid-{symbol}-{side}", "submitted_at": "2026-08-29T14:00:00Z",
                "filled_at": "2026-08-29T14:00:01Z",
                "filled_avg_price": str(self.fill), "filled_qty": str(qty)}


def _contract(_client, ticker, price, *, option_type, min_dte, max_dte,
              allow_0dte=False, **k):
    """Stub chain. Same-day expiry is only offered when it was asked for, which
    mirrors reality: daily expiries exist on the index ETFs, not on most names."""
    assert min_dte >= 0
    assert allow_0dte or min_dte >= 1, "0DTE requested without opting in"
    return ({"occ": f"{ticker}260904{'C' if option_type == 'call' else 'P'}00010000",
             "limit": 1.00, "mid": 0.98, "strike": 10.0, "expiry": "2026-09-04"}, "ok")


def _setup(setup_id="ABC:long:trend_pullback_continuation", ticker="ABC", direction="long"):
    return SimpleNamespace(
        setup_id=setup_id, ticker=ticker, direction=direction,
        setup_type="trend_pullback_continuation", entry_price=10.0, spot=10.2,
        invalidation=9.8, metadata={},
    )


def _executor(tmp_path, client=None, **over):
    kwargs = {"enabled": True, "target_notional": 1000.0,
              "state_path": str(tmp_path / "open.json")}
    kwargs.update(over)
    policy = ExecutionPolicy(**kwargs)
    return IntradayOptionExecutor(client or _Client(), policy,
                                  select_option_fn=_contract,
                                  ledger_root=str(tmp_path))


# --- config safety --------------------------------------------------------------

def test_execution_is_on_by_default_on_the_paper_account():
    ex = load_config().execution
    assert ex.enabled is True
    assert ex.allow_0dte is True


def test_0dte_requires_a_flatten_time(tmp_path):
    """The two settings are one decision: a same-day contract with no flatten
    time is a position that rides into expiry and gets assigned."""
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"version": "intraday_structure_v1", "paper_only": True,
                               "execution": {"allow_0dte": True, "expiring_exit_hhmm": ""}}))
    with pytest.raises(ValueError, match="expiring_exit_hhmm"):
        load_config(cfg)


def test_execution_requires_paper_only(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"version": "intraday_structure_v1", "paper_only": False,
                               "execution": {"enabled": True}}))
    with pytest.raises(ValueError, match="paper_only"):
        load_config(cfg)


# --- entry ----------------------------------------------------------------------

def test_a_long_setup_buys_a_call_and_a_short_buys_a_put(tmp_path):
    client = _Client()
    ex = _executor(tmp_path, client)
    ex.on_entry(_setup(), spot=10.0)
    ex.on_entry(_setup("XYZ:short:x", "XYZ", "short"), spot=10.0)
    sides = [o[1][-9] for o in client.orders]        # the C/P in the OCC symbol
    assert sides == ["C", "P"]
    assert all(o[0] == "buy" for o in client.orders)


def test_size_respects_the_notional_budget(tmp_path):
    """$1,000 / ($1.00 * 100) = 10 contracts."""
    client = _Client()
    _executor(tmp_path, client).on_entry(_setup(), spot=10.0)
    assert client.orders[0][2] == 10


def test_size_is_capped_by_max_contracts(tmp_path):
    client = _Client()
    _executor(tmp_path, client, target_notional=1e9, max_contracts=3).on_entry(_setup(), spot=10.0)
    assert client.orders[0][2] == 3


def test_the_underlying_leg_is_captured_at_entry(tmp_path):
    """u_entry/u_atr next to the premium — the pair the instrument question needs."""
    ex = _executor(tmp_path)
    rec = ex.on_entry(_setup(), spot=10.0, atr=0.4)
    assert rec["u_entry"] == 10.0 and rec["u_atr"] == 0.4


# --- capacity -------------------------------------------------------------------

def test_concurrent_positions_are_capped(tmp_path):
    client = _Client()
    ex = _executor(tmp_path, client, max_concurrent_positions=2)
    for i in range(5):
        ex.on_entry(_setup(f"T{i}:long:x", f"T{i}"))
    assert len(client.orders) == 2
    assert len(ex.open_positions) == 2


def test_new_positions_per_session_are_capped(tmp_path):
    client = _Client()
    ex = _executor(tmp_path, client, max_concurrent_positions=99,
                   max_new_positions_per_session=3)
    for i in range(6):
        ex.on_entry(_setup(f"T{i}:long:x", f"T{i}"))
    assert len(client.orders) == 3


def test_the_same_setup_is_not_bought_twice(tmp_path):
    client = _Client()
    ex = _executor(tmp_path, client)
    ex.on_entry(_setup())
    ex.on_entry(_setup())
    assert len(client.orders) == 1


# --- exit -----------------------------------------------------------------------

def test_exit_sells_and_writes_the_shared_ledger_row(tmp_path):
    client = _Client(fill=1.00)
    ex = _executor(tmp_path, client)
    ex.on_entry(_setup(), spot=10.0, atr=0.4)
    client.fill = 1.60                                   # exit richer than entry
    s = _setup()
    s.metadata["exit_price"] = 10.9
    rec = ex.on_exit(s, exit_reason="target reached", spot=10.9)
    assert client.orders[-1][0] == "sell"
    assert rec["realized_pnl"] == pytest.approx((1.60 - 1.00) * 100 * 10, rel=1e-6)
    row = json.loads((tmp_path / "intraday_structure" / "closed_trades.jsonl").read_text().strip())
    assert row["module"] == "intraday_structure"
    assert row["exit_reason"] == "target reached"
    assert row["u_entry"] == 10.0 and row["u_exit"] == 10.9
    assert row["modelled_exit_price"] == 10.9            # the simulation, kept alongside
    assert not ex.open_positions


def test_an_exit_for_a_setup_we_never_bought_is_a_no_op(tmp_path):
    client = _Client()
    ex = _executor(tmp_path, client)
    assert ex.on_exit(_setup(), exit_reason="invalidation touched") is None
    assert client.orders == []


def test_a_failed_exit_keeps_the_position_open_for_retry(tmp_path):
    """An unowned contract is how a sibling module ends up liquidating it."""
    class _Flaky(_Client):
        def submit_option_order(self, *, symbol, qty, side, **k):
            if side == "sell":
                raise RuntimeError("422 no quote")
            return super().submit_option_order(symbol=symbol, qty=qty, side=side, **k)

    ex = _executor(tmp_path, _Flaky())
    ex.on_entry(_setup())
    assert ex.on_exit(_setup(), exit_reason="stop") is None
    assert len(ex.open_positions) == 1                   # still ours


# --- resilience -----------------------------------------------------------------

def test_a_broker_outage_never_raises_into_the_engine(tmp_path):
    class _Dead(_Client):
        def submit_option_order(self, **k):
            raise ConnectionError("network down")

    ex = _executor(tmp_path, _Dead())
    assert ex.on_entry(_setup()) is None                 # degraded, not crashed


def test_no_contract_records_why_and_places_nothing(tmp_path):
    client = _Client()
    policy = ExecutionPolicy(enabled=True, state_path=str(tmp_path / "o.json"))
    ex = IntradayOptionExecutor(client, policy,
                                select_option_fn=lambda *a, **k: (None, "no_non_0dte_call_contracts"),
                                ledger_root=str(tmp_path))
    s = _setup()
    ex.on_entry(s, spot=10.0)
    assert client.orders == []
    assert s.metadata["execution_skip"] == "no_contract(no_non_0dte_call_contracts)"


def test_open_positions_survive_a_restart(tmp_path):
    client = _Client()
    ex = _executor(tmp_path, client)
    ex.on_entry(_setup(), spot=10.0)
    revived = _executor(tmp_path, client)                # same state_path
    assert len(revived.open_positions) == 1
    revived.on_exit(_setup(), exit_reason="stop", spot=9.9)
    assert client.orders[-1][0] == "sell"


# --- 0DTE: the expiry roll and the flatten --------------------------------------

def _at(hh, mm, day=4):
    """A UTC instant that is `hh:mm` New York time on a 2026-09 weekday."""
    from zoneinfo import ZoneInfo
    return datetime(2026, 9, day, hh, mm, tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)


def _executor_at(tmp_path, when, client=None, **over):
    kwargs = {"enabled": True, "target_notional": 1000.0,
              "state_path": str(tmp_path / "open.json")}
    kwargs.update(over)
    return IntradayOptionExecutor(client or _Client(), ExecutionPolicy(**kwargs),
                                  select_option_fn=_contract, ledger_root=str(tmp_path),
                                  now_fn=lambda: when)


def test_same_day_expiry_is_taken_before_the_cutoff(tmp_path):
    """A news burst at 10:05 ET should be expressed in today's contract."""
    ex = _executor_at(tmp_path, _at(10, 5))
    assert ex._min_dte_now() == 0


def test_the_expiry_rolls_to_the_next_session_after_the_cutoff(tmp_path):
    """Past 13:00 ET a same-day contract is mostly the move that already
    happened plus the theta about to be taken."""
    ex = _executor_at(tmp_path, _at(14, 30))
    assert ex._min_dte_now() == 1


def test_0dte_is_requested_from_the_selector_only_before_the_cutoff(tmp_path):
    seen = {}

    def _spy(_c, ticker, price, *, option_type, min_dte, max_dte, allow_0dte=False, **k):
        seen[ticker] = (min_dte, allow_0dte)
        return ({"occ": f"{ticker}260904C00010000", "limit": 1.0,
                 "expiry": "2026-09-04", "strike": 10.0}, "ok")

    for when, ticker, expect in ((_at(10, 0), "EARLY", (0, True)), (_at(15, 0), "LATE", (1, False))):
        ex = IntradayOptionExecutor(
            _Client(), ExecutionPolicy(enabled=True, state_path=str(tmp_path / f"{ticker}.json")),
            select_option_fn=_spy, ledger_root=str(tmp_path), now_fn=lambda w=when: w)
        ex.on_entry(_setup(f"{ticker}:long:x", ticker), spot=10.0)
    assert seen["EARLY"] == (0, True)
    assert seen["LATE"] == (1, False)


def test_a_same_day_expiry_is_flattened_before_the_close(tmp_path):
    """The whole reason 0DTE is safe here: it never reaches expiry."""
    client = _Client()
    ex = _executor_at(tmp_path, _at(10, 0), client)
    ex.on_entry(_setup(), spot=10.0)              # buys the 2026-09-04 contract
    ex._now = lambda: _at(15, 45)                 # past the 15:40 flatten
    closed = ex.maybe_flatten_expiring()
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "expiring_flatten"
    assert closed[0]["urgent_exit"] is True
    assert client.orders[-1][0] == "sell"
    assert not ex.open_positions


def test_nothing_is_flattened_before_the_cutoff(tmp_path):
    client = _Client()
    ex = _executor_at(tmp_path, _at(10, 0), client)
    ex.on_entry(_setup(), spot=10.0)
    ex._now = lambda: _at(15, 0)                  # 15:00 < 15:40
    assert ex.maybe_flatten_expiring() == []
    assert len(ex.open_positions) == 1


def test_a_later_dated_position_is_left_alone_by_the_flatten(tmp_path):
    def _next_week(_c, ticker, price, **k):
        return ({"occ": f"{ticker}260911C00010000", "limit": 1.0,
                 "expiry": "2026-09-11", "strike": 10.0}, "ok")

    ex = IntradayOptionExecutor(
        _Client(), ExecutionPolicy(enabled=True, state_path=str(tmp_path / "o.json")),
        select_option_fn=_next_week, ledger_root=str(tmp_path), now_fn=lambda: _at(15, 45))
    ex.on_entry(_setup(), spot=10.0)
    assert ex.maybe_flatten_expiring() == []
    assert len(ex.open_positions) == 1


def test_the_flatten_is_idempotent(tmp_path):
    client = _Client()
    ex = _executor_at(tmp_path, _at(10, 0), client)
    ex.on_entry(_setup(), spot=10.0)
    ex._now = lambda: _at(15, 45)
    ex.maybe_flatten_expiring()
    before = len(client.orders)
    assert ex.maybe_flatten_expiring() == []
    assert len(client.orders) == before


def test_dte_at_entry_is_recorded(tmp_path):
    ex = _executor_at(tmp_path, _at(10, 0))
    rec = ex.on_entry(_setup(), spot=10.0)
    assert rec["dte_at_entry"] == 0          # bought 09-04 on 09-04
