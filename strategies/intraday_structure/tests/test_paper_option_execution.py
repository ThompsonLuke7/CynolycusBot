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


# --- fill verification, broker truth and retry discipline (2026-09-01) ---------
#
# On 2026-09-01 this module bought ten contracts, multi_ticker_swing's reconcile
# adopted six and force-liquidated four within three minutes, and the executor —
# which had recorded every position straight off the submit response and kept it
# on failure "so the next event retries" — submitted 9,945 rejected sells over
# eight hours, ~190/min, against contracts the account no longer held.

class _UnfilledClient(_Client):
    """A limit buy that is accepted and then never fills."""

    def __init__(self):
        super().__init__()
        self.cancelled = []

    def submit_option_order(self, *, symbol, qty, side, **k):
        self.orders.append((side, symbol, qty))
        return {"id": f"oid-{symbol}", "submitted_at": "2026-09-01T14:00:00Z",
                "status": "new", "filled_avg_price": None, "filled_qty": "0"}

    def get_order(self, order_id):
        return {"id": order_id, "status": "new",
                "filled_avg_price": None, "filled_qty": "0"}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return {"id": order_id, "status": "canceled"}


class _PositionsClient(_Client):
    """Tracks what the account holds, and can have it taken away."""

    def __init__(self, fill=1.25):
        super().__init__(fill=fill)
        self.held: dict[str, float] = {}
        self.sell_attempts = 0

    def submit_option_order(self, *, symbol, qty, side, **k):
        self.orders.append((side, symbol, qty))
        if side == "buy":
            self.held[symbol.upper()] = float(qty)
        else:
            self.sell_attempts += 1
            if not self.held.get(symbol.upper()):
                raise RuntimeError(
                    'HTTP Error 403: {"code":40310000,"message":"account not '
                    'eligible to trade uncovered option contracts"}')
            self.held.pop(symbol.upper(), None)
        return {"id": f"oid-{symbol}-{side}", "submitted_at": "2026-09-01T14:00:00Z",
                "filled_avg_price": str(self.fill), "filled_qty": str(qty)}

    def get_positions(self, **k):
        return [{"symbol": s, "qty": str(q), "asset_class": "us_option"}
                for s, q in self.held.items()]


def test_an_unfilled_entry_is_not_claimed(tmp_path):
    """The root cause: a submit response is not a fill."""
    client = _UnfilledClient()
    ex = _executor(tmp_path, client)
    setup = _setup()

    assert ex.on_entry(setup, spot=10.2) is None
    assert ex.open_positions == {}
    assert setup.metadata["execution_skip"] == "entry_unfilled"
    assert client.cancelled, "an entry that never filled must be cancelled"


def test_a_filled_entry_records_the_real_fill_not_the_limit(tmp_path):
    client = _PositionsClient(fill=0.91)
    ex = _executor(tmp_path, client)

    rec = ex.on_entry(_setup(), spot=10.2)
    assert rec is not None
    assert rec["entry_fill_price"] == 0.91
    assert rec["entry_filled_qty"] > 0, "every stuck 2026-09-01 row read 0.0 here"
    assert rec["qty"] == int(rec["entry_filled_qty"])


def test_exit_releases_the_claim_when_the_broker_no_longer_holds_it(tmp_path):
    """A sibling liquidated it. Selling again is a naked short, not a close."""
    client = _PositionsClient()
    ex = _executor(tmp_path, client)
    setup = _setup()
    rec = ex.on_entry(setup, spot=10.2)
    occ = rec["occ"]

    client.held.pop(occ.upper())          # multi_ticker_swing got there first
    before = client.sell_attempts

    assert ex.on_exit(setup, exit_reason="invalidation touched") is None
    assert ex.open_positions == {}, "the claim must be released, not retried"
    assert client.sell_attempts == before, "no sell may be submitted for an unheld contract"


def test_a_failing_exit_backs_off_and_then_quarantines(tmp_path):
    """Bounded retries. The old path had none and ran ~190/min for eight hours."""
    from strategies.intraday_structure import execution as ex_mod

    client = _PositionsClient()
    ex = _executor(tmp_path, client)
    setup = _setup()
    rec = ex.on_entry(setup, spot=10.2)
    occ = rec["occ"]
    setup_id = rec["setup_id"]

    # Held at the broker (so the guard above passes) but every sell is refused.
    def _refuse(*a, **k):
        client.sell_attempts += 1
        raise RuntimeError("HTTP Error 422: options market orders are only allowed "
                           "during market hours")
    client.submit_option_order = _refuse

    attempts_made = 0
    for _ in range(200):
        ex.on_exit(setup, exit_reason="expiring_flatten")
        pos = ex.open_positions.get(setup_id)
        if pos is None:
            break
        if pos.get("quarantined"):
            attempts_made = pos["exit_attempts"]
            break
        # Pretend the backoff elapsed, so the loop reaches the cap in-test.
        ex._open[setup_id]["last_exit_attempt_at"] = None

    assert attempts_made == ex_mod.EXIT_MAX_ATTEMPTS
    # Each attempt walks the shared ladder (market rung, then limit rungs), so
    # raw submits exceed attempts. What must be bounded is the total.
    submits_at_quarantine = client.sell_attempts
    assert submits_at_quarantine < 40, "retries must be bounded, not a hot loop"

    # And it stays stopped: a quarantined position generates no further orders.
    ex._open[setup_id]["last_exit_attempt_at"] = None
    for _ in range(50):
        ex.on_exit(setup, exit_reason="expiring_flatten")
    assert client.sell_attempts == submits_at_quarantine


def test_backoff_blocks_a_resubmit_inside_the_window(tmp_path):
    client = _PositionsClient()
    ex = _executor(tmp_path, client)
    setup = _setup()
    rec = ex.on_entry(setup, spot=10.2)

    def _refuse(*a, **k):
        client.sell_attempts += 1
        raise RuntimeError("nope")
    client.submit_option_order = _refuse

    ex.on_exit(setup, exit_reason="expiring_flatten")
    first = client.sell_attempts
    for _ in range(50):                    # the runner's 1s queue-empty path
        ex.on_exit(setup, exit_reason="expiring_flatten")
    assert client.sell_attempts == first, "backoff must swallow the hot loop"


def test_startup_reconcile_drops_contracts_the_account_lost(tmp_path):
    """A restart used to resume the loop: `expiry <= today` is true forever once
    the expiry is in the past, so the stale claims came straight back."""
    state = tmp_path / "open.json"
    state.write_text(json.dumps({"open": {
        "QQQ:short:vwap": {"occ": "QQQ260901P00708000", "ticker": "QQQ", "qty": 9,
                           "expiry": "2026-09-01", "entry_filled_qty": 0.0},
    }}))
    client = _PositionsClient()          # holds nothing

    ex = _executor(tmp_path, client, state_path=str(state))
    assert ex.open_positions == {}, "a claim the broker cannot confirm must not survive restart"


def test_a_broker_outage_never_drops_a_claim(tmp_path):
    """None (unreachable) and {} (answered, you hold nothing) must not be conflated."""
    class _Down(_PositionsClient):
        def get_positions(self, **k):
            raise RuntimeError("connection reset")

    client = _Down()
    ex = _executor(tmp_path, client)
    rec = ex.on_entry(_setup(), spot=10.2)
    assert rec is not None
    assert ex.reconcile_with_broker() == []
    assert rec["setup_id"] in ex.open_positions


# --- shared-account overlap ----------------------------------------------------
#
# Modules are NOT fenced off each other's tickers: a 4H module holding QQQ for
# two days and intraday_structure holding a QQQ contract for an hour are
# different bets and both should be taken. But Alpaca nets option positions by
# symbol, so when two modules hold the SAME contract the broker reports one
# combined position — and a module that resizes its exit to that total sells the
# other module's contracts.

def test_an_exit_never_sells_more_than_the_module_claims(tmp_path):
    client = _PositionsClient()
    ex = _executor(tmp_path, client)
    setup = _setup()
    rec = ex.on_entry(setup, spot=10.2)
    occ, mine = rec["occ"], int(rec["qty"])

    # A sibling module buys the same contract. The broker now nets both.
    client.held[occ.upper()] = float(mine + 7)

    sold = []
    real = client.submit_option_order

    def _record(*, symbol, qty, side, **k):
        if side == "sell":
            sold.append(int(qty))
        return real(symbol=symbol, qty=qty, side=side, **k)
    client.submit_option_order = _record

    ex.on_exit(setup, exit_reason="invalidation touched")
    assert sold and sold[0] == mine, (
        f"sold {sold[0]} of a netted {mine + 7} — a sibling's contracts went with it")


def test_an_exit_caps_down_when_the_broker_holds_fewer(tmp_path):
    """The other direction still has to work: part of the position is gone."""
    client = _PositionsClient()
    ex = _executor(tmp_path, client)
    setup = _setup()
    rec = ex.on_entry(setup, spot=10.2)
    occ, mine = rec["occ"], int(rec["qty"])
    assert mine > 1

    client.held[occ.upper()] = 1.0

    sold = []
    real = client.submit_option_order

    def _record(*, symbol, qty, side, **k):
        if side == "sell":
            sold.append(int(qty))
        return real(symbol=symbol, qty=qty, side=side, **k)
    client.submit_option_order = _record

    ex.on_exit(setup, exit_reason="invalidation touched")
    assert sold and sold[0] == 1


# --- order-id attribution ------------------------------------------------------
#
# A module's own filled quantity, read back off its own entry order, is the only
# attribution available on a shared account: Alpaca nets option positions by
# symbol, so the broker can never say which contracts are whose.

class _PartialFillClient(_PositionsClient):
    """Fills 4 of whatever was asked for."""

    def __init__(self, fill=1.25, partial=4):
        super().__init__(fill=fill)
        self.partial = partial

    def submit_option_order(self, *, symbol, qty, side, **k):
        self.orders.append((side, symbol, qty))
        if side == "buy":
            self.held[symbol.upper()] = float(self.partial)
            return {"id": f"oid-{symbol}", "submitted_at": "2026-09-01T14:00:00Z",
                    "filled_avg_price": str(self.fill), "filled_qty": str(self.partial)}
        self.sell_attempts += 1
        self.held.pop(symbol.upper(), None)
        return {"id": f"oid-{symbol}-sell", "filled_avg_price": str(self.fill),
                "filled_qty": str(qty)}


def test_a_partial_fill_claims_only_what_filled(tmp_path):
    client = _PartialFillClient(partial=4)
    ex = _executor(tmp_path, client)          # $1000 / $1.00 -> asks for 10

    rec = ex.on_entry(_setup(), spot=10.0)
    assert client.orders[0][2] == 10, "still asks for the full size"
    assert rec["entry_filled_qty"] == 4
    assert rec["qty"] == 4
    assert rec["requested_qty"] == 10


def test_the_exit_sells_the_filled_size_not_the_requested_size(tmp_path):
    client = _PartialFillClient(partial=4)
    ex = _executor(tmp_path, client)
    setup = _setup()
    ex.on_entry(setup, spot=10.0)

    ex.on_exit(setup, exit_reason="invalidation touched")
    sells = [o for o in client.orders if o[0] == "sell"]
    assert sells and sells[0][2] == 4, "sold the requested 10, not the filled 4"


def test_the_exit_size_comes_from_our_order_not_the_netted_position(tmp_path):
    """The attribution property, stated directly."""
    client = _PositionsClient()
    ex = _executor(tmp_path, client)
    setup = _setup()
    rec = ex.on_entry(setup, spot=10.0)
    mine = int(rec["entry_filled_qty"])

    # Three sibling modules pile into the same contract.
    client.held[rec["occ"].upper()] = float(mine * 4)

    ex.on_exit(setup, exit_reason="invalidation touched")
    sells = [o for o in client.orders if o[0] == "sell"]
    assert sells and sells[0][2] == mine


def test_an_entry_that_fills_against_the_cancel_is_still_claimed(tmp_path):
    """Cancellation races the book; the fill must not become an orphan."""
    class _RacyClient(_UnfilledClient):
        def __init__(self):
            super().__init__()
            self._cancelled = False

        def cancel_order(self, order_id):
            self._cancelled = True
            return {"id": order_id, "status": "filled"}

        def get_order(self, order_id):
            # Unfilled while polling; filled by the time the cancel lands.
            if self._cancelled:
                return {"id": order_id, "status": "filled",
                        "filled_avg_price": "1.00", "filled_qty": "10"}
            return {"id": order_id, "status": "new",
                    "filled_avg_price": None, "filled_qty": "0"}

    ex = _executor(tmp_path, _RacyClient())
    setup = _setup()
    rec = ex.on_entry(setup, spot=10.0)
    assert rec is not None, "a position that filled anyway must be owned, not orphaned"
    assert rec["entry_filled_qty"] == 10


# A rejected cancel is evidence, not noise (2026-09-02 SPY260902C00765000).
#
# The cancel came back `422 order is already in "filled" state` — the broker
# stating that four contracts had been bought — and that answer was discarded
# in favour of a follow-up order read that reported nothing. The module logged
# "did not fill; not claiming it", live_risk_pass flagged the contract as an
# orphan three times, and fourteen minutes later the SPY daytrader's broker
# reconcile adopted all four and liquidated them for -$235.


class _AlreadyFilledCancelClient(_UnfilledClient):
    """Cancel is rejected because the order filled; the order read stays blind."""

    def __init__(self, held_qty: float | None = None, positions_readable=True):
        super().__init__()
        self._held_qty = held_qty
        self._positions_readable = positions_readable

    def cancel_order(self, order_id):
        raise RuntimeError(
            'HTTP Error 422: Unprocessable Entity: {"code":42210000,'
            '"message":"order is already in \\"filled\\" state"}'
        )

    def get_positions(self, **k):
        if not self._positions_readable:
            raise RuntimeError("broker unreachable")
        if not self._held_qty:
            return []
        return [{
            "symbol": self.orders[-1][1].upper(),
            "asset_class": "us_option",
            "qty": str(self._held_qty),
        }]


def test_a_cancel_rejected_as_already_filled_claims_the_position(tmp_path):
    """The account holds it, so the module must own it rather than walk away."""

    client = _AlreadyFilledCancelClient(held_qty=4)
    ex = _executor(tmp_path, client)

    rec = ex.on_entry(_setup(), spot=10.0)

    assert rec is not None, (
        "the broker said the order filled; abandoning it creates the orphan a "
        "sibling module then adopts and liquidates"
    )
    assert rec["entry_filled_qty"] == 4


def test_a_position_claimed_this_way_is_clamped_to_what_we_ordered(tmp_path):
    """The same contract may be held by a sibling; only our own order is ours."""

    client = _AlreadyFilledCancelClient(held_qty=500)
    ex = _executor(tmp_path, client)

    rec = ex.on_entry(_setup(), spot=10.0)

    assert rec is not None
    assert rec["entry_filled_qty"] == rec["requested_qty"]
    assert rec["entry_filled_qty"] < 500


def test_an_unreadable_account_does_not_abandon_a_filled_order(tmp_path):
    """"Could not ask" is never grounds for dropping a claim.

    Under-claiming leaves a position nothing stops or exits; over-claiming is
    corrected by the broker check `_close_position` runs on the way out.
    """

    client = _AlreadyFilledCancelClient(positions_readable=False)
    ex = _executor(tmp_path, client)

    rec = ex.on_entry(_setup(), spot=10.0)

    assert rec is not None
    assert rec["entry_filled_qty"] == rec["requested_qty"]


def test_a_genuinely_unfilled_entry_is_still_not_claimed(tmp_path):
    """The fix must not turn every failed entry into a phantom position."""

    client = _UnfilledClient()
    ex = _executor(tmp_path, client)

    assert ex.on_entry(_setup(), spot=10.0) is None
    assert client.cancelled, "an unfilled entry is still cancelled"


def test_a_cancel_rejected_for_any_other_reason_does_not_claim(tmp_path):
    """Only "already filled" is evidence of a fill."""

    class _OtherRejection(_AlreadyFilledCancelClient):
        def cancel_order(self, order_id):
            raise RuntimeError("HTTP Error 500: Internal Server Error")

    ex = _executor(tmp_path, _OtherRejection(held_qty=4))

    assert ex.on_entry(_setup(), spot=10.0) is None
