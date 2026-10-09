"""The consumer must not trade on bars it is reading late, and must not fall behind.

On 2026-10-05 and 2026-10-07 the shared stream's queue for this engine filled
(50,000 bars) and it went on evaluating midday bars after the close. Two things
were wrong. Broker upkeep ran once per bar, and with an exit pending each run is
a positions read plus an order read, so throughput fell far below the arrival
rate. And nothing compared a bar's timestamp with the clock before buying on it.
"""
from __future__ import annotations

import queue
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from strategies.intraday_structure import runner as runner_module
from strategies.intraday_structure.config import ExecutionPolicy, load_config
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.execution import IntradayOptionExecutor
from strategies.intraday_structure.models import Bar
from strategies.intraday_structure.options import NullOptionsProvider
from strategies.intraday_structure.runner import IntradayStructureRunner

NOW = datetime(2026, 10, 5, 19, 30, tzinfo=timezone.utc)


class _Client:
    def __init__(self):
        self.orders = []

    def submit_option_order(self, *, symbol, qty, side, **k):
        self.orders.append((side, symbol, qty))
        return {"id": f"oid-{symbol}-{side}", "status": "filled",
                "filled_avg_price": "1.00", "filled_qty": str(qty)}


def _contract(_client, ticker, price, **k):
    return ({"occ": f"{ticker}261009C00010000", "limit": 1.00, "mid": 0.98,
             "strike": 10.0, "expiry": "2026-10-09"}, "ok")


def _setup():
    return SimpleNamespace(setup_id="ABC:long:breakout_continuation", ticker="ABC",
                           direction="long", setup_type="breakout_continuation",
                           entry_price=10.0, spot=10.2, invalidation=9.8, metadata={})


def _executor(tmp_path, client, **over):
    policy = ExecutionPolicy(enabled=True, state_path=str(tmp_path / "open.json"), **over)
    return IntradayOptionExecutor(client, policy, select_option_fn=_contract,
                                  ledger_root=str(tmp_path), now_fn=lambda: NOW)


# --- entries on late bars -------------------------------------------------------

def test_entry_on_a_bar_read_hours_late_is_refused(tmp_path):
    client, setup = _Client(), _setup()
    ex = _executor(tmp_path, client)

    rec = ex.on_entry(setup, spot=10.0, bar_time=NOW - timedelta(hours=3))

    assert rec is None
    assert client.orders == []                      # nothing was sent to the broker
    assert ex.open_positions == {}
    assert setup.metadata["execution_skip"] == "stale_bar"
    assert setup.metadata["execution_bar_age_seconds"] == 3 * 3600


def test_entry_on_a_bar_delivered_on_time_is_taken(tmp_path):
    """A 1-minute bar is stamped at its open, so on-time delivery is ~60s old."""
    client = _Client()
    ex = _executor(tmp_path, client)

    rec = ex.on_entry(_setup(), spot=10.0, bar_time=NOW - timedelta(seconds=62))

    assert rec is not None
    assert [side for side, *_ in client.orders] == ["buy"]


def test_the_limit_is_the_boundary(tmp_path):
    ex = _executor(tmp_path, _Client(), max_entry_bar_age_seconds=180.0)
    assert ex.on_entry(_setup(), spot=10.0, bar_time=NOW - timedelta(seconds=180)) is not None
    ex = _executor(tmp_path / "b", _Client(), max_entry_bar_age_seconds=180.0)
    assert ex.on_entry(_setup(), spot=10.0, bar_time=NOW - timedelta(seconds=181)) is None


def test_zero_disables_the_gate(tmp_path):
    ex = _executor(tmp_path, _Client(), max_entry_bar_age_seconds=0)
    assert ex.on_entry(_setup(), spot=10.0, bar_time=NOW - timedelta(hours=3)) is not None


def test_engine_hands_the_executor_the_entry_bars_own_timestamp():
    seen = {}

    class _Sink:
        def on_entry(self, setup, **kwargs):
            seen.update(kwargs)

    engine = IntradayStructureEngine(load_config(), options_provider=NullOptionsProvider(),
                                     execution_sink=_Sink())
    stamp = datetime(2026, 10, 5, 16, 30, tzinfo=timezone.utc)
    bar = Bar(symbol="ABC", timestamp=stamp, open=10.0, high=10.1, low=9.9,
              close=10.05, volume=1000)

    engine._submit_entry(_setup(), bar)

    assert seen["bar_time"] == stamp
    assert seen["spot"] == 10.0


# --- upkeep on the clock, not per bar -------------------------------------------

class _Upkeep:
    def __init__(self):
        self.reconciles = 0
        self.flattens = 0

    def reconcile_exits(self):
        self.reconciles += 1

    def maybe_flatten_expiring(self):
        self.flattens += 1


def _bare_runner(executor, interval=5.0):
    runner = IntradayStructureRunner.__new__(IntradayStructureRunner)
    runner.executor = executor
    runner._maintenance_interval = interval
    runner._next_maintenance = 0.0
    runner._next_lag_report = 0.0
    runner.bar_queue = queue.Queue()
    runner.config = SimpleNamespace(execution=ExecutionPolicy())
    return runner


def test_broker_upkeep_runs_once_per_interval_however_many_bars_arrive(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: clock["t"])
    upkeep = _Upkeep()
    runner = _bare_runner(upkeep)

    for _ in range(500):                 # 500 bars inside one interval
        runner._flatten_expiring()
    assert (upkeep.reconciles, upkeep.flattens) == (1, 1)

    clock["t"] += 5.0
    for _ in range(500):
        runner._flatten_expiring()
    assert (upkeep.reconciles, upkeep.flattens) == (2, 2)


def test_a_failing_upkeep_run_does_not_retry_on_every_bar(monkeypatch):
    """A broker outage must cost one timeout per interval, not one per bar."""
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: 1000.0)
    calls = []

    class _Down:
        def reconcile_exits(self):
            calls.append(1)
            raise OSError("timed out")

    runner = _bare_runner(_Down())
    for _ in range(50):
        runner._flatten_expiring()
    assert len(calls) == 1


def test_lag_is_reported_once_a_minute_while_behind(monkeypatch, caplog):
    clock = {"t": 1000.0}
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: clock["t"])
    runner = _bare_runner(None)
    late = Bar(symbol="ABC", timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
               open=10.0, high=10.1, low=9.9, close=10.05, volume=1000)
    fresh = Bar(symbol="ABC", timestamp=datetime.now(timezone.utc) - timedelta(seconds=61),
                open=10.0, high=10.1, low=9.9, close=10.05, volume=1000)

    with caplog.at_level("WARNING", logger=runner_module.logger.name):
        runner._report_lag(fresh)
        assert not caplog.records
        for _ in range(100):
            runner._report_lag(late)
        assert len(caplog.records) == 1 and "behind the tape" in caplog.text
        clock["t"] += runner_module.LAG_REPORT_INTERVAL_SECONDS
        runner._report_lag(late)
        assert len(caplog.records) == 2
