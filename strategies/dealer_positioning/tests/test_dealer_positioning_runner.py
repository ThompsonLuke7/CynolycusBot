from __future__ import annotations

from datetime import date, datetime

from strategies.dealer_positioning.config import DealerPositioningConfig
from strategies.dealer_positioning import runner as runner_module
from strategies.dealer_positioning.runner import DealerPositioningRunner, _next_market_open
from strategies.dealer_positioning.schwab_adapter import SchwabDealerDataClient
from strategies.dealer_positioning.tests.test_dealer_positioning_core import _chain_fixture


class _FakeClient:
    def get_option_chain(self, symbol: str, ref_date: date) -> dict:
        return _chain_fixture()

    def get_quote_price(self, symbol: str) -> float:
        return 500.0


def test_runner_poll_writes_live_outputs(tmp_path, monkeypatch) -> None:
    class _FixtureDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 6, 12, 14, 30, tzinfo=runner_module.timezone.utc)
            return base if tz is None else base.astimezone(tz)

    config = DealerPositioningConfig(
        symbols=("SPY",),
        output_root=tmp_path,
        market_hours_only=False,
        write_archive_snapshots=False,
        strike_window_pct=0.10,
    )
    monkeypatch.setattr(runner_module, "datetime", _FixtureDateTime)
    runner = DealerPositioningRunner(config=config, data_client=_FakeClient())

    runner._poll_symbol("SPY")

    out_dir = tmp_path / "SPY"
    assert (out_dir / "live_gamma_ladder.csv").exists()
    assert (out_dir / "live_gamma_levels.json").exists()
    assert (out_dir / "live_trades.json").exists()
    assert runner.snapshot()["levels"]["SPY"]["call_wall"] == 510.0


def test_schwab_adapter_passes_date_objects_to_chain_client() -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return _chain_fixture()

    class _RawClient:
        class Options:
            class ContractType:
                ALL = "ALL"

            class Strategy:
                SINGLE = "SINGLE"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_option_chain(self, **kwargs):
            self.calls.append(kwargs)
            return _Response()

    raw_client = _RawClient()
    adapter = SchwabDealerDataClient.__new__(SchwabDealerDataClient)
    adapter._client = type("_Client", (), {"client": raw_client})()
    adapter._config = DealerPositioningConfig(dte_offsets=(0, 1, 2))

    adapter.get_option_chain("SPY", date(2026, 6, 16))

    call = raw_client.calls[0]
    assert call["from_date"] == date(2026, 6, 16)
    assert call["to_date"] == date(2026, 6, 18)


def test_next_market_open_skips_to_next_weekday_rth() -> None:
    friday_after_close = datetime(2026, 6, 12, 16, 30, tzinfo=runner_module._ET)
    saturday_midday = datetime(2026, 6, 13, 12, 0, tzinfo=runner_module._ET)

    assert _next_market_open(friday_after_close) == datetime(2026, 6, 15, 9, 30, tzinfo=runner_module._ET)
    assert _next_market_open(saturday_midday) == datetime(2026, 6, 15, 9, 30, tzinfo=runner_module._ET)


def test_runner_closed_hours_waits_until_open_without_polling(tmp_path, monkeypatch) -> None:
    captured: list[tuple[str, dict]] = []
    config = DealerPositioningConfig(
        symbols=("SPY",),
        output_root=tmp_path,
        market_hours_only=True,
        poll_seconds=60,
        write_archive_snapshots=False,
    )
    runner = DealerPositioningRunner(
        config=config,
        data_client=_FakeClient(),
        event_sink=lambda event_type, payload: captured.append((event_type, payload)),
    )

    class _FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 6, 15, 22, 30, tzinfo=runner_module._ET)
            return base if tz is None else base.astimezone(tz)

    class _OneShotStop:
        def __init__(self) -> None:
            self.wait_calls: list[float] = []
            self._set = False

        def is_set(self) -> bool:
            return self._set

        def wait(self, timeout=None) -> bool:
            self.wait_calls.append(float(timeout))
            self._set = True
            return True

        def set(self) -> None:
            self._set = True

    def _should_not_poll(symbol: str) -> None:
        raise AssertionError(f"unexpected poll during closed hours for {symbol}")

    monkeypatch.setattr(runner_module, "datetime", _FakeDateTime)
    runner._stop = _OneShotStop()
    monkeypatch.setattr(runner, "_poll_symbol", _should_not_poll)

    runner.start()

    market_closed = [payload for event_type, payload in captured if event_type == "market_closed"]
    assert market_closed, "expected a market_closed event"
    assert market_closed[0]["next_open_et"] == "2026-06-16T09:30:00-04:00"
    assert runner._stop.wait_calls == [11 * 3600.0]
