"""A position must not be opened and closed on the bar it was entered on.

On 2026-09-02 five of fifteen intraday entries were closed one to two seconds
after the fill, for a combined -$203 that was entirely spread:

    PATH 10:46:18 -> 10:46:19  invalidation touched
    HOOD 11:02:30 -> 11:02:32  invalidation touched
    NKE  13:24:17 -> 13:24:18  invalidation touched
    HOOD 14:03:52 -> 14:03:54  invalidation touched
    HOOD 15:33:40 -> 15:33:42  maximum setup duration reached

Two independent causes, both here.

*Managing the entry bar.* The engine only ever sees a bar once it is complete,
so the entry order is really sent after that bar closed. Evaluating
`manage_running_setup` on the same bar compared its already-known high/low
against the invalidation and stopped the trade out on price action from before
the position existed.

*Crossed clocks.* `max_setup_bars` limits how long an unentered setup may wait;
`time_exit_bars` is a position's holding period. `manage_running_setup` read the
first as a reason to close a position, so a setup that confirmed at or past the
staleness limit was bought and immediately closed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.models import Bar, Direction, SetupState
from strategies.intraday_structure.target_manager import manage_running_setup


NOW = datetime(2026, 9, 2, 14, 46, tzinfo=timezone.utc)


class _Setup:
    """The fields `manage_running_setup` reads."""

    def __init__(self, *, entry_time, bars_alive, bars_in_state, invalidation=None):
        self.direction = Direction.LONG
        self.entry_time = entry_time
        self.entry_price = 10.0
        self.bars_alive = bars_alive
        self.bars_in_state = bars_in_state
        self.invalidation = invalidation
        self.active_target = None
        self.active_target_index = 0
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0


class _Ctx:
    def __init__(self, bar, config):
        self.bar = bar
        self.config = config
        self.bars = []
        # `_tighten_structure_stop` runs on the no-exit path.
        self.features = {"atr": 0.2, "micro_swing_low": 9.5, "micro_swing_high": 10.5}
        self.levels = None
        self.market = None
        self.options = None


def _bar(ts=NOW, low=9.9):
    return Bar(symbol="XYZ", timestamp=ts, open=10.0, high=10.1, low=low,
               close=10.0, volume=1000.0)


def _config(**overrides):
    return IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0, **overrides)


def test_an_entered_position_is_not_closed_on_the_setup_staleness_clock() -> None:
    """`max_setup_bars` governs waiting, not holding."""

    config = _config()
    setup = _Setup(
        entry_time=NOW,
        bars_alive=config.target.max_setup_bars + 10,  # waited a long time to trigger
        bars_in_state=0,                               # but has only just been entered
    )

    decision = manage_running_setup(setup, _Ctx(_bar(), config))

    assert decision.state is not SetupState.CLOSED, (
        "a position entered this bar must not be closed by the setup's own age"
    )


def test_an_entered_position_is_still_closed_on_its_own_holding_clock() -> None:
    """The holding limit must keep working."""

    config = _config()
    setup = _Setup(
        entry_time=NOW,
        bars_alive=5,
        bars_in_state=config.target.time_exit_bars,
    )

    decision = manage_running_setup(setup, _Ctx(_bar(), config))

    assert decision.state is SetupState.CLOSED
    assert decision.evidence == ("time_exit",)


def test_an_unentered_setup_is_still_abandoned_when_it_goes_stale() -> None:
    """Nothing was bought, so aging out costs nothing and must still happen."""

    config = _config()
    setup = _Setup(
        entry_time=None,
        bars_alive=config.target.max_setup_bars,
        bars_in_state=0,
    )

    decision = manage_running_setup(setup, _Ctx(_bar(), config))

    assert decision.state is SetupState.CLOSED
    assert decision.evidence == ("time_exit",)
