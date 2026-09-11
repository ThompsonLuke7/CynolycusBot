"""Keep the swing test suite away from live trading state.

`position_manager` holds its three state files as module-level relative
`Path`s and writes them as a side effect of ordinary operations — restoring a
broker position calls `_persist_open_position_state()`. A test that exercises
that path therefore writes the *real*
`Data/inference/multi_ticker_swing/open_positions.json`.

That is not hypothetical. On 2026-09-08 a full-suite run overwrote the live
book with the single synthetic FIG260724C00022000 position from
`test_a_readable_empty_book_still_permits_adoption`, erasing the seven option
positions swing had genuinely restored from the broker minutes earlier. Because
every sibling module reads that file to decide what it may adopt
(`core.orphan_positions.claimed_symbols`), those seven live positions instantly
read as unowned to the orphan scan and to every other module's reconcile.

The blast radius of running the tests must never include live state, so this
redirects all three paths into a per-test tmp directory for the whole package.
"""
from __future__ import annotations

import pytest

from strategies.multi_ticker_swing.live import position_manager as pm_module

_REDIRECTED_STATE_PATHS = (
    "_OPEN_POSITION_STATE_PATH",
    "_DEFERRED_TRAIL_STATE_PATH",
    "_WORTHLESS_CLOSE_STATE_PATH",
)


@pytest.fixture(autouse=True)
def _isolate_swing_live_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "swing_live_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    for name in _REDIRECTED_STATE_PATHS:
        original = getattr(pm_module, name)
        monkeypatch.setattr(pm_module, name, state_dir / original.name)
    yield state_dir
