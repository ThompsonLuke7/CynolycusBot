"""Per-phase timing for the SPY bar handler.

The aggregate `loop cost` line proved a stall but could not attribute it. On
2026-09-02 and 2026-09-03 the live loop sat at busy=100% from 09:50 to 16:12 ET
with per-bar maxima of 230-1,366 seconds and the queue backing to 271, while
nine of every ten bars in the same window cost about 3 seconds between them —
one phase stalling, not general slowness. The meta feature frame build was
measured at 0.87s over the full 50,000-bar buffer and ruled out.
"""

from __future__ import annotations

import logging

import pytest

from UI.live_dashboard import SLOW_PHASE_WARN_SECONDS, _timed_phase


def test_a_fast_phase_is_silent(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert _timed_phase("cheap", "SPY", lambda: "done") == "done"
    assert caplog.records == []


def test_a_slow_phase_names_itself(caplog, monkeypatch) -> None:
    ticks = iter([0.0, SLOW_PHASE_WARN_SECONDS + 1.0])
    monkeypatch.setattr("UI.live_dashboard.time.monotonic", lambda: next(ticks))

    with caplog.at_level(logging.WARNING):
        _timed_phase("policy.on_decision", "SPY", lambda: None)

    assert any("SLOW PHASE policy.on_decision" in r.message for r in caplog.records)


def test_arguments_and_return_value_pass_through() -> None:
    assert _timed_phase("f", "SPY", lambda a, *, b: (a, b), 1, b=2) == (1, 2)


def test_a_raising_phase_is_still_timed_and_still_raises(caplog, monkeypatch) -> None:
    """Timing must never swallow a failure, and a stall that ends in an
    exception is exactly the case worth seeing."""

    ticks = iter([0.0, SLOW_PHASE_WARN_SECONDS + 1.0])
    monkeypatch.setattr("UI.live_dashboard.time.monotonic", lambda: next(ticks))

    def _boom():
        raise ValueError("broker timeout")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="broker timeout"):
            _timed_phase("snapshot_broker_state", "SPY", _boom)

    assert any("SLOW PHASE snapshot_broker_state" in r.message for r in caplog.records)
