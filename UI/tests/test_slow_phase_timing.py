"""Per-phase timing for the SPY bar handler.

The aggregate `loop cost` line proved a stall but could not attribute it. On
2026-09-02 and 2026-09-03 the live loop sat at busy=100% from 09:50 to 16:12 ET
with per-bar maxima of 230-1,366 seconds and the queue backing to 271, while
nine of every ten bars in the same window cost about 3 seconds between them —
one phase stalling, not general slowness.

It worked: on 2026-09-08 the phase timers named `inference.on_15m_close` (39
bars at 325-506s), and the sub-phase timers exercised at the bottom of this
file then attributed that to `base_frame_build` at 278.6s over 49,951 rows and
`swing_setup_probs` at 85.4s over 4,998 rows. An earlier note claiming the
feature frame build was "measured at 0.87s and ruled out" is contradicted by
that and has been dropped.
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


# --------------------------------------------------------------------------
# One level deeper: which part of inference.on_15m_close is stalling
# --------------------------------------------------------------------------


def test_a_slow_inference_subphase_names_itself(caplog, monkeypatch) -> None:
    """`inference.on_15m_close` was the phase; these say which part of it.

    The phase timers above narrowed the 2026-09 stall to one phase, and on
    2026-09-08 that phase ran 325-506s per SPY bar — enough to push the
    daytrader's decisions 152 minutes behind the tape. Attribution stops there
    without sub-timers, and a perf guess inside the inference path is a guess
    about what the agent decides.
    """

    from core.API.Alpaca_API.inference import live_inference as li

    # Any duration counts as slow, so the test needs no sleep and no clock patch.
    monkeypatch.setattr(li, "INFERENCE_SUBPHASE_WARN_SECONDS", 0.0)

    with caplog.at_level(logging.WARNING):
        with li._timed_subphase("base_frame_build", rows=50_000):
            pass

    assert "SLOW SUBPHASE base_frame_build" in caplog.text, caplog.text
    assert "over 50000 rows" in caplog.text, caplog.text


def test_a_fast_inference_subphase_is_silent(caplog) -> None:
    from core.API.Alpaca_API.inference import live_inference as li

    with caplog.at_level(logging.WARNING):
        with li._timed_subphase("process_rows", rows=1):
            pass
    assert [r for r in caplog.records if "SLOW SUBPHASE" in r.message] == []


def test_a_failing_subphase_still_propagates(caplog) -> None:
    """A diagnostic timer must never swallow the error it was measuring."""

    from core.API.Alpaca_API.inference import live_inference as li

    with pytest.raises(ValueError):
        with li._timed_subphase("swing_setup_probs"):
            raise ValueError("boom")
