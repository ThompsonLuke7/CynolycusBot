"""The dealer runner must not retry a credential it knows is dead.

2026-07-30: an expired Schwab refresh token produced 535 identical
``invalid_grant`` failures across five symbols in 1h54m. Renewal is an
interactive browser login, so not one of those retries could have succeeded.
"""

from __future__ import annotations

import queue
import threading

import pytest

from strategies.dealer_positioning import runner as dealer_runner
from strategies.dealer_positioning.config import DealerPositioningConfig
from strategies.dealer_positioning.runner import DealerPositioningRunner, _is_dead_credential

DEAD = (
    'unsupported_token_type: 400 Bad Request: "{"error_description":'
    '"Refresh token is invalid, expired or revoked","error":"invalid_grant"}"'
)


@pytest.mark.parametrize("message", [
    DEAD,
    "invalid_grant",
    "Refresh token is invalid, expired or revoked",
    "unsupported_token_type",
])
def test_dead_credential_errors_are_recognised(message):
    assert _is_dead_credential(RuntimeError(message))


@pytest.mark.parametrize("message", [
    "connection reset by peer",
    "429 Too Many Requests",
    "500 Internal Server Error",
    "timed out",
])
def test_transient_errors_are_not_treated_as_dead(message):
    """A transient failure must keep retrying — only auth death is terminal."""
    assert not _is_dead_credential(RuntimeError(message))


class _ValidToken:
    expired = False
    expiring_soon = False
    message = "Schwab refresh token valid for 5d 2h"


@pytest.fixture()
def runner(monkeypatch, tmp_path):
    """A runner with no real Schwab client, polling the five live symbols.

    The token status is stubbed valid so these tests exercise the *loop* and do
    not depend on whether the repo's real token happens to be live today.
    """
    monkeypatch.setattr(
        dealer_runner, "SchwabDealerDataClient", lambda config: object()
    )
    monkeypatch.setattr(
        "core.schwab_token_status.schwab_token_status", lambda *a, **k: _ValidToken()
    )
    config = DealerPositioningConfig(
        symbols=("SPY", "QQQ", "SLV", "IWM", "GLD"),
        poll_seconds=60,
        market_hours_only=False,
        output_root=tmp_path / "dealer",
    )
    inst = DealerPositioningRunner(
        config=config, data_client=object(), event_sink=None, bar_queue=queue.Queue()
    )
    inst._stop = threading.Event()
    return inst


def _run_one_pass(inst, monkeypatch, polls: list[str]):
    """Drive exactly one iteration of the polling loop."""
    def poll(symbol):
        polls.append(symbol)
        raise RuntimeError(DEAD)

    monkeypatch.setattr(inst, "_poll_symbol", poll)
    monkeypatch.setattr(inst, "_drain_bar_queue", lambda: None)
    monkeypatch.setattr(inst, "_finalize_expired_trades", lambda now: None)
    monkeypatch.setattr(inst, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(inst, "snapshot", lambda: {})

    original_wait = inst._stop.wait

    def wait_then_stop(timeout=None):
        inst._stop.set()
        return original_wait(0)

    monkeypatch.setattr(inst._stop, "wait", wait_then_stop)
    inst.start()


def test_first_dead_credential_stops_the_whole_pass(runner, monkeypatch):
    """One failure halts the pass — the other four symbols are not attempted."""
    polls: list[str] = []
    _run_one_pass(runner, monkeypatch, polls)

    assert polls == ["SPY"], "should stop after the first dead-credential error"
    assert runner._auth_dead_reason is not None


def test_polling_does_not_resume_on_later_iterations(runner, monkeypatch):
    """The flag must survive across loop iterations, not just the failing pass."""
    runner._auth_dead_reason = "already dead"
    polls: list[str] = []

    monkeypatch.setattr(runner, "_poll_symbol", lambda s: polls.append(s))
    monkeypatch.setattr(runner, "_drain_bar_queue", lambda: None)
    monkeypatch.setattr(runner, "_finalize_expired_trades", lambda now: None)
    monkeypatch.setattr(runner, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(runner, "snapshot", lambda: {})

    def wait_then_stop(timeout=None):
        runner._stop.set()
        return False

    monkeypatch.setattr(runner._stop, "wait", wait_then_stop)
    runner.start()

    assert polls == [], "no request may be issued once the credential is known dead"


def test_expired_token_at_startup_prevents_every_request(runner, monkeypatch):
    """Zero doomed requests: the token file already says it is dead."""
    class Expired:
        expired = True
        message = "Schwab refresh token EXPIRED 21h ago"

    monkeypatch.setattr(
        "core.schwab_token_status.schwab_token_status", lambda *a, **k: Expired()
    )
    emitted: list[tuple] = []
    monkeypatch.setattr(runner, "_emit", lambda kind, payload: emitted.append((kind, payload)))

    runner._halt_early_if_token_already_expired()

    assert runner._auth_dead_reason is not None
    assert emitted and emitted[0][0] == "auth_halted"
    assert "--reauth" in emitted[0][1]["reauth_command"]


def test_valid_token_at_startup_leaves_polling_enabled(runner, monkeypatch):
    class Valid:
        expired = False
        message = "Schwab refresh token valid for 5d 2h"

    monkeypatch.setattr(
        "core.schwab_token_status.schwab_token_status", lambda *a, **k: Valid()
    )

    runner._halt_early_if_token_already_expired()

    assert runner._auth_dead_reason is None


def test_transient_error_does_not_halt_polling(runner, monkeypatch):
    """A 429 must not disable the module for the rest of the day."""
    polls: list[str] = []

    def poll(symbol):
        polls.append(symbol)
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(runner, "_poll_symbol", poll)
    monkeypatch.setattr(runner, "_drain_bar_queue", lambda: None)
    monkeypatch.setattr(runner, "_finalize_expired_trades", lambda now: None)
    monkeypatch.setattr(runner, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(runner, "snapshot", lambda: {})

    def wait_then_stop(timeout=None):
        runner._stop.set()
        return False

    monkeypatch.setattr(runner._stop, "wait", wait_then_stop)
    runner.start()

    assert polls == ["SPY", "QQQ", "SLV", "IWM", "GLD"], "all symbols still attempted"
    assert runner._auth_dead_reason is None


def test_dead_token_at_startup_never_builds_a_schwab_client(monkeypatch, tmp_path):
    """The token check must run *before* the client is constructed.

    2026-08-24: an interrupted re-auth left no token file, so
    ``SchwabDealerDataClient`` dropped into schwab-py's interactive login and
    the runner thread died on ``EOFError`` at ``runner.start()`` — before the
    cheap, offline token check that would have reported the real cause.
    """
    class Expired:
        expired = True
        message = "Schwab refresh token EXPIRED 21h ago"

    monkeypatch.setattr(
        "core.schwab_token_status.schwab_token_status", lambda *a, **k: Expired()
    )

    def _explode(config):
        raise AssertionError("no Schwab client may be built on a dead credential")

    monkeypatch.setattr(dealer_runner, "SchwabDealerDataClient", _explode)

    config = DealerPositioningConfig(
        symbols=("SPY",),
        poll_seconds=60,
        market_hours_only=False,
        output_root=tmp_path / "dealer",
    )
    inst = DealerPositioningRunner(config=config, event_sink=None, bar_queue=queue.Queue())
    inst._stop = threading.Event()
    inst._stop.set()

    inst.start()

    assert inst._data_client is None
    assert inst._auth_dead_reason is not None
