"""The runner stage's timeout budget, and what happens when it is exceeded.

On 2026-08-25 the 14:20 ET Meta pass was killed at 901s against a 900s budget
and exited 1, so Meta produced no order plan for the 14:00 UTC bar at all — no
entries, no exits, no audit row. The 16:20 pass ran the same command in 23s.
See research/daily_live_reports/2026-08-25.md.
"""
from __future__ import annotations

import importlib
import logging
import subprocess

import pytest

from signals.meta_context.meta_ranker import run_4h_loop


def test_the_runner_budget_exceeds_the_observed_worst_case() -> None:
    """901s actually happened. A budget at or below it fails the same way."""

    assert run_4h_loop.RUNNER_TIMEOUT_SEC > 901


def test_the_budget_is_configurable_without_a_code_change(monkeypatch) -> None:
    monkeypatch.setenv("META_RANKER_RUNNER_TIMEOUT_SEC", "2400")
    reloaded = importlib.reload(run_4h_loop)
    try:
        assert reloaded.RUNNER_TIMEOUT_SEC == 2400
    finally:
        monkeypatch.delenv("META_RANKER_RUNNER_TIMEOUT_SEC", raising=False)
        importlib.reload(run_4h_loop)


def test_a_timeout_is_a_failure_and_says_so_at_error(monkeypatch, caplog) -> None:
    """A killed stage is not an ordinary non-zero exit: it was cut off part-way,
    so its work is neither complete nor undone. The log has to distinguish them.
    """

    def _boom(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd=["runner"], timeout=900)

    monkeypatch.setattr(run_4h_loop.subprocess, "run", _boom)

    with caplog.at_level(logging.ERROR):
        rc = run_4h_loop._run_subprocess("4/4 runner", ["runner"], timeout=900)

    assert rc == 1
    assert "KILLED" in caplog.text
    assert "META_RANKER_RUNNER_TIMEOUT_SEC" in caplog.text


def test_an_ordinary_failure_is_still_reported_without_the_timeout_language(
    monkeypatch, caplog
) -> None:
    class _Result:
        returncode = 2

    monkeypatch.setattr(run_4h_loop.subprocess, "run", lambda *_a, **_k: _Result())

    with caplog.at_level(logging.ERROR):
        rc = run_4h_loop._run_subprocess("4/4 runner", ["runner"], timeout=900)

    assert rc == 2
    assert "KILLED" not in caplog.text
