"""Repo-wide pytest guards.

Keeps the test suite from writing into live trading artefacts.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_live_ledgers(tmp_path, monkeypatch):
    """Point every 4H ledger/queue write at tmp_path for the duration of a test.

    core/tests/test_live_4h_exec_race_guard.py and
    core/tests/test_live_4h_exec_persist_on_fill.py call
    execute_plan(module="dealer_ranker") without a ledger_root. That default used
    to be the literal "Data/inference", so each pytest run appended synthetic
    "AAA" fills to the REAL Data/inference/dealer_ranker/closed_trades.jsonl:
    by 2026-08-03 that live P&L ledger held 38 test rows against 3 genuine
    trades (PSKY, BE, FIG), and a run at 22:09 ET that evening added more.

    Redirecting the module default rather than fixing those two call sites means
    a future test cannot reintroduce the leak by forgetting the argument.
    """
    try:
        import core.live_4h_exec as live_4h_exec
    except Exception:  # module not importable in this environment — nothing to guard
        return
    monkeypatch.setattr(live_4h_exec, "DEFAULT_LEDGER_ROOT", str(tmp_path / "ledger"))
