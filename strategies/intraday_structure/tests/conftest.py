from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _market_open_for_entries(monkeypatch):
    """The executor refuses entries while the market is closed. Tests here are
    about what happens after that gate, so hold it open whatever the wall clock
    says; a test of the gate itself restores the real calendar function."""
    import strategies.intraday_structure.execution as execution

    monkeypatch.setattr(execution, "is_market_open_now", lambda *_a, **_k: True)
