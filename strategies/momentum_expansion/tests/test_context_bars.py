import pandas as pd

from strategies.momentum_expansion.data import bars


def test_context_refresh_uses_live_end_not_fixed_training_cutoff(monkeypatch, tmp_path):
    calls = []

    def fake_fetch_one(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(bars, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(bars, "_path_for", lambda ticker, kind: tmp_path / f"{ticker}_{kind}.parquet")

    bars.fetch_context_bars(tickers=("VIXY",), end="2026-07-14T20:00:00+00:00")

    assert calls == [{"ticker": "VIXY", "kind": "context", "force": False, "end": "2026-07-14T20:00:00+00:00"}]


def test_context_refresh_default_end_is_current_not_training_cutoff(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bars, "fetch_one", lambda **kwargs: calls.append(kwargs) or None)
    monkeypatch.setattr(bars, "_path_for", lambda ticker, kind: tmp_path / f"{ticker}_{kind}.parquet")

    bars.fetch_context_bars(tickers=("VIXY",))

    assert pd.Timestamp(calls[0]["end"]) > pd.Timestamp("2026-06-02", tz="UTC")
