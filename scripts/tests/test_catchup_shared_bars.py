"""catchup_shared_bars: batched 1H by default, fetch_one only for the leftovers, and a
completion stamp only for a universe run (2026-11-22)."""
import json

import pandas as pd

import core.live_readiness as lr
from scripts import catchup_shared_bars as cs


def _wire(monkeypatch, tmp_path, batch_status):
    calls = {"batched": [], "one": [], "4h": []}
    monkeypatch.setattr(cs, "STATUS", tmp_path / "status.txt")
    monkeypatch.setattr(lr, "DEFAULT_INTRADAY_BARS_STAMP", tmp_path / "stamp.json")
    monkeypatch.setattr(cs.bars, "fetch_intraday_batched",
                        lambda tickers, **k: calls["batched"].append(list(tickers)) or dict(batch_status))
    monkeypatch.setattr(cs.bars, "fetch_one", lambda **k: calls["one"].append(k["ticker"]) or None)
    monkeypatch.setattr(cs.bars, "build_4h_for", lambda t, **k: calls["4h"].append(t) or None)
    universe = tmp_path / "u.csv"
    pd.DataFrame({"ticker": list(batch_status), "is_eligible": [True] * len(batch_status)}).to_csv(universe, index=False)
    return calls, universe


def test_batched_then_fallback_then_stamp(monkeypatch, tmp_path):
    status = {"AAPL": "updated", "MSFT": "fresh", "NEWCO": "uncached", "BAD": "error"}
    calls, universe = _wire(monkeypatch, tmp_path, status)
    end = "2026-11-20T19:00:30+00:00"

    assert cs.main(["--universe", str(universe), "--no-1d", "--eligible-only", "--end", end]) == 0

    assert calls["batched"] == [["AAPL", "BAD", "MSFT", "NEWCO"]]
    assert sorted(calls["one"]) == ["BAD", "NEWCO"]           # only the leftovers one by one
    assert sorted(calls["4h"]) == sorted(status)              # every ticker's 4H rebuilt
    stamp = json.loads((tmp_path / "stamp.json").read_text())
    assert stamp["end_utc"] == "2026-11-20T19:00:30+00:00" and stamp["scope"] == "eligible"


def test_ticker_subset_writes_no_stamp(monkeypatch, tmp_path):
    calls, universe = _wire(monkeypatch, tmp_path, {"AAPL": "updated"})
    assert cs.main(["--tickers", "AAPL", "--no-1d", "--end", "2026-11-20T19:00:30+00:00"]) == 0
    assert not (tmp_path / "stamp.json").exists()


def test_per_ticker_flag_skips_batching(monkeypatch, tmp_path):
    calls, universe = _wire(monkeypatch, tmp_path, {"AAPL": "updated", "MSFT": "fresh"})
    assert cs.main(["--universe", str(universe), "--no-1d", "--per-ticker",
                    "--end", "2026-11-20T19:00:30+00:00"]) == 0
    assert calls["batched"] == [] and sorted(calls["one"]) == ["AAPL", "MSFT"]
