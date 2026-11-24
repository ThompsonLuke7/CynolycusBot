"""CBOE snapshot: a blocked source (HTTP 403 on every ticker) must fail fast."""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from signals.news import sources


def _forbidden(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 403, "Forbidden", {}, io.BytesIO(b""))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(sources.time, "sleep", lambda _s: None)


def test_consecutive_403s_raise_without_retrying(monkeypatch):
    calls: list[str] = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        raise _forbidden(req.full_url)

    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen)
    tickers = [f"T{i}" for i in range(40)]
    with pytest.raises(RuntimeError, match="HTTP 403 for 5 consecutive"):
        sources.fetch_cboe_options_snapshot(tickers, min_interval_s=0, max_consecutive_forbidden=5)
    # One request per ticker (no retries on 403), stopping at the 5th.
    assert len(calls) == 5


def test_isolated_403s_do_not_abort(monkeypatch):
    payload = json.dumps({"data": {}}).encode()

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout):
        if "/BAD" in req.full_url:
            raise _forbidden(req.full_url)
        return _Resp(payload)

    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sources, "_cboe_options_aggregate", lambda p, sigma_threshold: (_ for _ in ()).throw(ValueError()))
    tickers = ["BADA", "AAA", "BADB", "BBB", "BADC", "CCC"]
    summary, records, strikes = sources.fetch_cboe_options_snapshot(
        tickers, min_interval_s=0, max_consecutive_forbidden=2
    )
    assert summary.empty and records.empty and strikes.empty
