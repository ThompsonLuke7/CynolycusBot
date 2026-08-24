"""Option positions carry a mid mark alongside the broker's bid mark.

Alpaca prices an option position at the BID. Verified 2026-08-22 against live
quotes for all 12 open contracts: `current_price` matched the bid to within two
cents in 12 of 12. Entries fill at or above the mid, so every option position
books the full bid-ask spread as an unrealized loss on day one.

On 2026-08-21 that made the day's eight fresh option positions read -$8,582 when
the mid said -$4,304 — on a session where SPY closed -0.05% and SPGI's
underlying closed UP 0.25% while its 0.40-delta call showed -35%. The daily
review attributed that to execution before the marks were checked.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from core.broker_equity_snapshot import capture_snapshot


class _Client:
    """Two option positions and one equity, with a quote feed that can fail."""

    def __init__(self, *, quotes=None, raise_on_quotes=False):
        self._quotes = quotes or {}
        self._raise = raise_on_quotes
        self.quote_calls = []

    def get_account(self):
        return {"equity": "100000", "last_equity": "100000", "cash": "0",
                "buying_power": "0", "portfolio_value": "100000"}

    def get_positions(self):
        return [
            {"symbol": "SPGI260918C00440000", "asset_class": "us_option", "qty": "6",
             "side": "long", "avg_entry_price": "8.50", "current_price": "5.50",
             "market_value": "3300", "cost_basis": "5100"},
            {"symbol": "BP260911C00046000", "asset_class": "us_option", "qty": "61",
             "side": "long", "avg_entry_price": "0.85", "current_price": "0.55",
             "market_value": "3355", "cost_basis": "5185"},
            {"symbol": "CAPR", "asset_class": "us_equity", "qty": "795",
             "side": "long", "avg_entry_price": "6.16", "current_price": "6.38",
             "market_value": "5072.1", "cost_basis": "4897.2"},
        ]

    def get_orders(self, **_k):
        return []

    def get_option_quotes(self, **params):
        self.quote_calls.append(params.get("symbols"))
        if self._raise:
            raise RuntimeError("quote feed down")
        return {"quotes": self._quotes}


_LIVE_QUOTES = {
    "SPGI260918C00440000": {"bp": 5.48, "ap": 8.75},   # 46% spread
    "BP260911C00046000": {"bp": 0.53, "ap": 0.97},     # 59% spread
}


def _capture(tmp_path, client):
    return capture_snapshot(
        client=client, account_label="paper", root=Path(tmp_path),
        now=datetime(2026, 8, 21, 20, 5, tzinfo=timezone.utc),
    )


def _rows(record):
    return {row["symbol"]: row for row in record["positions"]}


def test_option_rows_carry_bid_ask_and_mid(tmp_path):
    import json
    _capture(tmp_path, _Client(quotes=_LIVE_QUOTES))
    written = json.loads(
        (Path(tmp_path) / "broker_equity_20260821_paper.jsonl").read_text().strip()
    )
    rows = _rows(written)

    spgi = rows["SPGI260918C00440000"]
    assert spgi["bid"] == 5.48 and spgi["ask"] == 8.75
    assert spgi["mid_mark"] == 7.115
    assert round(spgi["spread_pct_mid"], 3) == 0.460
    # 6 contracts x 100 x 7.115 = 4269 against a 5100 basis. -831 is exactly the
    # figure the live re-mark produced on 2026-08-22, against -1800 at the bid.
    assert spgi["mid_market_value"] == 4269.0
    assert spgi["unrealized_pl_at_mid"] == -831.0
    # The broker's own bid mark is untouched.
    assert spgi["current_price"] == "5.50"
    assert spgi["market_value"] == "3300"


def test_equity_rows_are_not_touched(tmp_path):
    record = _capture(tmp_path, _Client(quotes=_LIVE_QUOTES))
    capr = _rows(record)["CAPR"]
    assert "mid_mark" not in capr and "unrealized_pl_at_mid" not in capr


def test_portfolio_understatement_is_reported(tmp_path):
    record = _capture(tmp_path, _Client(quotes=_LIVE_QUOTES))
    # SPGI 4269 - 3300 = 969; BP 61*100*0.75 = 4575 - 3355 = 1220.
    assert record["option_bid_vs_mid_understatement"] == 2189.0
    assert record["option_positions_priced_at_mid"] == 2


def test_a_quote_outage_still_produces_the_snapshot(tmp_path):
    """The mark record is the durable artefact; quotes are an enrichment."""
    record = _capture(tmp_path, _Client(raise_on_quotes=True))
    assert len(record["positions"]) == 3
    assert "option_bid_vs_mid_understatement" not in record
    assert "mid_mark" not in _rows(record)["SPGI260918C00440000"]


def test_a_crossed_or_zero_quote_is_ignored(tmp_path):
    record = _capture(tmp_path, _Client(quotes={
        "SPGI260918C00440000": {"bp": 0.0, "ap": 8.75},     # no bid
        "BP260911C00046000": {"bp": 1.20, "ap": 0.97},      # crossed
    }))
    rows = _rows(record)
    assert "mid_mark" not in rows["SPGI260918C00440000"]
    assert "mid_mark" not in rows["BP260911C00046000"]


def test_equity_only_book_never_calls_the_quote_feed(tmp_path):
    client = _Client(quotes=_LIVE_QUOTES)
    client.get_positions = lambda: [
        {"symbol": "CAPR", "asset_class": "us_equity", "qty": "795", "side": "long",
         "avg_entry_price": "6.16", "current_price": "6.38",
         "market_value": "5072.1", "cost_basis": "4897.2"},
    ]
    _capture(tmp_path, client)
    assert client.quote_calls == []
