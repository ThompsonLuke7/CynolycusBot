"""Optionable-candidate backfill for the Dealer Ranker top-K selection.

Regression coverage for the 7/17 (6/10 skipped) and 7/20 (10/10 skipped, zero
orders for the whole session) `no_non_0dte_call_contracts` shutouts: a fixed
top-K slice of the ranking table can be entirely non-optionable on Alpaca even
though lower-ranked names in the same table are perfectly tradable.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from strategies.dealer_positioning.live_ranked_options import (
    ContractLookupError,
    ScanUnevaluableError,
    _has_tradable_contracts,
    _select_optionable_targets,
)

_TODAY = date(2026, 7, 20)


def _contract(symbol: str, root: str, expiry: date, tradable: bool = True) -> dict:
    return {
        "symbol": symbol,
        "root_symbol": root,
        "expiration_date": expiry.isoformat(),
        "strike_price": "10",
        "multiplier": "100",
        "size": "100",
        "tradable": tradable,
        "type": "call",
    }


class _FakeClient:
    """Maps underlying -> raw contract list; empty list means "no chain"."""

    def __init__(self, chains: dict[str, list[dict]]) -> None:
        self.chains = chains
        self.calls: list[str] = []

    def get_option_contracts(self, *, underlying_symbol: str, **_kwargs) -> dict:
        self.calls.append(underlying_symbol)
        return {"option_contracts": self.chains.get(underlying_symbol.upper(), [])}


def test_has_tradable_contracts_true_when_chain_in_window():
    expiry = _TODAY + timedelta(days=7)
    client = _FakeClient({"AAPL": [_contract("AAPL260727C00200000", "AAPL", expiry)]})
    assert _has_tradable_contracts(
        client, "AAPL", option_type="call", min_dte=1, max_dte=21, now_et=_dt(_TODAY)
    )


def test_has_tradable_contracts_false_when_no_chain():
    client = _FakeClient({})
    assert not _has_tradable_contracts(
        client, "XPO", option_type="call", min_dte=1, max_dte=21, now_et=_dt(_TODAY)
    )


def test_has_tradable_contracts_false_when_only_0dte():
    client = _FakeClient({"XPO": [_contract("XPO260720C00090000", "XPO", _TODAY)]})
    assert not _has_tradable_contracts(
        client, "XPO", option_type="call", min_dte=1, max_dte=21, now_et=_dt(_TODAY)
    )


def test_has_tradable_contracts_false_when_not_tradable():
    expiry = _TODAY + timedelta(days=7)
    client = _FakeClient({"XPO": [_contract("XPO260727C00090000", "XPO", expiry, tradable=False)]})
    assert not _has_tradable_contracts(
        client, "XPO", option_type="call", min_dte=1, max_dte=21, now_et=_dt(_TODAY)
    )


class _RaisingClient:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("boom")
        self.calls: list[str] = []

    def get_option_contracts(self, *, underlying_symbol: str = "", **_kwargs):
        self.calls.append(underlying_symbol)
        raise self.exc


def test_has_tradable_contracts_raises_on_client_error():
    # "We could not read the chain" must not be reported as "this name lists no
    # contracts" — a DNS outage on 2026-08-07 made every scanned name look
    # non-optionable and the module placed zero orders for the session.
    with pytest.raises(ContractLookupError):
        _has_tradable_contracts(
            _RaisingClient(), "XPO", option_type="call", min_dte=1, max_dte=21, now_et=_dt(_TODAY)
        )


def test_select_optionable_targets_aborts_when_lookups_fail_wholesale():
    symbols = [f"T{i}" for i in range(50)]
    client = _RaisingClient(OSError("[Errno -3] Temporary failure in name resolution"))
    rankings = _rankings(symbols)

    with pytest.raises(ScanUnevaluableError) as excinfo:
        _select_optionable_targets(
            client, rankings, top_k=3, side_mode="call", min_dte=1, max_dte=21,
            scan_multiple=4, now_et=_dt(_TODAY),
        )
    # Still bounded by the scan cap — an outage must not turn one pass into an
    # unbounded retry storm against the broker.
    assert len(client.calls) == 12
    assert "could not evaluate" in str(excinfo.value)


def test_select_optionable_targets_tolerates_a_few_lookup_failures():
    # Isolated failures are normal; only bulk failure invalidates the scan.
    expiry = _TODAY + timedelta(days=7)
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    chains = {s: [_contract(f"{s}260727C00010000", s, expiry)] for s in ("BBB", "CCC", "DDD")}

    class _FlakyClient(_FakeClient):
        def get_option_contracts(self, *, underlying_symbol: str, **kwargs):
            if underlying_symbol.upper() == "AAA":
                raise RuntimeError("transient")
            return super().get_option_contracts(underlying_symbol=underlying_symbol, **kwargs)

    top, skipped = _select_optionable_targets(
        _FlakyClient(chains), _rankings(symbols), top_k=2, side_mode="call",
        min_dte=1, max_dte=21, now_et=_dt(_TODAY),
    )
    assert list(top["symbol"]) == ["BBB", "CCC"]
    # AAA was never evaluated, so it is not evidence of non-optionability.
    assert "AAA" not in skipped


def _dt(d: date):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime(d.year, d.month, d.day, 15, 54, tzinfo=ZoneInfo("America/New_York"))


def _rankings(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": symbols,
        "dealer_swing_rank": range(1, len(symbols) + 1),
        "dealer_direction": ["bullish"] * len(symbols),
    })


def test_select_optionable_targets_backfills_past_non_optionable_names():
    # Top-10 style shutout: the first 4 ranked names have no listed chain
    # (like XPO/MTSI/UMC/MOD on 7/20); ranks 5-6 do. top_k=2 should skip the
    # non-optionable heads and keep the first 2 that actually have contracts,
    # in rank order.
    expiry = _TODAY + timedelta(days=7)
    symbols = ["XPO", "MTSI", "UMC", "MOD", "PSKY", "SU"]
    chains = {
        "PSKY": [_contract("PSKY260727C00010000", "PSKY", expiry)],
        "SU": [_contract("SU260727C00060000", "SU", expiry)],
    }
    client = _FakeClient(chains)
    rankings = _rankings(symbols)

    top, skipped = _select_optionable_targets(
        client, rankings, top_k=2, side_mode="call", min_dte=1, max_dte=21, now_et=_dt(_TODAY),
    )
    assert list(top["symbol"]) == ["PSKY", "SU"]
    assert skipped == ["XPO", "MTSI", "UMC", "MOD"]


def test_select_optionable_targets_scan_cap_bounds_api_calls():
    # If nothing in the table is optionable, the scan must stop at
    # top_k * scan_multiple rather than walking the whole universe.
    symbols = [f"T{i}" for i in range(50)]
    client = _FakeClient({})
    rankings = _rankings(symbols)

    top, skipped = _select_optionable_targets(
        client, rankings, top_k=3, side_mode="call", min_dte=1, max_dte=21, scan_multiple=4,
        now_et=_dt(_TODAY),
    )
    assert top.empty
    assert len(skipped) == 12  # top_k(3) * scan_multiple(4)
    assert len(client.calls) == 12


def test_select_optionable_targets_preserves_rank_order_when_all_optionable():
    expiry = _TODAY + timedelta(days=7)
    symbols = ["AAA", "BBB", "CCC"]
    chains = {s: [_contract(f"{s}260727C00010000", s, expiry)] for s in symbols}
    client = _FakeClient(chains)
    rankings = _rankings(symbols)

    top, skipped = _select_optionable_targets(
        client, rankings, top_k=3, side_mode="call", min_dte=1, max_dte=21, now_et=_dt(_TODAY),
    )
    assert list(top["symbol"]) == symbols
    assert skipped == []
