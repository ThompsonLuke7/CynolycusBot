import re
from pathlib import Path

import pytest

from core.personal_portfolio import holdings as h

REPO = Path(__file__).resolve().parents[3]


def _raw_account():
    return [{"securitiesAccount": {
        "accountNumber": "12345678", "type": "MARGIN", "isDayTrader": False, "isClosingOnlyRestricted": False,
        "currentBalances": {"liquidationValue": 10_000.0, "availableFunds": 2_000.0, "cashBalance": 1_500.0,
                            "buyingPower": 4_000.0, "maintenanceRequirement": 3_000.0},
        "positions": [
            {"instrument": {"assetType": "OPTION", "symbol": "XYZ   261120C00050000", "underlyingSymbol": "XYZ",
                            "putCall": "CALL", "description": "XYZ INC 11/20/2026 $50 Call"},
             "longQuantity": 2.0, "shortQuantity": 0.0, "averageLongPrice": 1.25, "marketValue": 300.0,
             "longOpenProfitLoss": 50.0, "currentDayProfitLoss": -20.0},
            {"instrument": {"assetType": "COLLECTIVE_INVESTMENT", "symbol": "XYZL", "type": "EXCHANGE_TRADED_FUND",
                            "description": "TRADR 2X LONG XYZ DAILY ETF"},
             "longQuantity": 10.0, "shortQuantity": 0.0, "averageLongPrice": 40.0, "marketValue": 500.0,
             "longOpenProfitLoss": 100.0, "currentDayProfitLoss": 5.0},
        ]}, "aggregatedBalance": {"liquidationValue": 10_000.0}}]


def test_parse_occ_symbol():
    assert h.parse_occ_symbol("SOXL  261218C00070000") == {"root": "SOXL", "expiry": "2026-12-18",
                                                           "put_call": "CALL", "strike": 70.0}
    assert h.parse_occ_symbol("BRK.B 270115P00412500")["strike"] == 412.5
    assert h.parse_occ_symbol("AAPL") is None


def test_normalize_accounts_masks_the_account_and_derives_cost():
    acct = h.normalize_accounts(_raw_account())[0]
    assert acct["account"] == "…678" and "12345678" not in str(acct)
    opt, etf = acct["positions"]
    assert opt["kind"] == "option" and opt["underlying"] == "XYZ" and opt["expiry"] == "2026-11-20"
    assert opt["strike"] == 50.0 and opt["multiplier"] == 100.0 and opt["cost_basis"] == pytest.approx(250.0)
    assert opt["open_pl_pct"] == pytest.approx(0.2)
    assert etf["kind"] == "etf" and etf["cost_basis"] == 400.0 and etf["quantity"] == 10.0
    assert acct["balances"]["available_funds"] == 2_000.0


def test_normalize_rejects_an_unexpected_shape():
    with pytest.raises(ValueError):
        h.normalize_accounts({"error": "nope"})
    with pytest.raises(ValueError):
        h.normalize_accounts([{"other": {}}])


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self.payload


class _FakeClient:
    """Only the two GET methods exist; touching anything else (an order method) fails the test."""

    class Account:
        class Fields:
            POSITIONS = "positions"

    def __init__(self, status=200):
        self.calls, self.status = [], status

    def get_accounts(self, *, fields=None):
        self.calls.append(("get_accounts", tuple(fields)))
        return _Resp(_raw_account(), self.status)

    def get_quotes(self, symbols):
        self.calls.append(("get_quotes", tuple(symbols)))
        return _Resp({"XYZ   261120C00050000": {"assetMainType": "OPTION",
                                                 "quote": {"mark": 1.5, "delta": 0.6, "underlyingPrice": 52.0,
                                                           "bidPrice": 1.4, "askPrice": 1.6, "theta": -0.05},
                                                 "reference": {"daysToExpiration": 8, "multiplier": 100.0,
                                                               "strikePrice": 50.0}},
                      "XYZL": {"assetMainType": "EQUITY", "assetSubType": "ETF",
                               "quote": {"mark": 50.0}, "reference": {"description": "2X LONG XYZ"}}})

    def __getattr__(self, name):
        raise AssertionError(f"holdings touched a non-read method: {name}")


def test_fetch_holdings_reads_only_and_enriches_quotes():
    client = _FakeClient()
    out = h.fetch_holdings(client, extra_symbols=["abc", "SPY"])
    assert [c[0] for c in client.calls] == ["get_accounts", "get_quotes"]
    assert set(client.calls[1][1]) == {"XYZ   261120C00050000", "XYZL", "XYZ", "ABC", "SPY"}
    opt = out["accounts"][0]["positions"][0]
    assert opt["quote"]["delta"] == 0.6 and opt["quote"]["dte"] == 8.0
    assert out["quotes"]["XYZL"]["asset_sub_type"] == "ETF"


def test_fetch_holdings_fails_fast_on_http_error():
    with pytest.raises(RuntimeError, match="HTTP 401"):
        h.fetch_holdings(_FakeClient(status=401))


def test_personal_code_has_no_order_paths():
    files = list((REPO / "core" / "personal_portfolio").glob("*.py")) + [REPO / "UI" / "personal_dashboard.py"]
    pattern = re.compile(r"place_order|replace_order|cancel_order|equity_buy|equity_sell|option_buy|option_sell"
                         r"|submit_order|AlpacaOptionsClient|TradingClient")
    hits = [f.name for f in files if f.exists() and pattern.search(f.read_text())]
    assert hits == []
