from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from strategies.dealer_positioning.iv_surface import (
    clean_contracts,
    flatten_chain,
    interpolate_atm_iv,
    surface_features,
)
from strategies.dealer_positioning.scripts.capture_iv_surface import capture_iv_surface

pytestmark = pytest.mark.safe

_ET = ZoneInfo("America/New_York")
_SPOT = 100.0
_RATE = 0.04


def _bs(strike: float, years: float, sigma: float, kind: str) -> tuple[float, float]:
    """European Black-Scholes price and delta."""
    n = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # noqa: E731
    d1 = (math.log(_SPOT / strike) + (_RATE + 0.5 * sigma**2) * years) / (sigma * math.sqrt(years))
    d2 = d1 - sigma * math.sqrt(years)
    if kind == "C":
        return _SPOT * n(d1) - strike * math.exp(-_RATE * years) * n(d2), n(d1)
    return strike * math.exp(-_RATE * years) * n(-d2) - _SPOT * n(-d1), n(d1) - 1.0


def _chain(*, dtes=(30, 60), iv=lambda strike, dte, kind: 0.30, oi=100, quoted_vol=None) -> dict:
    """A Schwab-shaped chain whose quotes are exact Black-Scholes prices (mid = model price).

    ``quoted_vol`` overrides the ``volatility`` field, mimicking Schwab's one-IV-per-strike quote.
    """
    strikes = [60.0 + i for i in range(81)]
    maps: dict[str, dict] = {"C": {}, "P": {}}
    for dte in dtes:
        key = f"{(date(2026, 9, 10) + timedelta(days=dte)).isoformat()}:{dte}"
        for kind in ("C", "P"):
            side = maps[kind].setdefault(key, {})
            for strike in strikes:
                sigma = iv(strike, dte, kind)
                price, delta = _bs(strike, dte / 365.0, sigma, kind)
                half = max(0.005, 0.01 * price)
                side[str(strike)] = [
                    {
                        "putCall": "CALL" if kind == "C" else "PUT",
                        "strikePrice": strike,
                        "bid": price - half,
                        "ask": price + half,
                        "mark": price,
                        "volatility": (quoted_vol if quoted_vol is not None else sigma) * 100.0,
                        "delta": delta,
                        "openInterest": oi,
                        "totalVolume": 10,
                        "daysToExpiration": dte,
                        "quoteTimeInLong": 1789070400000,
                        "nonStandard": False,
                        "mini": False,
                    }
                ]
    return {
        "underlyingPrice": _SPOT,
        "interestRate": _RATE * 100.0,
        "dividendYield": 0.0,
        "isDelayed": False,
        "callExpDateMap": maps["C"],
        "putExpDateMap": maps["P"],
    }


def _features(chain: dict) -> dict:
    features, _ = surface_features(chain, symbol="tst", snapshot_date="2026-09-10", captured_at="2026-09-10T16:45:00-04:00")
    return features


def test_flatten_keeps_raw_values_and_cleaning_drops_sentinels_and_one_sided_quotes():
    chain = _chain(dtes=(30,))
    first_call = next(iter(chain["callExpDateMap"].values()))
    first_call["100.0"][0]["volatility"] = -999.0  # Schwab's "no IV" sentinel
    first_call["101.0"][0]["bid"] = 0.0  # one-sided quote

    raw = flatten_chain(chain)
    assert len(raw) == 162
    assert set(raw["option_type"]) == {"C", "P"}
    assert raw.loc[(raw.strike == 100.0) & (raw.option_type == "C"), "volatility_pct"].item() == -999.0
    assert raw["quote_time_utc"].iloc[0] == pd.Timestamp("2026-09-10T20:00:00Z")

    clean = clean_contracts(raw)
    calls = clean[clean.option_type == "C"]
    assert 100.0 not in set(calls.strike)
    assert 101.0 not in set(calls.strike)
    assert clean["iv"].between(0.01, 5.0).all()
    assert clean["delta"].abs().between(0.05, 0.95).all()


def test_flat_surface_has_no_spread_smirk_slope_or_skew():
    features = _features(_chain())

    assert features["iv_atm_30"] == pytest.approx(0.30)
    assert features["iv_atm_60"] == pytest.approx(0.30)
    assert features["iv_term_slope_30_60"] == pytest.approx(0.0, abs=1e-12)
    assert features["cw_iv_spread_30"] == pytest.approx(0.0, abs=1e-9)
    assert features["smirk_30"] == pytest.approx(0.0, abs=1e-12)
    # Black-Scholes log returns are normal, so the true skewness is 0; what is
    # left is strike discretisation and the delta-filter truncation.
    assert abs(features["rn_skew_30"]) < 0.15
    assert features["surface_dte_30"] == 30
    assert features["atm_spread_pct_30"] == pytest.approx(0.02, rel=1e-6)


def test_calls_richer_than_puts_gives_positive_cw_spread():
    features = _features(_chain(iv=lambda strike, dte, kind: 0.32 if kind == "C" else 0.30))
    assert features["cw_iv_spread_30"] == pytest.approx(0.02, abs=1e-9)


def test_cw_spread_reads_prices_not_schwabs_per_strike_iv():
    # Schwab quotes one IV per strike for both sides; the spread lives only in the prices.
    chain = _chain(iv=lambda strike, dte, kind: 0.30 if kind == "C" else 0.33, quoted_vol=0.31)
    features = _features(chain)
    assert features["cw_iv_spread_30"] == pytest.approx(-0.03, abs=1e-9)
    assert features["iv_atm_30"] == pytest.approx(0.31)


def test_put_skew_raises_smirk_and_makes_risk_neutral_skew_negative():
    flat = _features(_chain())
    skewed = _features(_chain(iv=lambda strike, dte, kind: 0.30 + 0.8 * max(0.0, (100.0 - strike) / 100.0)))

    assert skewed["smirk_30"] == pytest.approx(0.8 * 0.05)
    assert skewed["cw_iv_spread_30"] == pytest.approx(0.0, abs=1e-9)
    assert skewed["rn_skew_30"] < flat["rn_skew_30"] - 0.2
    assert skewed["rn_skew_30"] < 0


def test_atm_iv_interpolates_in_total_variance_and_never_extrapolates_far():
    shapes = pd.DataFrame({"dte": [20.0, 40.0], "atm_iv": [0.20, 0.30]})
    expected = math.sqrt((0.20**2 * 20 + (0.30**2 * 40 - 0.20**2 * 20) * 0.5) / 30)

    assert interpolate_atm_iv(shapes, 30) == pytest.approx(expected)
    assert interpolate_atm_iv(shapes, 45) == pytest.approx(0.30)  # 5d past the last expiry: flat
    assert math.isnan(interpolate_atm_iv(shapes, 60))  # 20d past: refuse


def test_missing_spot_fails_fast():
    chain = _chain()
    chain["underlyingPrice"] = None
    with pytest.raises(ValueError, match="underlyingPrice"):
        _features(chain)


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def get_option_chain(self, symbol, ref_date, *, from_date, to_date):
        self.calls.append((symbol, from_date, to_date))
        if symbol == "BAD":
            raise RuntimeError("Schwab chain lookup failed")
        return _chain()


def test_capture_writes_features_contracts_and_isolates_failures(tmp_path):
    client = _FakeClient()
    now = datetime(2026, 9, 10, 16, 45, tzinfo=_ET)

    result = capture_iv_surface(symbols=["AAA", "BAD"], client=client, output_root=tmp_path, sleep_seconds=0, now=now)

    out = tmp_path / "20260910"
    assert result.feature_rows == 1 and result.errors == 1
    assert client.calls[0] == ("AAA", date(2026, 9, 10), date(2027, 1, 8))
    features = pd.read_parquet(out / "iv_surface_features.parquet")
    assert list(features["symbol"]) == ["AAA"]
    assert features["feature_version"].item() == "iv_surface_v1"
    contracts = pd.read_parquet(out / "iv_contracts.parquet")
    assert len(contracts) == result.contract_rows == 324
    assert {"bid", "ask", "volatility_pct", "delta", "open_interest"} <= set(contracts.columns)
    errors = [json.loads(line) for line in (out / "errors.jsonl").read_text().splitlines()]
    assert errors[0]["symbol"] == "BAD"

    # A same-day re-run replaces rather than duplicates.
    capture_iv_surface(symbols=["AAA"], client=client, output_root=tmp_path, sleep_seconds=0, now=now)
    assert len(pd.read_parquet(out / "iv_contracts.parquet")) == 324
    assert len(pd.read_parquet(out / "iv_surface_features.parquet")) == 1
