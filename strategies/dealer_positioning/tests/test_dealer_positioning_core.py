from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from strategies.dealer_positioning.chain import parse_schwab_option_chain, target_expiration_dates, trading_expiration_buckets
from strategies.dealer_positioning.config import DealerPositioningConfig
from strategies.dealer_positioning.levels import compute_gamma_levels
from strategies.dealer_positioning.signals import DealerSignalEngine, DealerTradeSimulator, PriceCandle


def _chain_fixture() -> dict:
    return {
        "underlyingPrice": 500.0,
        "callExpDateMap": {
            "2026-06-12:0": {
                "505.0": [{"putCall": "CALL", "strikePrice": 505, "openInterest": 1000, "totalVolume": 50, "gamma": 0.08, "delta": 0.40, "volatility": 18}],
                "510.0": [{"putCall": "CALL", "strikePrice": 510, "openInterest": 5000, "totalVolume": 200, "gamma": 0.10, "delta": 0.30, "volatility": 19}],
                "515.0": [{"putCall": "CALL", "strikePrice": 515, "openInterest": 1200, "totalVolume": 30, "gamma": 0.03, "delta": 0.20, "volatility": 20}],
            },
            "2026-06-13:1": {
                "506.0": [{"putCall": "CALL", "strikePrice": 506, "openInterest": 800, "totalVolume": 60, "gamma": 0.07, "delta": 0.38, "volatility": 17}],
                "511.0": [{"putCall": "CALL", "strikePrice": 511, "openInterest": 4000, "totalVolume": 180, "gamma": 0.09, "delta": 0.28, "volatility": 18}],
            }
        },
        "putExpDateMap": {
            "2026-06-12:0": {
                "495.0": [{"putCall": "PUT", "strikePrice": 495, "openInterest": 1000, "totalVolume": 40, "gamma": 0.06, "delta": -0.35, "volatility": 19}],
                "490.0": [{"putCall": "PUT", "strikePrice": 490, "openInterest": 7000, "totalVolume": 300, "gamma": 0.09, "delta": -0.25, "volatility": 20}],
                "485.0": [{"putCall": "PUT", "strikePrice": 485, "openInterest": 1300, "totalVolume": 20, "gamma": 0.02, "delta": -0.15, "volatility": 21}],
            },
            "2026-06-13:1": {
                "496.0": [{"putCall": "PUT", "strikePrice": 496, "openInterest": 900, "totalVolume": 30, "gamma": 0.05, "delta": -0.33, "volatility": 18}],
                "491.0": [{"putCall": "PUT", "strikePrice": 491, "openInterest": 4500, "totalVolume": 210, "gamma": 0.08, "delta": -0.24, "volatility": 19}],
            }
        },
    }


def test_parse_schwab_chain_filters_dte_and_computes_levels() -> None:
    spot, rows = parse_schwab_option_chain(
        _chain_fixture(),
        symbol="SPY",
        timestamp=datetime(2026, 6, 12, 14, 30, tzinfo=timezone.utc),
        dte_offsets={0, 1, 2},
    )
    assert spot == 500.0
    assert len(rows) == 10

    ladder, levels = compute_gamma_levels(rows, symbol="SPY", spot=spot)
    assert len(ladder) >= 8
    assert levels.call_wall == 510.0
    assert levels.put_wall == 490.0
    assert levels.nearest_magnet in {490.0, 510.0}
    assert levels.gamma_flip is not None
    assert levels.total_gex < 0
    assert "D0" in levels.per_dte_levels
    assert "D1" in levels.per_dte_levels
    assert levels.per_dte_levels["D0"]["call_wall"] == 510.0
    assert levels.per_dte_levels["D1"]["call_wall"] == 511.0


def test_trading_expiration_buckets_skip_weekends() -> None:
    buckets = trading_expiration_buckets(datetime(2026, 6, 12, tzinfo=timezone.utc).date(), (0, 1, 2))
    assert buckets == {
        "2026-06-12": 0,
        "2026-06-15": 1,
        "2026-06-16": 2,
    }


def test_target_expiration_dates_returns_date_objects() -> None:
    start, end = target_expiration_dates(date(2026, 6, 12), (0, 1, 2))

    assert start == date(2026, 6, 12)
    assert end == date(2026, 6, 16)


def test_bullish_magnet_signal_requires_two_holding_candles() -> None:
    _, rows = parse_schwab_option_chain(_chain_fixture(), symbol="SPY", dte_offsets={0})
    _, levels = compute_gamma_levels(rows, symbol="SPY", spot=508.0, magnet_quantile=0.50)
    levels = type(levels)(
        **{
            **levels.to_dict(),
            "nearest_magnet": 505.0,
            "next_magnet_above": 510.0,
            "air_gap_above_score": 5.0,
        }
    )
    engine = DealerSignalEngine(DealerPositioningConfig(hold_candles=2, air_gap_threshold=2.0))
    t0 = datetime(2026, 6, 12, 14, 30, tzinfo=timezone.utc)

    assert engine.on_candle("SPY", PriceCandle(t0, 504, 504.5, 503.5, 504.0), levels) == []
    assert engine.on_candle("SPY", PriceCandle(t0 + timedelta(minutes=1), 504, 506, 504, 506.0), levels) == []
    signals = engine.on_candle("SPY", PriceCandle(t0 + timedelta(minutes=2), 506, 507, 505.5, 506.5), levels)

    assert len(signals) == 1
    assert signals[0].signal_type == "bullish_magnet"
    assert signals[0].target == 510.0
    assert signals[0].stop == 505.0


def test_sim_trade_closes_at_target() -> None:
    _, rows = parse_schwab_option_chain(_chain_fixture(), symbol="SPY", dte_offsets={0})
    _, levels = compute_gamma_levels(rows, symbol="SPY", spot=508.0, magnet_quantile=0.50)
    levels = type(levels)(
        **{
            **levels.to_dict(),
            "nearest_magnet": 505.0,
            "next_magnet_above": 510.0,
            "air_gap_above_score": 5.0,
        }
    )
    config = DealerPositioningConfig(hold_candles=2, air_gap_threshold=2.0)
    engine = DealerSignalEngine(config)
    sim = DealerTradeSimulator(config)
    t0 = datetime(2026, 6, 12, 14, 30, tzinfo=timezone.utc)

    engine.on_candle("SPY", PriceCandle(t0, 504, 505, 503, 504.0), levels)
    engine.on_candle("SPY", PriceCandle(t0 + timedelta(minutes=1), 504, 506, 504, 506.0), levels)
    signal = engine.on_candle("SPY", PriceCandle(t0 + timedelta(minutes=2), 506, 507, 505.5, 506.5), levels)[0]
    trade = sim.on_signal(signal)
    assert trade is not None

    closed = sim.on_candle("SPY", PriceCandle(t0 + timedelta(minutes=3), 506.5, 510.5, 506.0, 510.0))
    assert len(closed) == 1
    assert closed[0].exit_reason == "target"
    assert closed[0].pnl_points == 3.5
