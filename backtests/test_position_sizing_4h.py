import pandas as pd

from backtests.position_sizing_4h import apply_portfolio_cap, size_trades


def test_vol_scaled_uses_stop_risk_and_notional_cap():
    trades = pd.DataFrame({
        "entry_price": [10.0], "pnl_pct": [0.10], "atr_at_entry": [1.0],
        "sl_atr_mult": [2.0], "entry_ts": [pd.Timestamp("2026-01-01", tz="UTC")],
        "exit_ts": [pd.Timestamp("2026-01-02", tz="UTC")], "ticker": ["ABC"],
    })
    out = size_trades(
        trades, scheme="vol_scaled", fixed_shares=100, equal_notional=10_000,
        risk_budget=250, round_trip_cost_bps=20,
    )
    # $250 / ($2 stop distance) = 125 shares, below the $10k notional cap.
    assert out.iloc[0].shares == 125
    assert out.iloc[0].entry_notional == 1250
    assert out.iloc[0].net_pnl == 122.5


def test_portfolio_cap_releases_exits_and_blocks_overlapping_ticker():
    ts = pd.Timestamp
    trades = pd.DataFrame({
        "entry_ts": [ts("2026-01-01", tz="UTC"), ts("2026-01-01", tz="UTC"), ts("2026-01-02", tz="UTC")],
        "exit_ts": [ts("2026-01-02", tz="UTC"), ts("2026-01-03", tz="UTC"), ts("2026-01-04", tz="UTC")],
        "strategy": ["htf", "htf", "htf"], "ticker": ["A", "A", "B"],
        "score": [0.9, 0.8, 0.7], "entry_notional": [60.0, 40.0, 60.0],
    })
    out = apply_portfolio_cap(trades, max_gross_notional=100.0)
    assert out["portfolio_reason"].tolist() == ["accepted", "ticker_already_open", "accepted"]
