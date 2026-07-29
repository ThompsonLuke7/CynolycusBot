import pandas as pd
import pytest

from strategies.spy_intraday.Policy.option_mark_capture import (
    _alpaca_quote,
    _quote_fields,
    schwab_option_symbol,
    OptionMarkCapture,
)
from strategies.spy_intraday.Policy.order_policy import OptionOrderPolicy, OptionOrderPolicyConfig


def test_schwab_option_symbol_uses_six_character_root():
    assert schwab_option_symbol("SPY260728C00738000") == "SPY   260728C00738000"


def test_alpaca_quote_and_normalized_bid_ask_fields():
    quote = _alpaca_quote({"quotes": {"SPY260728C00738000": {"bp": 1.2, "ap": 1.4}}}, "SPY260728C00738000")
    assert quote is not None
    fields = _quote_fields(quote)
    assert fields["bid"] == 1.2 and fields["ask"] == 1.4
    assert fields["mid"] == pytest.approx(1.3) and fields["spread"] == pytest.approx(.2)
    assert fields["mark"] is None and fields["last"] is None


def test_setup_failure_grace_defers_an_otherwise_invalidated_setup():
    policy = OptionOrderPolicy(OptionOrderPolicyConfig(meta_setup_failure_grace_minutes=3))
    entry = policy._trail_state("long")
    entry.entry_ts = pd.Timestamp("2026-07-28T13:46:00Z").to_pydatetime()
    entry.entry_atr = 2.0
    policy._meta_entry_structure["long"] = {"signal_low": 100.0, "signal_atr": 2.0}
    assert not policy._setup_failure_exit_hit(
        side="long", close=99.0, high=100.0, low=99.0,
        local_ts=pd.Timestamp("2026-07-28T13:48:00Z").to_pydatetime(),
    )
    assert policy._setup_failure_exit_hit(
        side="long", close=99.0, high=100.0, low=99.0,
        local_ts=pd.Timestamp("2026-07-28T13:49:00Z").to_pydatetime(),
    )


def test_capture_writes_each_source_for_active_contract(tmp_path):
    class Alpaca:
        def get_option_quotes(self, **_kwargs):
            return {"quotes": {"SPY260728C00738000": {"bp": 1.0, "ap": 1.2}}}

    class Schwab:
        def get_quotes(self, symbols):
            return {symbols[0]: {"quote": {"bidPrice": 1.01, "askPrice": 1.21}}}

    capture = OptionMarkCapture(output_dir=tmp_path, alpaca_factory=Alpaca, schwab_factory=Schwab)
    assert capture.capture_active(
        underlying="SPY", bar={"timestamp": "2026-07-28T13:46:00Z", "close": 737.0},
        policy_state={"position": 1, "open_symbol": "SPY260728C00738000"}, phase="pre_1m_policy",
    ) == 2
    rows = [line for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert len(rows) == 2
