"""Alpaca Data API helpers built on the official `alpaca-py` SDK."""

from __future__ import annotations

from .core.config import AlpacaConfig
from .market_data.bar_aggregator import AggregatedBar, OhlcvAggregator
from .market_data.bar_buffer import BarRingBuffer
from .market_data.fetch_intraday import (
    fetch_intraday_spy,
    fetch_latest_quote,
    fetch_latest_quote_spy,
)
from .market_data.live_stream import AlpacaBarStreamer
from .options.options_api import AlpacaOptionsClient, OptionsClientConfig

__all__ = [
    "AggregatedBar",
    "OhlcvAggregator",
    "BarRingBuffer",
    "AlpacaConfig",
    "fetch_intraday_spy",
    "fetch_latest_quote",
    "fetch_latest_quote_spy",
    "AlpacaOptionsClient",
    "OptionsClientConfig",
    "AlpacaBarStreamer",
    "LiveInferenceEngine",
    "LivePPOAgent",
    "LiveGAXGBPredictor",
    "build_15m",
]


def __getattr__(name: str):
    # Keep package import light; only import live inference stack when requested.
    if name in {"LiveGAXGBPredictor", "LiveInferenceEngine", "LivePPOAgent", "build_15m"}:
        from .inference.live_inference import (
            LiveGAXGBPredictor,
            LiveInferenceEngine,
            LivePPOAgent,
            build_15m,
        )

        mapping = {
            "LiveGAXGBPredictor": LiveGAXGBPredictor,
            "LiveInferenceEngine": LiveInferenceEngine,
            "LivePPOAgent": LivePPOAgent,
            "build_15m": build_15m,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
