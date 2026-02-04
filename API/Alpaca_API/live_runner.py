from __future__ import annotations

import argparse
import queue as queue_mod
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from alpaca.data.enums import DataFeed

from .bar_aggregator import OhlcvAggregator
from .bar_buffer import BarRingBuffer
from .live_inference import LiveInferenceEngine, LivePPOAgent
from .live_stream import AlpacaBarStreamer


class LiveBarProcessor:
    def __init__(
        self,
        *,
        interval_minutes: int = 15,
        buffer_size: int = 5000,
        agg_label: str = "left",
        on_1m: Optional[Callable[[str, dict, BarRingBuffer], None]] = None,
        on_15m_close: Optional[Callable[[str, dict, BarRingBuffer], None]] = None,
    ) -> None:
        self._interval_minutes = interval_minutes
        self._buffer_size = buffer_size
        self._agg_label = agg_label
        self._on_1m = on_1m
        self._on_15m_close = on_15m_close
        self._buffers: dict[str, BarRingBuffer] = {}
        self._aggregators: dict[str, OhlcvAggregator] = {}

    def _get_buffer(self, symbol: str) -> BarRingBuffer:
        if symbol not in self._buffers:
            self._buffers[symbol] = BarRingBuffer(maxlen=self._buffer_size)
        return self._buffers[symbol]

    def _get_aggregator(self, symbol: str) -> OhlcvAggregator:
        if symbol not in self._aggregators:
            self._aggregators[symbol] = OhlcvAggregator(
                interval_minutes=self._interval_minutes,
                label=self._agg_label,
            )
        return self._aggregators[symbol]

    def handle_bar(self, bar: dict) -> None:
        symbol = str(bar.get("symbol", ""))
        buffer = self._get_buffer(symbol)
        agg = self._get_aggregator(symbol)

        buffer.append(bar)
        if self._on_1m is not None:
            self._on_1m(symbol, bar, buffer)

        closed, _current = agg.update(bar)
        if closed and self._on_15m_close is not None:
            self._on_15m_close(symbol, closed, buffer)


def _parse_feed(feed: str) -> DataFeed:
    feed_key = feed.strip().upper()
    if feed_key == "SIP":
        return DataFeed.SIP
    return DataFeed.IEX


def _print_1m(symbol: str, bar: dict, _buffer: BarRingBuffer) -> None:
    ts = bar.get("timestamp")
    print(f"{symbol} 1m: {ts} o={bar.get('open')} h={bar.get('high')} l={bar.get('low')} c={bar.get('close')} v={bar.get('volume')}")


def _make_15m_handler(
    *,
    inference: LiveInferenceEngine,
    print_15m: bool,
) -> Callable[[str, dict, BarRingBuffer], None]:
    def _handler(symbol: str, bar15: dict, buffer: BarRingBuffer) -> None:
        if print_15m:
            ts = bar15.get("timestamp")
            print(f"{symbol} 15m closed: {ts} o={bar15.get('open')} h={bar15.get('high')} l={bar15.get('low')} c={bar15.get('close')} v={bar15.get('volume')}")
        action = inference.on_15m_close(df_1m=buffer.to_dataframe(), closed_bar=bar15)
        if action is not None:
            print(f"{symbol} inference action: {action}")
    return _handler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream 1m bars from Alpaca and aggregate into 15m candles."
    )
    parser.add_argument("--symbols", default="SPY", help="Comma-separated symbols.")
    parser.add_argument("--feed", default="IEX", help="IEX or SIP.")
    parser.add_argument("--interval", type=int, default=15, help="Aggregation interval in minutes.")
    parser.add_argument("--buffer-size", type=int, default=5000, help="Ring buffer size in 1m bars.")
    parser.add_argument("--queue-size", type=int, default=5000, help="Max queued bars before dropping.")
    parser.add_argument("--print-1m", action="store_true", help="Print each 1m bar.")
    parser.add_argument("--print-15m", action="store_true", help="Print completed 15m bars.")
    parser.add_argument("--resample-label", default="left", help="Resample label (left/right).")
    parser.add_argument("--resample-closed", default="left", help="Resample closed (left/right).")
    parser.add_argument("--tz", default="America/New_York", help="Timezone for resampling (e.g. America/New_York).")
    parser.add_argument("--assume-tz", default="UTC", help="Assume timezone for naive timestamps.")
    parser.add_argument("--model-path", default="Data/outputs/agent/ppo_model.pt", help="PPO model checkpoint.")
    parser.add_argument("--no-agent", action="store_true", help="Disable PPO inference.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions (default is argmax).")
    parser.add_argument("--device", default="auto", help="Device for inference (auto/cpu/cuda/mps).")
    parser.add_argument("--min-15m-bars", type=int, default=20, help="Minimum 15m bars before inference.")
    parser.add_argument("--no-pivot-probs", action="store_true", help="Disable pivot probability features.")
    parser.add_argument("--no-tb-probs", action="store_true", help="Disable triple-barrier probability features.")
    parser.add_argument("--fill-missing-prob", type=float, default=0.0, help="Value for missing prob features.")
    parser.add_argument("--session-open", default="09:30", help="Session open for time features.")
    parser.add_argument("--session-close", default="16:00", help="Session close for time features.")
    parser.add_argument("--ga-model-root", default="Data/models/ga_xgboost/15min", help="GA-XGB model root.")
    parser.add_argument("--ga-feature-list", default=None, help="Path to GA-XGB feature list txt.")
    parser.add_argument("--ga-dataset-name", default="15min", help="Dataset name for GA-XGB feature list fallback.")
    parser.add_argument("--ga-pivot-label-dir", default="pivots", help="Label dir for pivot GA-XGB models.")
    parser.add_argument("--ga-tb-label-dir", default="tb", help="Label dir for TB GA-XGB models.")
    parser.add_argument("--env-file", default=".env", help="Path to .env with Alpaca credentials.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    feed = _parse_feed(args.feed)

    bar_queue: queue_mod.Queue = queue_mod.Queue(maxsize=args.queue_size)
    stop_event = threading.Event()

    agent = None
    if not args.no_agent:
        model_path = args.model_path
        if not model_path:
            raise SystemExit("Missing --model-path for PPO inference.")
        ga_feature_list = args.ga_feature_list
        if ga_feature_list is None:
            try:
                from Data.load_data import get_ticker_processed_base_dir
                from Data.retrieve_data import normalize_ticker

                ticker = normalize_ticker(symbols[0])
                dataset_name = args.ga_dataset_name
                candidate = (
                    get_ticker_processed_base_dir(ticker)
                    / "datasets"
                    / dataset_name
                    / f"features_X_{dataset_name}_tree.txt"
                )
                if candidate.exists():
                    ga_feature_list = str(candidate)
            except Exception:
                ga_feature_list = None

        if ga_feature_list is None and not (args.no_pivot_probs and args.no_tb_probs):
            print("[live] Warning: GA-XGB feature list not found; pivot/TB probs will be filled with defaults.")

        agent = LivePPOAgent(
            model_path=model_path,
            deterministic=not args.stochastic,
            device=args.device,
            include_pivot_probs=not args.no_pivot_probs,
            include_tb_probs=not args.no_tb_probs,
            tz=args.tz or "America/New_York",
            assume_tz=args.assume_tz,
            session_open=args.session_open,
            session_close=args.session_close,
            min_15m_bars=args.min_15m_bars,
            fill_missing_prob=args.fill_missing_prob,
            ga_model_root=args.ga_model_root if ga_feature_list else None,
            ga_feature_list_path=ga_feature_list,
            ga_pivot_label_dir=args.ga_pivot_label_dir,
            ga_tb_label_dir=args.ga_tb_label_dir,
            resample_label=args.resample_label,
            resample_closed=args.resample_closed,
            label_timeframe_rule=f"{args.interval}min",
        )

    inference = LiveInferenceEngine(
        agent=agent,
        label=args.resample_label,
        closed=args.resample_closed,
        rule=f"{args.interval}min",
        tz=args.tz,
        assume_tz=args.assume_tz,
    )

    on_1m = _print_1m if args.print_1m else None
    on_15m = _make_15m_handler(inference=inference, print_15m=args.print_15m)

    processor = LiveBarProcessor(
        interval_minutes=args.interval,
        buffer_size=args.buffer_size,
        agg_label=args.resample_label,
        on_1m=on_1m,
        on_15m_close=on_15m,
    )

    streamer = AlpacaBarStreamer(
        symbols=symbols,
        feed=feed,
        env_file=args.env_file,
        queue=bar_queue,
    )
    streamer.start_in_thread()

    print("Streaming started. Ctrl+C to stop.")
    def _handle_sigint(_signum, _frame) -> None:
        stop_event.set()
        streamer.stop()

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        # Signal registration can fail in some environments; fallback to KeyboardInterrupt.
        pass

    try:
        while not stop_event.is_set():
            try:
                bar = bar_queue.get(timeout=0.5)
            except queue_mod.Empty:
                continue
            processor.handle_bar(bar)
    except KeyboardInterrupt:
        stop_event.set()
        print("Stopping stream...")
    finally:
        streamer.stop()
        streamer.join(timeout=5)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
