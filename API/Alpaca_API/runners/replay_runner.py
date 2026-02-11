import argparse
import time
from pathlib import Path

import pandas as pd

from ..inference.live_inference import LiveInferenceEngine, LivePPOAgent
from .live_runner import LiveBarProcessor, _format_ts_local, _load_test_split_warmup_1m
from Policy.order_policy import OptionOrderPolicy, OptionOrderPolicyConfig


def _load_history(path: Path, *, assume_tz: str = "UTC") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("History file must be .csv or .parquet")

    rename_map = {
        "Date": "timestamp",
        "date": "timestamp",
        "Datetime": "timestamp",
        "datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        else:
            raise ValueError("History data must include a timestamp column or DatetimeIndex.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(assume_tz)
    df = df.dropna(subset=["timestamp"])
    return df


def _apply_regular_hours(df: pd.DataFrame, *, tz: str = "America/New_York") -> pd.DataFrame:
    ts = df["timestamp"].dt.tz_convert(tz)
    minutes = ts.dt.hour * 60 + ts.dt.minute
    regular_mask = minutes.between(570, 960)
    return df.loc[regular_mask].copy()


def _make_15m_handler(
    *,
    inference: LiveInferenceEngine,
    print_15m: bool,
    print_tz: str,
    order_policies: dict[str, OptionOrderPolicy] | None = None,
):
    def _handler(symbol: str, bar15: dict, buffer) -> None:
        if print_15m:
            ts = _format_ts_local(bar15.get("timestamp"), tz=print_tz)
            print(
                f"{symbol} 15m closed: {ts} "
                f"o={bar15.get('open')} h={bar15.get('high')} "
                f"l={bar15.get('low')} c={bar15.get('close')} v={bar15.get('volume')}"
            )
        if order_policies is not None and symbol in order_policies:
            order_policies[symbol].on_15m_bar(closed_bar=bar15)
        action = inference.on_15m_close(df_1m=buffer.to_dataframe(), closed_bar=bar15)
        if action is not None:
            print(f"{symbol} inference action: {float(action):+.4f}")
            if order_policies is not None and symbol in order_policies:
                result = order_policies[symbol].on_decision(
                    action=float(action),
                    closed_bar=bar15,
                    update_bar_state=False,
                )
                event = str(result.get("event", "unknown"))
                if event not in {"hold", "no_change"}:
                    print(f"{symbol} order_policy event={event} details={result}")

    return _handler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay historical 1m bars through the live inference pipeline."
    )
    parser.add_argument(
        "--data-path",
        default="Data/raw/spy/inference_buffer_1m.parquet",
        help="CSV/Parquet with 1m bars.",
    )
    parser.add_argument("--symbols", default="SPY", help="Comma-separated symbols.")
    parser.add_argument("--start", default=None, help="Optional ISO start timestamp (UTC).")
    parser.add_argument("--end", default=None, help="Optional ISO end timestamp (UTC).")
    parser.add_argument("--regular-only", action="store_true", help="Filter to 9:30-16:00 ET.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between bars.")
    parser.add_argument("--max-bars", type=int, default=None, help="Max bars to replay.")
    parser.add_argument("--buffer-size", type=int, default=5000, help="Ring buffer size.")
    parser.add_argument("--print-1m", action="store_true", help="Print each 1m bar.")
    parser.add_argument("--print-15m", action="store_true", help="Print completed 15m bars.")
    parser.add_argument("--resample-label", default="left", help="Resample label (left/right).")
    parser.add_argument("--resample-closed", default="left", help="Resample closed (left/right).")
    parser.add_argument("--tz", default="America/New_York", help="Timezone for resampling.")
    parser.add_argument("--assume-tz", default="UTC", help="Assume timezone for naive timestamps.")
    parser.add_argument("--model-path", default="Data/outputs/agent/ppo_model.pt", help="PPO model checkpoint.")
    parser.add_argument("--no-agent", action="store_true", help="Disable PPO inference.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions from policy (default is deterministic mean).")
    parser.add_argument("--device", default="auto", help="Device for inference (auto/cpu/cuda/mps).")
    parser.add_argument("--min-15m-bars", type=int, default=20, help="Minimum 15m bars before inference.")
    parser.add_argument("--no-pivot-probs", action="store_true", help="Disable pivot probability features.")
    parser.add_argument("--no-tb-probs", action="store_true", help="Disable triple-barrier probability features.")
    parser.add_argument("--fill-missing-prob", type=float, default=0.0, help="Value for missing prob features.")
    parser.add_argument("--session-open", default="09:30", help="Session open for time features.")
    parser.add_argument("--session-close", default="16:00", help="Session close for time features.")
    parser.add_argument("--ga-model-root", default="Data/models/ga_xgboost/15min", help="GA-XGB model root.")
    parser.add_argument("--ga-feature-list", default=None, help="Path to GA-XGB feature list txt.")
    parser.add_argument("--ga-dataset-name", default="15min", help="Dataset name for split-warmup lookup.")
    parser.add_argument(
        "--split-x-filename",
        default="X_15min_tree.parquet",
        help="Feature filename stem used to locate split indices for test warmup preload.",
    )
    parser.add_argument("--ga-pivot-label-dir", default="pivots", help="Label dir for pivot GA-XGB models.")
    parser.add_argument("--ga-tb-label-dir", default="tb", help="Label dir for TB GA-XGB models.")
    parser.add_argument("--env-file", default=".env", help="Path to .env with Alpaca credentials.")
    parser.add_argument(
        "--enable-option-orders",
        action="store_true",
        help="Enable option order policy execution on each 15m inference action.",
    )
    parser.add_argument(
        "--option-order-qty",
        type=int,
        default=1,
        help="Fallback max contracts when account/quote sizing is unavailable.",
    )
    parser.add_argument(
        "--option-price-mode",
        default="ask",
        choices=["ask", "mid", "bid", "last", "mark"],
        help="Price input for sizing max contracts (ask is conservative, mid for sim).",
    )
    parser.add_argument(
        "--option-action-ema-alpha",
        type=float,
        default=0.85,
        help="EMA alpha for action smoothing (higher = smoother).",
    )
    parser.add_argument(
        "--option-rebalance-deadband",
        type=float,
        default=0.10,
        help="Ignore action changes smaller than this after smoothing.",
    )
    parser.add_argument(
        "--option-max-step-contracts",
        type=int,
        default=2,
        help="Max absolute signed-contract change per decision step.",
    )
    parser.add_argument(
        "--option-max-contracts-cap",
        type=int,
        default=0,
        help="Optional hard cap on max contracts (<=0 disables cap).",
    )
    parser.add_argument(
        "--option-atr-mult",
        type=float,
        default=1.0,
        help="ATR multiplier for target strike distance (default 1.0 ATR).",
    )
    parser.add_argument(
        "--option-dte-cutoff",
        default="14:00",
        help="Local HH:MM cutoff; before cutoff use 0DTE, otherwise 1DTE.",
    )
    parser.add_argument(
        "--simulate-orders",
        action="store_true",
        help="Do not submit to Alpaca; print intended order payloads only.",
    )
    parser.add_argument(
        "--option-no-close-on-flat",
        action="store_true",
        help="Do not auto close open option when agent action goes flat.",
    )
    parser.add_argument(
        "--option-no-close-on-flip",
        action="store_true",
        help="Do not auto close existing option before flipping side.",
    )
    parser.add_argument(
        "--no-prepend-split-test-warmup",
        action="store_true",
        help="Disable prepending test-split 1m warmup bars before replay data.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    df = _load_history(Path(args.data_path), assume_tz=args.assume_tz)
    if args.regular_only:
        df = _apply_regular_hours(df, tz=args.tz)

    if "symbol" not in df.columns:
        if len(symbols) != 1:
            raise ValueError("History data missing symbol column; use single --symbols or add symbol column.")
        df["symbol"] = symbols[0]
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df[df["symbol"].isin(symbols)]

    if args.start:
        start = pd.to_datetime(args.start, utc=True, errors="coerce")
        df = df[df["timestamp"] >= start]
    if args.end:
        end = pd.to_datetime(args.end, utc=True, errors="coerce")
        df = df[df["timestamp"] <= end]

    if not args.no_prepend_split_test_warmup:
        warm_frames = []
        for symbol in symbols:
            warm_df = _load_test_split_warmup_1m(
                symbol=symbol,
                dataset_name=args.ga_dataset_name,
                x_filename=args.split_x_filename,
            )
            if warm_df is None or warm_df.empty:
                continue
            if args.regular_only:
                warm_df = _apply_regular_hours(warm_df, tz=args.tz)
            warm_frames.append(warm_df)
        if warm_frames:
            warmup = pd.concat(warm_frames, axis=0, ignore_index=True)
            df = pd.concat([warmup, df], axis=0, ignore_index=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp")
            # Prefer bars from explicit replay file when timestamps overlap.
            df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
            print(
                f"[replay] Prepended split-test warmup bars: {len(warmup):,} "
                f"(combined rows: {len(df):,})"
            )
        else:
            print("[replay] Split-test warmup not found; replaying provided data only.")

    required = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"History data missing required columns: {missing}")

    df = df.sort_values("timestamp")

    if args.max_bars is not None:
        keep_n = int(args.max_bars)
        if keep_n > 0:
            df = df.tail(keep_n).copy()

    agent = None
    if not args.no_agent:
        agent = LivePPOAgent(
            model_path=args.model_path,
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
            ga_model_root=args.ga_model_root if args.ga_feature_list else None,
            ga_feature_list_path=args.ga_feature_list,
            ga_pivot_label_dir=args.ga_pivot_label_dir,
            ga_tb_label_dir=args.ga_tb_label_dir,
            resample_label=args.resample_label,
            resample_closed=args.resample_closed,
            label_timeframe_rule="15min",
        )

    inference = LiveInferenceEngine(
        agent=agent,
        label=args.resample_label,
        closed=args.resample_closed,
        rule="15min",
        tz=args.tz,
        assume_tz=args.assume_tz,
    )

    order_policies: dict[str, OptionOrderPolicy] | None = None
    if args.enable_option_orders:
        order_policies = {}
        for symbol in symbols:
            cfg = OptionOrderPolicyConfig(
                underlying=symbol,
                env_file=args.env_file,
                tz_name=args.tz or "America/New_York",
                atr_multiplier=float(args.option_atr_mult),
                dte_cutoff_hhmm=args.option_dte_cutoff,
                qty=int(args.option_order_qty),
                close_on_flat=not args.option_no_close_on_flat,
                close_on_flip=not args.option_no_close_on_flip,
                submit_orders=not args.simulate_orders,
                ema_alpha=float(args.option_action_ema_alpha),
                rebalance_deadband=float(args.option_rebalance_deadband),
                max_step_contracts=int(args.option_max_step_contracts),
                price_mode=str(args.option_price_mode),
                max_contracts_fallback=int(args.option_order_qty),
                max_contracts_cap=int(args.option_max_contracts_cap),
            )
            order_policies[symbol] = OptionOrderPolicy(cfg)
        mode = "SIMULATED" if args.simulate_orders else "LIVE"
        print(f"[replay] Option order policy enabled ({mode}) for symbols: {', '.join(symbols)}")

    processor = LiveBarProcessor(
        interval_minutes=15,
        buffer_size=args.buffer_size,
        agg_label=args.resample_label,
        on_1m=(lambda symbol, bar, _buf: print(
            f"{symbol} 1m: {_format_ts_local(bar.get('timestamp'), tz=args.tz or 'America/New_York')} "
            f"o={bar.get('open')} h={bar.get('high')} l={bar.get('low')} "
            f"c={bar.get('close')} v={bar.get('volume')}"
        )) if args.print_1m else None,
        on_15m_close=_make_15m_handler(
            inference=inference,
            print_15m=args.print_15m,
            print_tz=args.tz or "America/New_York",
            order_policies=order_policies,
        ),
    )

    count = 0
    for row in df.itertuples(index=False):
        bar = {
            "symbol": getattr(row, "symbol"),
            "timestamp": getattr(row, "timestamp"),
            "open": float(getattr(row, "open")),
            "high": float(getattr(row, "high")),
            "low": float(getattr(row, "low")),
            "close": float(getattr(row, "close")),
            "volume": float(getattr(row, "volume")),
        }
        processor.handle_bar(bar)
        count += 1
        if args.max_bars and count >= args.max_bars:
            break
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"[replay] Done. Bars processed: {count:,}.")


if __name__ == "__main__":
    main()
