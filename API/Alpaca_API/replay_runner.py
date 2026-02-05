import argparse
import time
from pathlib import Path

import pandas as pd

from .live_inference import LiveInferenceEngine, LivePPOAgent
from .live_runner import LiveBarProcessor


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


def _make_15m_handler(*, inference: LiveInferenceEngine, print_15m: bool):
    def _handler(symbol: str, bar15: dict, buffer) -> None:
        if print_15m:
            ts = bar15.get("timestamp")
            print(
                f"{symbol} 15m closed: {ts} "
                f"o={bar15.get('open')} h={bar15.get('high')} "
                f"l={bar15.get('low')} c={bar15.get('close')} v={bar15.get('volume')}"
            )
        action = inference.on_15m_close(df_1m=buffer.to_dataframe(), closed_bar=bar15)
        if action is not None:
            print(f"{symbol} inference action: {action}")

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
    parser.add_argument("--ga-pivot-label-dir", default="pivots", help="Label dir for pivot GA-XGB models.")
    parser.add_argument("--ga-tb-label-dir", default="tb", help="Label dir for TB GA-XGB models.")
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

    required = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"History data missing required columns: {missing}")

    df = df.sort_values("timestamp")

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

    processor = LiveBarProcessor(
        interval_minutes=15,
        buffer_size=args.buffer_size,
        agg_label=args.resample_label,
        on_1m=(lambda symbol, bar, _buf: print(
            f"{symbol} 1m: {bar.get('timestamp')} o={bar.get('open')} "
            f"h={bar.get('high')} l={bar.get('low')} c={bar.get('close')} v={bar.get('volume')}"
        )) if args.print_1m else None,
        on_15m_close=_make_15m_handler(inference=inference, print_15m=args.print_15m),
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
