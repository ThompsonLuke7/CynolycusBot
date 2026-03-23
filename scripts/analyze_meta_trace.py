from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Policy.execution_latch import DirectionExecutionLatch


@dataclass
class Trade:
    side: int
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    bars_held: int

    @property
    def ret(self) -> float:
        if not np.isfinite(self.entry_price) or self.entry_price == 0.0:
            return float("nan")
        if self.side > 0:
            return float(self.exit_price / self.entry_price - 1.0)
        return float(self.entry_price / self.exit_price - 1.0)


@dataclass
class _MetaSweepCache:
    agent: object
    base_frame: pd.DataFrame
    entry_long_probs: np.ndarray
    entry_short_probs: np.ndarray


def _parse_grid(spec: str | None, *, fallback: float) -> list[float]:
    if spec is None or not str(spec).strip():
        return [float(fallback)]
    values: list[float] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        values.append(float(fallback))
    return values


def _default_threshold(df: pd.DataFrame, col: str, fallback: float) -> float:
    if col not in df.columns:
        return float(fallback)
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return float(fallback)
    return float(series.iloc[-1])


def _decide_raw_pos(
    *,
    current_pos: int,
    p_enter_long: float,
    p_enter_short: float,
    p_exit_long: float,
    p_exit_short: float,
    thr_enter_long: float,
    thr_enter_short: float,
    thr_exit_long: float,
    thr_exit_short: float,
) -> int:
    if current_pos > 0:
        return 0 if np.isfinite(p_exit_long) and p_exit_long >= thr_exit_long else 1
    if current_pos < 0:
        return 0 if np.isfinite(p_exit_short) and p_exit_short >= thr_exit_short else -1

    long_ready = np.isfinite(p_enter_long) and p_enter_long >= thr_enter_long
    short_ready = np.isfinite(p_enter_short) and p_enter_short >= thr_enter_short
    if long_ready and short_ready:
        long_margin = p_enter_long - thr_enter_long
        short_margin = p_enter_short - thr_enter_short
        if abs(long_margin - short_margin) <= 1e-9:
            return 0
        return 1 if long_margin > short_margin else -1
    if long_ready:
        return 1
    if short_ready:
        return -1
    return 0


def _simulate_trace(
    df: pd.DataFrame,
    *,
    thr_enter_long: float,
    thr_enter_short: float,
    thr_exit_long: float,
    thr_exit_short: float,
    entry_confirm_bars: int,
    exit_confirm_bars: int,
    initial_position: int = 0,
) -> tuple[list[Trade], pd.Series]:
    latch = DirectionExecutionLatch(
        entry_confirm_bars=entry_confirm_bars,
        exit_confirm_bars=exit_confirm_bars,
        initial_position=initial_position,
    )
    current_pos = int(initial_position)
    entry_ts: pd.Timestamp | None = None
    entry_price = float("nan")
    entry_idx = -1
    trades: list[Trade] = []
    exec_positions: list[int] = []

    for idx, row in enumerate(df.itertuples(index=True)):
        ts = pd.Timestamp(row.Index)
        close = float(getattr(row, "close"))
        if idx == 0 and current_pos != 0:
            entry_ts = ts
            entry_price = float(close)
            entry_idx = idx
        p_enter_long = float(getattr(row, "p_enter_long"))
        p_enter_short = float(getattr(row, "p_enter_short"))
        p_exit_long = float(getattr(row, "p_exit_long")) if np.isfinite(getattr(row, "p_exit_long")) else float("nan")
        p_exit_short = float(getattr(row, "p_exit_short")) if np.isfinite(getattr(row, "p_exit_short")) else float("nan")

        raw_pos = _decide_raw_pos(
            current_pos=current_pos,
            p_enter_long=p_enter_long,
            p_enter_short=p_enter_short,
            p_exit_long=p_exit_long,
            p_exit_short=p_exit_short,
            thr_enter_long=thr_enter_long,
            thr_enter_short=thr_enter_short,
            thr_exit_long=thr_exit_long,
            thr_exit_short=thr_exit_short,
        )
        update = latch.step(raw_pos)
        exec_pos = int(update.executed_pos)

        if exec_pos != current_pos:
            if current_pos != 0 and entry_ts is not None and np.isfinite(entry_price):
                trades.append(
                    Trade(
                        side=int(current_pos),
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        entry_price=float(entry_price),
                        exit_price=float(close),
                        bars_held=max(1, idx - entry_idx),
                    )
                )
            if exec_pos != 0:
                entry_ts = ts
                entry_price = float(close)
                entry_idx = idx
            else:
                entry_ts = None
                entry_price = float("nan")
                entry_idx = -1
            current_pos = exec_pos

        exec_positions.append(exec_pos)

    return trades, pd.Series(exec_positions, index=df.index, name="exec_pos")


def _trades_from_exec_positions(df: pd.DataFrame, exec_pos: pd.Series) -> list[Trade]:
    current_pos = 0
    entry_ts: pd.Timestamp | None = None
    entry_price = float("nan")
    entry_idx = -1
    trades: list[Trade] = []

    aligned_pos = pd.to_numeric(exec_pos, errors="coerce").fillna(0).astype(int).reindex(df.index).fillna(0).astype(int)
    for idx, (ts, row) in enumerate(df.iterrows()):
        close = float(row["close"])
        pos = int(aligned_pos.loc[ts])
        if idx == 0 and pos != 0:
            current_pos = pos
            entry_ts = ts
            entry_price = close
            entry_idx = idx
            continue
        if pos != current_pos:
            if current_pos != 0 and entry_ts is not None and np.isfinite(entry_price):
                trades.append(
                    Trade(
                        side=int(current_pos),
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        entry_price=float(entry_price),
                        exit_price=float(close),
                        bars_held=max(1, idx - entry_idx),
                    )
                )
            if pos != 0:
                entry_ts = ts
                entry_price = close
                entry_idx = idx
            else:
                entry_ts = None
                entry_price = float("nan")
                entry_idx = -1
            current_pos = pos
    return trades


def _summarize_trades(trades: list[Trade]) -> dict[str, float]:
    if not trades:
        return {
            "trades": 0.0,
            "long_trades": 0.0,
            "short_trades": 0.0,
            "win_rate": float("nan"),
            "avg_trade": float("nan"),
            "median_trade": float("nan"),
            "cum_return": 0.0,
            "avg_bars": float("nan"),
        }

    rets = np.asarray([t.ret for t in trades], dtype=float)
    valid = rets[np.isfinite(rets)]
    equity = np.cumprod(1.0 + valid) if valid.size else np.asarray([1.0])
    longs = sum(1 for t in trades if t.side > 0)
    shorts = sum(1 for t in trades if t.side < 0)
    return {
        "trades": float(len(trades)),
        "long_trades": float(longs),
        "short_trades": float(shorts),
        "win_rate": float((valid > 0.0).mean()) if valid.size else float("nan"),
        "avg_trade": float(valid.mean()) if valid.size else float("nan"),
        "median_trade": float(np.median(valid)) if valid.size else float("nan"),
        "cum_return": float(equity[-1] - 1.0) if equity.size else 0.0,
        "avg_bars": float(np.mean([t.bars_held for t in trades])),
    }


def _normalize_ts(ts: str | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _filter_window(df: pd.DataFrame, *, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index <= end]
    return out


def _filter_trades_window(
    trades: list[Trade],
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> list[Trade]:
    out: list[Trade] = []
    for trade in trades:
        if start is not None and trade.exit_ts < start:
            continue
        if end is not None and trade.exit_ts > end:
            continue
        out.append(trade)
    return out


def _default_meta_matrix_path(*, symbol: str, trace_path: Path) -> Path:
    symbol_dir = str(symbol).lower()
    return trace_path.parents[1] / "debug_matrices_warmup" / symbol_dir / "live_meta_matrix_on_trace_ts.parquet"


def _load_meta_matrix(path: Path, *, end: pd.Timestamp | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.loc[ts.notna()].copy()
        df.index = ts[ts.notna()]
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Meta matrix at {path} has no DatetimeIndex or timestamp column.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if end is not None:
        df = df[df.index <= end]
    return df.sort_index()


def _build_meta_sweep_cache(*, meta_matrix_path: Path, model_root: Path, end: pd.Timestamp | None) -> _MetaSweepCache:
    from API.Alpaca_API.inference.live_inference import LiveMetaXGBAgent

    base_frame = _load_meta_matrix(meta_matrix_path, end=end)
    if base_frame.empty:
        raise ValueError(f"Cached meta matrix is empty: {meta_matrix_path}")
    agent = LiveMetaXGBAgent(
        model_root=model_root,
        precomputed_base_frame=base_frame,
    )
    entry_long_probs = agent._entry_long.predict_frame(base_frame)
    entry_short_probs = agent._entry_short.predict_frame(base_frame)
    return _MetaSweepCache(
        agent=agent,
        base_frame=base_frame,
        entry_long_probs=entry_long_probs,
        entry_short_probs=entry_short_probs,
    )


def _trades_from_action_trace(actions: list[dict[str, object]], *, entry_confirm_bars: int, exit_confirm_bars: int) -> list[Trade]:
    if not actions:
        return []
    latch = DirectionExecutionLatch(
        entry_confirm_bars=entry_confirm_bars,
        exit_confirm_bars=exit_confirm_bars,
        initial_position=0,
    )
    current_pos = 0
    entry_ts: pd.Timestamp | None = None
    entry_price = float("nan")
    entry_idx = -1
    trades: list[Trade] = []
    for idx, row in enumerate(actions):
        ts = pd.Timestamp(row["timestamp"])
        close = float(row.get("close", np.nan))
        raw_pos = float(row.get("action", 0.0))
        update = latch.step(raw_pos)
        exec_pos = int(update.executed_pos)
        if exec_pos != current_pos:
            if current_pos != 0 and entry_ts is not None and np.isfinite(entry_price):
                trades.append(
                    Trade(
                        side=int(current_pos),
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        entry_price=float(entry_price),
                        exit_price=float(close),
                        bars_held=max(1, idx - entry_idx),
                    )
                )
            if exec_pos != 0:
                entry_ts = ts
                entry_price = close
                entry_idx = idx
            else:
                entry_ts = None
                entry_price = float("nan")
                entry_idx = -1
            current_pos = exec_pos
    return trades


def _simulate_with_meta_matrix(
    *,
    cache: _MetaSweepCache,
    thr_enter_long: float,
    thr_enter_short: float,
    thr_exit_long: float,
    thr_exit_short: float,
    entry_confirm_bars: int,
    exit_confirm_bars: int,
) -> list[Trade]:
    agent = cache.agent
    base_frame = cache.base_frame
    agent._entry_thresholds["enter_long"] = float(thr_enter_long)
    agent._entry_thresholds["enter_short"] = float(thr_enter_short)
    agent._exit_thresholds["exit_long"] = float(thr_exit_long)
    agent._exit_thresholds["exit_short"] = float(thr_exit_short)
    agent._reset_trade_state()

    latch = DirectionExecutionLatch(
        entry_confirm_bars=entry_confirm_bars,
        exit_confirm_bars=exit_confirm_bars,
        initial_position=0,
    )
    current_pos = 0
    entry_ts: pd.Timestamp | None = None
    entry_price = float("nan")
    entry_idx = -1
    trades: list[Trade] = []

    for idx, (_, row) in enumerate(base_frame.iterrows()):
        p_enter_long = float(cache.entry_long_probs[idx]) if idx < cache.entry_long_probs.size else float("nan")
        p_enter_short = float(cache.entry_short_probs[idx]) if idx < cache.entry_short_probs.size else float("nan")
        row = row.copy()
        row["p_enter_long_oof"] = p_enter_long
        row["p_enter_short_oof"] = p_enter_short
        exit_row = agent._annotate_current_context(row)
        exit_df = pd.DataFrame([exit_row], index=[row.name])
        p_exit_long = agent._exit_long.predict_row(exit_df, target_ts=row.name) if agent._state.position > 0 else float("nan")
        p_exit_short = agent._exit_short.predict_row(exit_df, target_ts=row.name) if agent._state.position < 0 else float("nan")
        action = agent._decide_action(
            p_enter_long=p_enter_long,
            p_enter_short=p_enter_short,
            p_exit_long=p_exit_long,
            p_exit_short=p_exit_short,
        )
        agent._advance_state(action=action, row=row)

        close = float(row.get("close", np.nan))
        update = latch.step(action)
        exec_pos = int(update.executed_pos)
        if exec_pos != current_pos:
            if current_pos != 0 and entry_ts is not None and np.isfinite(entry_price):
                trades.append(
                    Trade(
                        side=int(current_pos),
                        entry_ts=entry_ts,
                        exit_ts=pd.Timestamp(row.name),
                        entry_price=float(entry_price),
                        exit_price=float(close),
                        bars_held=max(1, idx - entry_idx),
                    )
                )
            if exec_pos != 0:
                entry_ts = pd.Timestamp(row.name)
                entry_price = close
                entry_idx = idx
            else:
                entry_ts = None
                entry_price = float("nan")
                entry_idx = -1
            current_pos = exec_pos

    return trades


def _load_trace(path: Path, *, symbol: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if symbol:
        df = df[df["symbol"].astype(str).str.upper() == str(symbol).upper()]
    required = ["close", "p_enter_long", "p_enter_short", "p_exit_long", "p_exit_short"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("timestamp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze meta replay trace performance and threshold sweeps.")
    parser.add_argument("--trace", default="Data/inference/spy/10min/meta/meta_trace_warmup.csv")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default=None, help="Optional UTC start timestamp, e.g. 2026-02-13T00:00:00Z")
    parser.add_argument("--end", default=None, help="Optional UTC end timestamp, e.g. 2026-03-13T23:59:59Z")
    parser.add_argument("--entry-confirm-bars", type=int, default=1)
    parser.add_argument("--exit-confirm-bars", type=int, default=2)
    parser.add_argument("--enter-long-grid", default=None, help="Comma-separated grid, e.g. 0.4,0.5,0.6")
    parser.add_argument("--enter-short-grid", default=None, help="Comma-separated grid, e.g. 0.4,0.5,0.6")
    parser.add_argument("--exit-long-grid", default=None, help="Comma-separated grid, e.g. 0.55,0.65,0.75")
    parser.add_argument("--exit-short-grid", default=None, help="Comma-separated grid, e.g. 0.55,0.65,0.75")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-trades", type=int, default=1, help="Only show threshold combinations with at least this many closed trades in the scored window.")
    parser.add_argument("--model-root", default="Data/models/meta_xgboost/10min", help="Meta model root used for path-aware threshold sweeps.")
    parser.add_argument("--meta-matrix", default=None, help="Optional cached meta matrix parquet. Defaults to the replay dump next to the trace.")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    full_df = _load_trace(trace_path, symbol=args.symbol)
    if full_df.empty:
        raise SystemExit("No trace rows after symbol filtering.")
    start_ts = _normalize_ts(args.start)
    end_ts = _normalize_ts(args.end)
    df = _filter_window(full_df, start=start_ts, end=end_ts)
    if df.empty:
        raise SystemExit("No trace rows after time filtering.")

    base_enter_long = _default_threshold(df, "thr_enter_long", 0.5)
    base_enter_short = _default_threshold(df, "thr_enter_short", 0.5)
    base_exit_long = _default_threshold(df, "thr_exit_long", 0.65)
    base_exit_short = _default_threshold(df, "thr_exit_short", 0.65)

    if "exec_pos" in full_df.columns:
        baseline_trades = _trades_from_exec_positions(full_df, full_df["exec_pos"])
    else:
        baseline_trades, _ = _simulate_trace(
            full_df,
            thr_enter_long=base_enter_long,
            thr_enter_short=base_enter_short,
            thr_exit_long=base_exit_long,
            thr_exit_short=base_exit_short,
            entry_confirm_bars=args.entry_confirm_bars,
            exit_confirm_bars=args.exit_confirm_bars,
        )
    baseline_trades = _filter_trades_window(baseline_trades, start=start_ts, end=end_ts)
    baseline = _summarize_trades(baseline_trades)
    print("Baseline thresholds:")
    print(
        f"  enter_long={base_enter_long:.3f} enter_short={base_enter_short:.3f} "
        f"exit_long={base_exit_long:.3f} exit_short={base_exit_short:.3f}"
    )
    print("Baseline performance:")
    for key in ("trades", "long_trades", "short_trades", "win_rate", "avg_trade", "median_trade", "cum_return", "avg_bars"):
        print(f"  {key}={baseline[key]:.6f}" if np.isfinite(baseline[key]) else f"  {key}=nan")

    enter_long_grid = _parse_grid(args.enter_long_grid, fallback=base_enter_long)
    enter_short_grid = _parse_grid(args.enter_short_grid, fallback=base_enter_short)
    exit_long_grid = _parse_grid(args.exit_long_grid, fallback=base_exit_long)
    exit_short_grid = _parse_grid(args.exit_short_grid, fallback=base_exit_short)

    combos = list(itertools.product(enter_long_grid, enter_short_grid, exit_long_grid, exit_short_grid))
    if len(combos) <= 1:
        return

    meta_matrix_path = Path(args.meta_matrix) if args.meta_matrix else _default_meta_matrix_path(symbol=args.symbol, trace_path=trace_path)
    use_path_aware = meta_matrix_path.exists()
    sweep_cache: _MetaSweepCache | None = None
    if use_path_aware:
        print(f"\nUsing path-aware threshold sweep via cached meta matrix: {meta_matrix_path}")
        cache_started = time.perf_counter()
        sweep_cache = _build_meta_sweep_cache(
            meta_matrix_path=meta_matrix_path,
            model_root=Path(args.model_root),
            end=end_ts,
        )
        print(
            f"Loaded sweep cache: rows={len(sweep_cache.base_frame)} "
            f"setup_time={time.perf_counter() - cache_started:.1f}s"
        )
    else:
        print(
            "\nWarning: cached meta matrix not found; falling back to frozen-probability sweep.\n"
            "This can be misleading because exit probabilities in the trace depend on the realized trade path."
        )
    print(
        f"Sweeping {len(combos)} threshold combinations "
        f"for {args.symbol} from {start_ts if start_ts is not None else 'start'} "
        f"to {end_ts if end_ts is not None else 'end'}..."
    )

    rows: list[dict[str, float]] = []
    started_at = time.perf_counter()
    for combo_idx, (thr_enter_long, thr_enter_short, thr_exit_long, thr_exit_short) in enumerate(combos, start=1):
        combo_started_at = time.perf_counter()
        print(
            f"[{combo_idx}/{len(combos)}] "
            f"enter_long={thr_enter_long:.3f} enter_short={thr_enter_short:.3f} "
            f"exit_long={thr_exit_long:.3f} exit_short={thr_exit_short:.3f}",
            flush=True,
        )
        if use_path_aware:
            trades = _simulate_with_meta_matrix(
                cache=sweep_cache,
                thr_enter_long=thr_enter_long,
                thr_enter_short=thr_enter_short,
                thr_exit_long=thr_exit_long,
                thr_exit_short=thr_exit_short,
                entry_confirm_bars=args.entry_confirm_bars,
                exit_confirm_bars=args.exit_confirm_bars,
            )
        else:
            trades, _ = _simulate_trace(
                full_df,
                thr_enter_long=thr_enter_long,
                thr_enter_short=thr_enter_short,
                thr_exit_long=thr_exit_long,
                thr_exit_short=thr_exit_short,
                entry_confirm_bars=args.entry_confirm_bars,
                exit_confirm_bars=args.exit_confirm_bars,
            )
        trades = _filter_trades_window(trades, start=start_ts, end=end_ts)
        metrics = _summarize_trades(trades)
        elapsed = time.perf_counter() - started_at
        combo_elapsed = time.perf_counter() - combo_started_at
        print(
            f"  -> trades={int(metrics['trades'])} cum_return={metrics['cum_return']:.6f} "
            f"combo_time={combo_elapsed:.1f}s elapsed={elapsed:.1f}s",
            flush=True,
        )
        rows.append(
            {
                "thr_enter_long": thr_enter_long,
                "thr_enter_short": thr_enter_short,
                "thr_exit_long": thr_exit_long,
                "thr_exit_short": thr_exit_short,
                **metrics,
            }
        )

    result = pd.DataFrame(rows)
    result = result[result["trades"] >= float(max(0, int(args.min_trades)))]
    result = result.sort_values(
        ["cum_return", "win_rate", "avg_trade", "trades"],
        ascending=[False, False, False, False],
    )
    print("\nTop threshold combinations:")
    print(result.head(max(1, int(args.top_n))).to_string(index=False))


if __name__ == "__main__":
    main()
