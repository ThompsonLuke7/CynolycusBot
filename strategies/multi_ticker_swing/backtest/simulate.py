"""
Event-driven backtest simulator for the multi-ticker swing pipeline.

Signal generation (30-minute):
  - Model probability > threshold for class 2 (long) or class 0 (short)
  - Optional: swing state gate must be active

Execution (10-minute):
  - Entry at the open of the first 10m bar after the 30m signal
  - Up to entry_bars_max bars to find a qualifying entry
  - TP / SL tracked bar-by-bar on 10m OHLCV

Usage:
  python multi_ticker_swing/backtest/simulate.py [--test-start DATE] [--test-end DATE]
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier, XGBRanker
except ImportError:
    XGBClassifier = None
    XGBRanker = None


def load_model_any(model_path: Path, model_kind: str = "classifier"):
    """Load a model saved as .joblib (sklearn xgb/lgbm), .json (xgb native), or .native.

    Returns an object exposing predict_proba (classifier) or predict (ranker).
    """
    model_path = Path(model_path)
    suffix = model_path.suffix.lower()
    if suffix in {".joblib", ".pkl"}:
        import joblib
        return joblib.load(model_path)
    if XGBClassifier is None:
        raise ImportError("xgboost required to load native model")
    model = XGBRanker() if model_kind == "ranker" else XGBClassifier()
    model.load_model(str(model_path))
    return model

from strategies.multi_ticker_swing.config.pipeline_config import (
    BACKTEST_CONFIG,
    BACKTEST_RESULTS_DIR,
    FEATURE_COLUMNS,
    MODEL_PATH,
    RAW_10M_DIR,
    TRADING_BLACKLIST,
    TRAINING_MATRIX,
)
from strategies.multi_ticker_swing.data.load_data import load_raw_5m, load_raw_30m, load_training_matrix


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker:       str
    direction:    int          # +1 long, -1 short
    entry_time:   pd.Timestamp
    entry_price:  float
    tp_price:     float
    sl_price:     float
    atr_at_entry: float
    size:         float        # dollar notional


@dataclass
class Trade:
    ticker:       str
    direction:    int
    entry_time:   pd.Timestamp
    exit_time:    pd.Timestamp
    entry_price:  float
    exit_price:   float
    pnl_pct:      float
    pnl_dollar:   float
    exit_reason:  str          # "tp", "sl", "time"
    size:         float
    score:        float = float("nan")   # model score at signal (for magnitude/rank analysis)


# ---------------------------------------------------------------------------
# 10m data loader (per ticker, test window only)
# ---------------------------------------------------------------------------

def _load_5m_slice(
    ticker: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame | None:
    try:
        df = load_raw_5m(ticker)
        df.columns = [c.lower() for c in df.columns]
        mask = (df.index >= test_start) & (df.index <= test_end)
        return df.loc[mask].copy()
    except FileNotFoundError:
        return None


def _load_30m_slice(
    ticker: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame | None:
    try:
        df = load_raw_30m(ticker)
        df.columns = [c.lower() for c in df.columns]
        mask = (df.index >= test_start) & (df.index <= test_end)
        return df.loc[mask].copy()
    except FileNotFoundError:
        logger.warning("[%s] 30m data missing — no execution data", ticker)
        return None


# ---------------------------------------------------------------------------
# Signal generation from model
# ---------------------------------------------------------------------------

def generate_signals(
    matrix: pd.DataFrame,
    model,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    feature_columns: list[str],
    cfg: dict,
    *,
    model_kind: str = "classifier",
    selection: str = "threshold",
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Run model on test-period rows and attach a `score` + `signal` column.

    model_kind: "classifier" (predict_proba -> P(long)) or "ranker" (predict -> score).
    selection:
      - "threshold": classifier only; long if P(long)>th_long, short if P(short)>th_short.
      - "topk":      per timestamp, take the top_k tickers by score as long signals (no shorts).
                     This mirrors the competition's NDCG@K evaluation and is the apples-to-apples
                     mode for comparing a ranker against a classifier's P(long) as a score.

    signal:  +1 = long signal,  -1 = short signal,  0 = no signal
    """
    ts_col = _find_ts_col(matrix)
    mask   = (pd.to_datetime(matrix[ts_col]) >= test_start) & \
             (pd.to_datetime(matrix[ts_col]) <= test_end)
    df = matrix.loc[mask].copy()

    # Reindex to the exact trained feature set/order. The models (sklearn xgb/lgbm) take
    # positional numpy input, so selecting only the *present* columns (df[avail]) silently
    # corrupts predictions whenever any column is missing or out of order — the test-set bug.
    # reindex guarantees count + order; genuinely-missing columns become NaN (handled natively).
    if model is None:
        # Precomputed-scores path: the matrix already carries a "score" column (e.g. leak-free
        # walk-forward OOF preds). Used to backtest OOF scores on the test window without the
        # leakage of predicting with a model that was trained on that window.
        if "score" not in df.columns:
            raise ValueError("model=None requires a precomputed 'score' column in the matrix")
    else:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            logger.warning("generate_signals: %d/%d feature columns missing, filling NaN: %s",
                           len(missing), len(feature_columns), missing[:10])
        X = df.reindex(columns=feature_columns).to_numpy(dtype=np.float32)

        if model_kind == "ranker":
            df["score"] = np.asarray(model.predict(X), dtype=float)
        else:
            proba = model.predict_proba(X)
            classes = list(getattr(model, "classes_", [0, 1, 2]))
            long_idx  = classes.index(2) if 2 in classes else proba.shape[1] - 1
            short_idx = classes.index(0) if 0 in classes else 0
            df["prob_short"]   = proba[:, short_idx]
            df["prob_neutral"] = proba[:, classes.index(1)] if 1 in classes else np.nan
            df["prob_long"]    = proba[:, long_idx]
            df["score"]        = df["prob_long"]

    if selection == "precomputed":
        # scores parquet already carries a per-row `signal` (+1/-1/0), e.g. from the live
        # directional logic (p_long_dir vs p_short_dir). Pass it straight through.
        if "signal" not in df.columns:
            raise ValueError("selection='precomputed' requires a 'signal' column in the matrix")
        n_long  = int((df["signal"] == 1).sum())
        n_short = int((df["signal"] == -1).sum())
        logger.info("Signals[precomputed]: long=%d short=%d (%d rows)", n_long, n_short, len(df))
        return df

    df["signal"] = 0

    if selection in ("topk", "bottomk", "both", "topk_short"):
        hi = df.groupby(ts_col)["score"].rank(ascending=False, method="first")
        lo = df.groupby(ts_col)["score"].rank(ascending=True, method="first")
        if selection in ("topk", "both"):
            df.loc[hi <= top_k, "signal"] = 1      # top-K by score -> long
        if selection == "topk_short":
            df.loc[hi <= top_k, "signal"] = -1     # top-K of a SHORT ranker -> short
        if selection in ("bottomk", "both"):
            df.loc[lo <= top_k, "signal"] = -1     # bottom-K by score -> short
        n_long  = int((df["signal"] == 1).sum())
        n_short = int((df["signal"] == -1).sum())
    else:
        if model_kind == "ranker":
            raise ValueError("selection='threshold' requires a classifier (no calibrated probs from a ranker)")
        th_long  = cfg["prob_threshold_long"]
        th_short = cfg["prob_threshold_short"]
        long_signal  = df["prob_long"]  > th_long
        short_signal = df["prob_short"] > th_short
        if cfg.get("swing_gate_required", True) and "p_long_state_gate" in df.columns:
            long_signal  = long_signal  & (df["p_long_state_gate"]  == 1)
            short_signal = short_signal & (df["p_short_state_gate"] == 1)
        df.loc[long_signal,  "signal"] = 1
        df.loc[short_signal, "signal"] = -1
        n_long, n_short = int(long_signal.sum()), int(short_signal.sum())

    logger.info(
        "Signals[%s/%s]: long=%d  short=%d  (test window %s→%s, %d rows)",
        model_kind, selection, n_long, n_short,
        test_start.date(), test_end.date(), len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Per-ticker simulation
# ---------------------------------------------------------------------------

def _simulate_ticker(
    ticker: str,
    signals: pd.DataFrame,           # rows for this ticker, has ts_col, signal, prob_long, prob_short
    df_5m: Optional[pd.DataFrame],   # 5m bars for execution; None → fall back to df_30m
    df_30m: Optional[pd.DataFrame],  # raw 30m OHLCV bars; used for ATR/price and as exec fallback
    cfg: dict,
    equity_curve: list[float],
    equity_times: list[pd.Timestamp],
    initial_equity: float,
    ts_col: str,
) -> list[Trade]:
    """Simulate all trades for one ticker during the test period."""
    trades: list[Trade] = []
    if signals.empty:
        return trades

    # Execution bars: prefer 5m, fall back to raw 30m bars
    if df_5m is not None and not df_5m.empty:
        exec_df = df_5m
        max_holding_bars = 144 * 3   # 432 × 5m ≈ 3 days
    elif df_30m is not None and not df_30m.empty:
        exec_df = df_30m
        max_holding_bars = 24 * 3    # 72 × 30m ≈ 3 days
    else:
        return trades  # no execution data at all

    # Per-side TP/SL (order policy v2). Falls back to the scalar keys when side-specific
    # keys are absent (v1 behaviour).
    tp_long  = cfg.get("tp_atr_mult_long",  cfg["tp_atr_mult"])
    sl_long  = cfg.get("sl_atr_mult_long",  cfg["sl_atr_mult"])
    tp_short = cfg.get("tp_atr_mult_short", cfg["tp_atr_mult"])
    sl_short = cfg.get("sl_atr_mult_short", cfg["sl_atr_mult"])
    entry_bars_max  = cfg["entry_bars_max"]
    pos_size_pct    = cfg["position_size_pct"]
    commission_pct  = cfg["commission_pct"]

    # Build a timestamp → close lookup from raw 30m bars for ATR/price at signal time
    raw30_close: dict[pd.Timestamp, float] = {}
    raw30_atr14: dict[pd.Timestamp, float] = {}
    if df_30m is not None and not df_30m.empty:
        raw30_close = df_30m["close"].to_dict()
        if "atr_14" in df_30m.columns:
            raw30_atr14 = df_30m["atr_14"].to_dict()

    active: Optional[Position] = None

    exec_times = exec_df.index.tolist()
    exec_lookup = {ts: i for i, ts in enumerate(exec_times)}

    signal_rows = signals.sort_values(ts_col)

    for _, sig_row in signal_rows.iterrows():
        sig_ts  = pd.Timestamp(sig_row[ts_col])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.tz_localize("UTC")
        signal  = int(sig_row["signal"])

        # Skip if no signal or already in a position
        if signal == 0 or active is not None:
            continue

        # ATR in absolute price terms at signal time
        # Prefer raw 30m atr_14; fall back to atr_pct_14 × close
        close_at_signal = raw30_close.get(sig_ts, np.nan)
        atr_val = raw30_atr14.get(sig_ts, np.nan)
        if np.isnan(atr_val) and not np.isnan(close_at_signal):
            atr_pct = float(sig_row.get("atr_pct_14", np.nan))
            if not np.isnan(atr_pct) and atr_pct > 0:
                atr_val = atr_pct * close_at_signal

        if np.isnan(atr_val) or atr_val <= 0:
            continue
        if np.isnan(close_at_signal):
            continue

        # -----------------------------------------------------------------
        # Find entry on next 10m bars after signal
        # -----------------------------------------------------------------
        # Locate first exec bar strictly after sig_ts
        entry_idx = None
        for ts in exec_times:
            if ts > sig_ts:
                entry_idx = exec_lookup[ts]
                break

        if entry_idx is None:
            continue

        # Try up to entry_bars_max bars to enter
        entry_ts    = None
        entry_price = None
        for k in range(entry_bars_max):
            idx = entry_idx + k
            if idx >= len(exec_times):
                break
            bar = exec_df.iloc[idx]
            ep  = float(bar.get("open", bar.get("close", np.nan)))
            if not np.isnan(ep):
                entry_ts    = exec_times[idx]
                entry_price = ep
                break

        if entry_ts is None or entry_price is None:
            continue

        # Reject stale fills: if no execution bar exists within max_entry_delay_days of the
        # signal, the ticker's exec data does not cover the signal time. Without this guard,
        # every signal before a ticker's first available bar is dumped onto that first bar as
        # a duplicate entry (e.g. tickers whose 5m history starts mid-test-window).
        if (entry_ts - sig_ts) > pd.Timedelta(days=cfg.get("max_entry_delay_days", 4)):
            continue

        # Equity at this point for position sizing
        current_equity = equity_curve[-1] if equity_curve else initial_equity
        notional = current_equity * pos_size_pct

        if signal == 1:    # long
            tp_price = entry_price + tp_long * atr_val
            sl_price = entry_price - sl_long * atr_val
        else:              # short
            tp_price = entry_price - tp_short * atr_val
            sl_price = entry_price + sl_short * atr_val

        active = Position(
            ticker=ticker,
            direction=signal,
            entry_time=entry_ts,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            atr_at_entry=atr_val,
            size=notional,
        )

        # -----------------------------------------------------------------
        # Track position until TP / SL / time expiry
        # -----------------------------------------------------------------
        exit_ts     = None
        exit_price  = None
        exit_reason = "time"

        for j in range(entry_idx + 1, min(entry_idx + max_holding_bars + 1, len(exec_times))):
            bar = exec_df.iloc[j]
            bar_h = float(bar.get("high",  close_at_signal))
            bar_l = float(bar.get("low",   close_at_signal))
            bar_c = float(bar.get("close", close_at_signal))

            if signal == 1:   # long
                if bar_l <= active.sl_price:
                    exit_ts, exit_price, exit_reason = exec_times[j], active.sl_price, "sl"
                    break
                if bar_h >= active.tp_price:
                    exit_ts, exit_price, exit_reason = exec_times[j], active.tp_price, "tp"
                    break
            else:              # short
                if bar_h >= active.sl_price:
                    exit_ts, exit_price, exit_reason = exec_times[j], active.sl_price, "sl"
                    break
                if bar_l <= active.tp_price:
                    exit_ts, exit_price, exit_reason = exec_times[j], active.tp_price, "tp"
                    break
        else:
            # Time expiry: exit at last bar's close
            last_idx = min(entry_idx + max_holding_bars, len(exec_times) - 1)
            exit_ts    = exec_times[last_idx]
            exit_price = float(exec_df.iloc[last_idx].get("close", entry_price))
            exit_reason = "time"

        if exit_ts is None:
            active = None
            continue

        # PnL
        raw_ret = (exit_price - entry_price) / entry_price * signal
        net_ret = raw_ret - commission_pct
        pnl_dollar = notional * net_ret

        trades.append(Trade(
            ticker=ticker,
            direction=signal,
            entry_time=active.entry_time,
            exit_time=exit_ts,
            entry_price=active.entry_price,
            exit_price=exit_price,
            pnl_pct=net_ret,
            pnl_dollar=pnl_dollar,
            exit_reason=exit_reason,
            size=notional,
            score=float(sig_row.get("score", float("nan"))),
        ))
        active = None

    return trades


# ---------------------------------------------------------------------------
# Portfolio-level simulation
# ---------------------------------------------------------------------------

def simulate(
    *,
    matrix_path: Path = TRAINING_MATRIX,
    model_path: Path = MODEL_PATH,
    results_dir: Path = BACKTEST_RESULTS_DIR,
    feature_columns: list[str] = FEATURE_COLUMNS,
    cfg: dict = BACKTEST_CONFIG,
    test_start_str: str = "2024-01-01",
    test_end_str: str   = "2026-04-15",
    force: bool = False,
    model_kind: str = "classifier",
    selection: str = "threshold",
    top_k: int = 10,
    apply_blacklist: bool = False,
    scores_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full backtest. Returns (trades_df, equity_curve_df).

    scores_path: optional parquet of precomputed leak-free scores (timestamp, ticker, score),
    e.g. walk-forward OOF preds. When given, no model is loaded — selection runs on those scores.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    trades_path = results_dir / "trades.parquet"
    equity_path = results_dir / "equity_curve.parquet"

    if trades_path.exists() and equity_path.exists() and not force:
        logger.info("Backtest results exist — loading from cache.")
        return pd.read_parquet(trades_path), pd.read_parquet(equity_path)

    if scores_path is not None:
        clf = None
        matrix = pd.read_parquet(scores_path)
        logger.info("Loaded precomputed scores from %s (%d rows)", scores_path, len(matrix))
    else:
        # Load model (handles .joblib / .json / .native; classifier or ranker)
        clf = load_model_any(model_path, model_kind)
        logger.info("Loaded %s model from %s", model_kind, model_path)
        matrix = load_training_matrix(matrix_path)
    ts_col = _find_ts_col(matrix)

    # Exclude blacklisted tickers BEFORE top-K selection so the next allowed name is
    # promoted into the top-K (rather than a blacklisted name silently shrinking the book).
    if apply_blacklist and "ticker" in matrix.columns:
        n0 = len(matrix)
        matrix = matrix[~matrix["ticker"].str.upper().isin({t.upper() for t in TRADING_BLACKLIST})].copy()
        logger.info("Applied TRADING_BLACKLIST: dropped %d rows (%d names)", n0 - len(matrix), len(TRADING_BLACKLIST))

    test_start = pd.Timestamp(test_start_str, tz="UTC")
    test_end   = pd.Timestamp(test_end_str, tz="UTC")

    # Generate signals for entire test window (all tickers together)
    signals_all = generate_signals(
        matrix, clf, test_start, test_end, feature_columns, cfg,
        model_kind=model_kind, selection=selection, top_k=top_k,
    )

    tickers = signals_all["ticker"].unique().tolist()
    logger.info("Running backtest for %d tickers (%s → %s)", len(tickers), test_start.date(), test_end.date())

    initial_equity = cfg["initial_capital"]
    equity: float  = initial_equity
    equity_times: list[pd.Timestamp] = [test_start]
    equity_vals:  list[float]        = [initial_equity]

    all_trades: list[Trade] = []
    open_positions: dict[str, Position] = {}

    for i, ticker in enumerate(tickers, 1):
        logger.info("  (%d/%d) %s", i, len(tickers), ticker)

        ticker_sigs = signals_all[signals_all["ticker"] == ticker].copy()
        if ticker_sigs.empty:
            continue

        df_5m  = _load_5m_slice(ticker, test_start, test_end)
        df_30m = _load_30m_slice(ticker, test_start, test_end)

        ticker_trades = _simulate_ticker(
            ticker=ticker,
            signals=ticker_sigs,
            df_5m=df_5m,
            df_30m=df_30m,
            cfg=cfg,
            equity_curve=equity_vals,
            equity_times=equity_times,
            initial_equity=initial_equity,
            ts_col=ts_col,
        )
        all_trades.extend(ticker_trades)

    # Build trades DataFrame
    if not all_trades:
        logger.warning("No trades generated in backtest window.")
        trades_df  = pd.DataFrame()
        equity_df  = pd.DataFrame({"timestamp": [test_start], "equity": [initial_equity]})
    else:
        trades_df = pd.DataFrame([vars(t) for t in all_trades]).sort_values("entry_time")

        # Reconstruct equity curve from sorted trades
        equity_series = [initial_equity]
        equity_ts     = [test_start]
        running = initial_equity
        for _, row in trades_df.iterrows():
            running += row["pnl_dollar"]
            equity_series.append(running)
            equity_ts.append(row["exit_time"])
        equity_df = pd.DataFrame({"timestamp": equity_ts, "equity": equity_series})

        # Summary stats
        total_trades = len(trades_df)
        win_rate     = (trades_df["pnl_pct"] > 0).mean()
        avg_win      = trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].mean()
        avg_loss     = trades_df.loc[trades_df["pnl_pct"] <= 0, "pnl_pct"].mean()
        total_pnl    = trades_df["pnl_dollar"].sum()
        max_dd       = _max_drawdown(equity_series)

        logger.info(
            "\n=== Backtest Summary ===\n"
            "  Trades:      %d\n"
            "  Win rate:    %.2f%%\n"
            "  Avg win:     %.2f%%\n"
            "  Avg loss:    %.2f%%\n"
            "  Total PnL:   $%.0f\n"
            "  Max DD:      %.2f%%\n"
            "  Final equity:$%.0f",
            total_trades, win_rate * 100,
            (avg_win  or 0) * 100,
            (avg_loss or 0) * 100,
            total_pnl, max_dd * 100, running,
        )

    trades_df.to_parquet(trades_path, index=False)
    equity_df.to_parquet(equity_path, index=False)
    logger.info("Results saved → %s", results_dir)
    return trades_df, equity_df


def _max_drawdown(equity: list[float]) -> float:
    arr = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / peak
    return float(dd.min())


def _find_ts_col(df: pd.DataFrame) -> str:
    for name in ("timestamp", "index", "t", "time", "date"):
        if name in df.columns:
            return name
    return df.columns[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-ticker swing backtest.")
    p.add_argument("--test-start", default="2024-01-01")
    p.add_argument("--test-end",   default="2026-04-15")
    p.add_argument("--force",      action="store_true")
    p.add_argument("--model",      default=None, help="Path to model .joblib/.json/.native (default: models/swing_xgb_model.json)")
    p.add_argument("--features",   default=None, help="Path to newline-delimited feature list .txt (default: FEATURE_COLUMNS from config)")
    p.add_argument("--results-dir", default=None, help="Directory to save results (default: backtest/results)")
    p.add_argument("--matrix",     default=None, help="Path to training matrix parquet (default: config TRAINING_MATRIX)")
    p.add_argument("--model-kind", default="classifier", choices=["classifier", "ranker"], help="Model output type")
    p.add_argument("--selection",  default="threshold", choices=["threshold", "topk", "bottomk", "both", "topk_short", "precomputed"], help="threshold (classifier) | topk longs | topk_short | bottomk shorts | both | precomputed (use signal column)")
    p.add_argument("--top-k",      type=int, default=10, help="K for --selection topk")
    p.add_argument("--scores",     default=None, help="Parquet of precomputed scores (timestamp,ticker,score), e.g. OOF preds — no model loaded")
    p.add_argument("--apply-blacklist", action="store_true", help="Exclude TRADING_BLACKLIST tickers before selection")
    p.add_argument("--tp-atr-mult",   type=float, default=None, help="Override take-profit ATR multiple")
    p.add_argument("--sl-atr-mult",   type=float, default=None, help="Override stop-loss ATR multiple")
    p.add_argument("--entry-bars-max", type=int,  default=None, help="Override max entry-confirmation bars")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    kwargs = dict(
        test_start_str=args.test_start, test_end_str=args.test_end, force=args.force,
        model_kind=args.model_kind, selection=args.selection, top_k=args.top_k,
        apply_blacklist=args.apply_blacklist,
    )
    overrides = {k: v for k, v in {
        "tp_atr_mult": args.tp_atr_mult,
        "sl_atr_mult": args.sl_atr_mult,
        "entry_bars_max": args.entry_bars_max,
    }.items() if v is not None}
    if overrides:
        kwargs["cfg"] = {**BACKTEST_CONFIG, **overrides}
        logger.info("Order-policy overrides: %s", overrides)
    if args.scores:
        kwargs["scores_path"] = Path(args.scores)
    if args.model:
        kwargs["model_path"] = Path(args.model)
    if args.matrix:
        kwargs["matrix_path"] = Path(args.matrix)
    if args.results_dir:
        kwargs["results_dir"] = Path(args.results_dir)
    if args.features:
        feat_path = Path(args.features)
        kwargs["feature_columns"] = [l.strip() for l in feat_path.read_text().splitlines() if l.strip()]
        logger.info("Loaded %d features from %s", len(kwargs["feature_columns"]), feat_path)
    simulate(**kwargs)
