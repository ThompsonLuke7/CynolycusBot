"""
Shared, model-family-agnostic test-set backtest engine.

Used by both momentum_expansion and multi_ticker_swing_htf to answer two questions
on the held-out (last-20%-by-timestamp) test window:

  1. What is the best *order policy* (TP/SL ATR mult, top_k, max-hold) for a strategy?
  2. Do classifiers or rankers produce the better tradeable signal?

Design (apples-to-apples across families):
  - score every (timestamp, ticker) test row with a competition model;
  - per timestamp, take the top_k names long (+1) and, optionally, the bottom_k short (-1);
  - execute on the underlying 4H bars: enter at the *next* 4H open after the signal bar,
    exit on an ATR take-profit / ATR stop-loss / max-holding time stop;
  - P&L is the underlying % move on a fixed notional (sizing-neutral signal quality).

Scoring is separated from execution so the policy sweep only re-runs the cheap
execution layer over cached per-family scores.

Correctness guards (see plan):
  - features are always reindexed to the manifest column order before scoring
    (xgb sklearn / native take positional input — wrong order silently corrupts);
  - xgb families load the canonical ``.native`` booster (the joblib pickle triggers a
    cross-version warning and predicts slightly off); lgbm families load joblib;
  - classifier probability is P(positive=1) via ``classes_`` (binary);
  - the test cutoff is strictly global + time-based; entries use the *next* bar (no leakage).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from strategies.momentum_expansion.data.load_bars import load_4h

logger = logging.getLogger(__name__)

FAMILIES = ["xgb_classifier", "xgb_ranker", "lgbm_classifier", "lgbm_ranker"]
NOTIONAL = 1_000.0          # fixed $ per trade (sizing-neutral)
INITIAL_CAPITAL = 100_000.0


# ---------------------------------------------------------------------------
# Test-window loading
# ---------------------------------------------------------------------------

def _read_reset(mtx: Path, cols: list[str], filt=None) -> pd.DataFrame:
    """Read parquet columns and surface the stored (timestamp, ticker) index as columns."""
    df = pq.read_table(str(mtx), columns=cols, filters=filt).to_pandas().reset_index()
    if "index" in df.columns and "index" not in cols:
        df = df.drop(columns=["index"])
    return df


def compute_test_cutoff(mtx: Path, target: str, train_frac: float, val_frac: float) -> pd.Timestamp:
    """Global, time-based split boundary: sort by timestamp, drop null target, take row int(n*(train+val))."""
    t = _read_reset(mtx, ["timestamp", target])
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    t = t.sort_values("timestamp").dropna(subset=[target]).reset_index(drop=True)
    return t["timestamp"].iloc[int(len(t) * (train_frac + val_frac))]


@dataclass
class StrategyConfig:
    name: str
    models_dir: Path
    matrix_path: Path
    feature_cols: list[str]
    target: str                 # regression target used only to define the split
    relevance_src: str          # forward-return col used to reconstruct the binary relevance label
    relevance_thr: float
    forward_window: int         # bars; drives default max-hold grid
    allow_short: bool
    train_frac: float = 0.6
    val_frac: float = 0.2
    exclude_low_price: bool = True

    @staticmethod
    def from_manifest(name, models_dir, matrix_path, *, allow_short, forward_window):
        models_dir = Path(models_dir)
        em = json.loads((models_dir / "eval_metrics.json").read_text())
        feats = em["feature_columns"]
        fm = json.loads((models_dir / "feature_manifest.json").read_text())
        return StrategyConfig(
            name=name, models_dir=models_dir, matrix_path=Path(matrix_path),
            feature_cols=feats,
            target=fm.get("regression_target_column", fm["target_column"]),
            relevance_src=fm["strong_setup_source_column"],
            relevance_thr=float(fm["strong_setup_threshold"]),
            forward_window=forward_window, allow_short=allow_short,
            train_frac=float(fm.get("train_frac", 0.6)), val_frac=float(fm.get("val_frac", 0.2)),
        )


def load_test_window(cfg: StrategyConfig) -> pd.DataFrame:
    """Return the test-slice frame: [timestamp, ticker, relevance, <relevance_src>, *features].

    Low-priced names are excluded (``low_price_flag``) to mirror the live ranking gate
    (momentum_candidate_mask: ``exclude_low_price``). Neither live strategy trades sub-$1
    microcaps, whose huge %-ATR otherwise dominates the underlying-return P&L tails.
    """
    cutoff = compute_test_cutoff(cfg.matrix_path, cfg.target, cfg.train_frac, cfg.val_frac)
    cols = ["timestamp", "ticker", cfg.relevance_src] + cfg.feature_cols
    cols = list(dict.fromkeys(cols))  # de-dup if relevance_src is also a feature
    df = _read_reset(cfg.matrix_path, cols, [("timestamp", ">=", cutoff.to_pydatetime())])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ticker"] = df["ticker"].astype(str)
    n0 = len(df)
    if cfg.exclude_low_price and "low_price_flag" in df.columns:
        keep = pd.to_numeric(df["low_price_flag"], errors="coerce").fillna(1.0) <= 0.0
        df = df[keep].reset_index(drop=True)
    df["relevance"] = (df[cfg.relevance_src] >= cfg.relevance_thr).astype(int)
    logger.info("[%s] test window: cutoff=%s rows=%d (low-price dropped %d) tickers=%d base_rate=%.4f",
                cfg.name, cutoff.date(), len(df), n0 - len(df), df["ticker"].nunique(), df["relevance"].mean())
    return df


# ---------------------------------------------------------------------------
# Model loading + scoring
# ---------------------------------------------------------------------------

def best_seed_per_family(models_dir: Path) -> dict[str, int]:
    s = pd.read_csv(Path(models_dir) / "seed_results.csv")
    idx = s.groupby("family")["test_ndcg_at_10"].idxmax()
    return {r.family: int(r.seed) for r in s.loc[idx].itertuples()}


def score_family(df: pd.DataFrame, cfg: StrategyConfig, family: str, seed: int) -> np.ndarray:
    """Score the test frame with one (family, seed). Returns a float score array (higher = stronger long)."""
    feats = cfg.feature_cols
    X = df.reindex(columns=feats).astype("float32")
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"{cfg.name}: {len(missing)} feature cols missing from matrix: {missing[:10]}")
    if family.startswith("xgb"):
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(cfg.models_dir / f"model_{family}_seed{seed}.native"))
        return booster.predict(xgb.DMatrix(X.values, feature_names=feats, missing=np.nan))
    import joblib
    model = joblib.load(cfg.models_dir / f"model_{family}_seed{seed}.joblib")
    Xn = X.copy()
    Xn.columns = list(feats)  # keep names so lgbm doesn't warn
    if "classifier" in family:
        classes = list(model.classes_)
        pos = classes.index(1) if 1 in classes else len(classes) - 1
        return model.predict_proba(Xn)[:, pos]
    return model.predict(Xn)


# ---------------------------------------------------------------------------
# Signal selection
# ---------------------------------------------------------------------------

def select_signals(scored: pd.DataFrame, top_k: int, allow_short: bool) -> pd.DataFrame:
    """Per-timestamp top_k -> long (+1); if allow_short, bottom_k -> short (-1)."""
    g = scored.groupby("timestamp")["score"]
    hi = g.rank(ascending=False, method="first")
    out = scored.loc[hi <= top_k, ["timestamp", "ticker", "score"]].copy()
    out["direction"] = 1
    if allow_short:
        lo = g.rank(ascending=True, method="first")
        sh = scored.loc[lo <= top_k, ["timestamp", "ticker", "score"]].copy()
        sh["direction"] = -1
        out = pd.concat([out, sh], ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Execution on 4H bars (cached per ticker)
# ---------------------------------------------------------------------------

def _atr(high, low, close, length=14):
    h, l, pc = high, low, np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    out = np.full(len(tr), np.nan)
    if len(tr) >= length:
        c = np.cumsum(np.nan_to_num(tr))
        out[length - 1:] = (c[length - 1:] - np.concatenate([[0], c[:-length]])) / length
    return out


@dataclass
class BarCache:
    cache: dict = field(default_factory=dict)

    def get(self, ticker: str):
        if ticker not in self.cache:
            try:
                df = load_4h(ticker)
                idx = df.index.tz_convert("UTC")
                self.cache[ticker] = {
                    "ts": idx.asi8,                  # int64 ns (UTC) for fast, warning-free searchsorted
                    "ts_dt": idx.to_numpy(),         # datetime64 for trade labels
                    "open": df["open"].to_numpy(float),
                    "high": df["high"].to_numpy(float),
                    "low": df["low"].to_numpy(float),
                    "close": df["close"].to_numpy(float),
                    "atr": _atr(df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)),
                }
            except FileNotFoundError:
                self.cache[ticker] = None
        return self.cache[ticker]


@dataclass
class Trade:
    ticker: str
    direction: int
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_dollar: float
    exit_reason: str
    bars_held: int


def _simulate_signal(bars, signal_ts, direction, tp_mult, sl_mult, max_hold):
    """Resolve one trade outcome. Entry = next bar open after signal_ts. SL has same-bar priority."""
    ts = bars["ts"]
    sig_i = np.searchsorted(ts, signal_ts.value, side="right") - 1   # bar at/just-before signal
    if sig_i < 0:
        return None
    atr0 = bars["atr"][sig_i]
    if not np.isfinite(atr0) or atr0 <= 0:
        return None
    entry_i = sig_i + 1
    if entry_i >= len(ts):
        return None
    entry = bars["open"][entry_i]
    if not np.isfinite(entry) or entry <= 0:
        return None
    tp = entry + direction * tp_mult * atr0
    sl = entry - direction * sl_mult * atr0
    last = min(entry_i + max_hold, len(ts) - 1)
    lvl = dict(atr_at_entry=atr0, tp_price=tp, sl_price=sl)
    for j in range(entry_i, last + 1):
        hi, lo = bars["high"][j], bars["low"][j]
        if direction > 0:
            if lo <= sl:
                return _mk(bars, j, entry_i, direction, entry, sl, "sl", lvl)
            if hi >= tp:
                return _mk(bars, j, entry_i, direction, entry, tp, "tp", lvl)
        else:
            if hi >= sl:
                return _mk(bars, j, entry_i, direction, entry, sl, "sl", lvl)
            if lo <= tp:
                return _mk(bars, j, entry_i, direction, entry, tp, "tp", lvl)
    return _mk(bars, last, entry_i, direction, entry, bars["close"][last], "time", lvl)


def _mk(bars, j, entry_i, direction, entry, exit_price, reason, lvl):
    pnl_pct = (exit_price - entry) / entry * direction
    return dict(entry_i=entry_i, exit_i=j, entry_ts=bars["ts_dt"][entry_i], exit_ts=bars["ts_dt"][j],
                entry_price=entry, exit_price=exit_price, pnl_pct=pnl_pct,
                pnl_dollar=pnl_pct * NOTIONAL, exit_reason=reason, bars_held=j - entry_i, **lvl)


def simulate(signals: pd.DataFrame, bar_cache: BarCache, *, tp_mult, sl_mult, max_hold) -> pd.DataFrame:
    rows = []
    for tkr, grp in signals.groupby("ticker"):
        bars = bar_cache.get(tkr)
        if bars is None or len(bars["ts"]) < 20:
            continue
        for r in grp.itertuples():
            res = _simulate_signal(bars, r.timestamp, r.direction, tp_mult, sl_mult, max_hold)
            if res is None:
                continue
            res.update(ticker=tkr, direction=r.direction, score=r.score)
            rows.append(res)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0}
    t = trades.sort_values("exit_ts").reset_index(drop=True)
    eq = INITIAL_CAPITAL + t["pnl_dollar"].cumsum()
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    pos = t.loc[t.pnl_dollar > 0, "pnl_dollar"].sum()
    neg = -t.loc[t.pnl_dollar <= 0, "pnl_dollar"].sum()
    total_pnl = t["pnl_dollar"].sum()
    total_ret = total_pnl / INITIAL_CAPITAL
    r = t["pnl_pct"].to_numpy()
    sharpe_pt = float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else float("nan")
    return {
        "trades": int(len(t)),
        "win_rate": round(float((t.pnl_pct > 0).mean()), 4),
        "profit_factor": round(float(pos / neg), 3) if neg > 0 else float("inf"),
        "avg_trade_pct": round(float(r.mean()) * 100, 4),
        "median_trade_pct": round(float(np.median(r)) * 100, 4),
        "total_pnl": round(float(total_pnl), 2),
        "total_return_pct": round(float(total_ret) * 100, 3),
        "max_dd_pct": round(float(dd) * 100, 3),
        "ret_over_dd": round(float(total_ret / abs(dd)), 3) if dd < 0 else float("inf"),
        "sharpe_seq": round(sharpe_pt, 3),
        "tp_rate": round(float((t.exit_reason == "tp").mean()), 3),
        "sl_rate": round(float((t.exit_reason == "sl").mean()), 3),
        "time_rate": round(float((t.exit_reason == "time").mean()), 3),
        "avg_bars_held": round(float(t.bars_held.mean()), 2),
    }


# ---------------------------------------------------------------------------
# Order-policy sweep (per family, scores cached)
# ---------------------------------------------------------------------------

def default_grid(forward_window: int) -> dict:
    return {
        "tp_atr_mult": [2.0, 3.0, 4.0, 5.0],
        "sl_atr_mult": [2.0, 3.0, 4.0, 5.0],
        "top_k": [5, 10, 15, 20],
        "max_hold": [forward_window, 2 * forward_window, 3 * forward_window],
    }


def sweep_family(scored: pd.DataFrame, cfg: StrategyConfig, bar_cache: BarCache,
                 grid: dict, family: str, rank_metric="ret_over_dd") -> pd.DataFrame:
    """Run all policy combos for one family's cached scores. scored has [timestamp,ticker,score]."""
    out = []
    # group by top_k so signal selection (and its bar simulation) is reused across tp/sl/hold
    for top_k in grid["top_k"]:
        sig = select_signals(scored, top_k, cfg.allow_short)
        for tp, sl, hold in product(grid["tp_atr_mult"], grid["sl_atr_mult"], grid["max_hold"]):
            trades = simulate(sig, bar_cache, tp_mult=tp, sl_mult=sl, max_hold=hold)
            m = metrics(trades)
            m.update(family=family, top_k=top_k, tp_atr_mult=tp, sl_atr_mult=sl, max_hold=hold)
            out.append(m)
    res = pd.DataFrame(out)
    return res.sort_values(rank_metric, ascending=False).reset_index(drop=True)
