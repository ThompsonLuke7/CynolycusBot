"""
SwingScanner — runs at each 30m bar close, computes directional probabilities
for all universe tickers, ranks signals by EV score, and returns the top N.

Signal ranking: ev_score = p_dir * avg_win_pct + (1-p_dir) * avg_loss_pct
This weights live signal strength (p_dir) by per-ticker historical trade quality.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategies.multi_ticker_swing.config.pipeline_config import FEATURE_COLUMNS, MODEL_PATH
from strategies.multi_ticker_swing.live.universe import TickerConfig, load_universe
from strategies.multi_ticker_swing.live.feature_builder import LiveSwingFeatureBuilder
from strategies.multi_ticker_swing.live.catalyst_signal import (
    LiveCatalystSignal,
    catalyst_ev_multiplier,
)

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    ticker: str
    direction: int          # 1=long, -1=short
    p_dir: float            # directional-conditional probability (long or short)
    ev_score: float         # expected trade return at this signal strength
    entry_threshold: float
    atr: float              # ATR at signal bar (for SL calculation)
    ref_high: float         # signal bar high (confirmation reference)
    ref_low: float          # signal bar low (confirmation reference)
    signal_ts: datetime = field(default_factory=lambda: datetime.now())
    config: TickerConfig | None = None
    features: dict[str, Any] = field(default_factory=dict)
    catalyst_score: float = float("nan")   # recency-weighted live catalyst strength (NaN = none)
    catalyst_count: int = 0                 # fresh catalyst records in the lookback window


class SwingScanner:
    """
    Loads the XGBoost model once, then scores all universe tickers on each 30m close.
    Returns signals sorted by EV score, capped at max_entries_per_bar.
    """

    def __init__(
        self,
        feature_builder: LiveSwingFeatureBuilder,
        model_path: Path | str = MODEL_PATH,
        max_entries_per_bar: int = 5,
        universe_path: Path | str | None = None,
        catalyst_signal: LiveCatalystSignal | None = None,
        catalyst_ev_alpha: float = 0.5,
    ) -> None:
        from xgboost import XGBClassifier
        self._clf = XGBClassifier()
        self._clf.load_model(str(model_path))
        logger.info("Loaded swing model from %s", model_path)

        self._fb = feature_builder
        self._universe = load_universe(universe_path) if universe_path else load_universe()
        self._max_entries = max_entries_per_bar
        # Optional live news tilt: re-ranks signals by fresh catalyst strength.
        # When None (or the ledger is absent) the scanner is unchanged.
        self._catalyst = catalyst_signal
        self._cat_alpha = float(catalyst_ev_alpha)
        logger.info(
            "Universe: %d tradeable tickers (live catalyst tilt %s)",
            len(self._universe),
            f"on, alpha={self._cat_alpha}" if catalyst_signal is not None else "off",
        )

    @property
    def universe(self) -> dict[str, TickerConfig]:
        return self._universe

    def scan(
        self,
        active_tickers: list[str],
        skip_tickers: set[str] | None = None,
    ) -> list[Signal]:
        """
        Compute probabilities for all active_tickers that have fresh features.
        Returns up to max_entries_per_bar signals sorted by EV score descending.

        skip_tickers: tickers already in CONFIRMING or OPEN state (don't double-enter).
        """
        skip = skip_tickers or set()
        candidates = [
            t for t in active_tickers
            if t in self._universe and t not in skip
        ]
        if not candidates:
            return []

        # Build feature matrix for all candidates in one batch
        rows: list[tuple[str, pd.Series]] = []
        for ticker in candidates:
            last_bar = self._fb.get_last_bar(ticker)
            if last_bar is None:
                continue
            feat = self._fb._compute_latest(ticker)
            if feat is None:
                continue
            rows.append((ticker, feat))

        if not rows:
            return []

        tickers_batch = [r[0] for r in rows]
        feat_df = pd.DataFrame([r[1] for r in rows], index=tickers_batch)

        # Align to model feature columns
        avail = [c for c in FEATURE_COLUMNS if c in feat_df.columns]
        X = feat_df[avail].values.astype(np.float32)

        try:
            proba = self._clf.predict_proba(X)   # (N, 3): [P(short), P(neutral), P(long)]
        except Exception as exc:
            logger.error("Model inference failed: %s", exc)
            return []

        signals: list[Signal] = []
        for idx, ticker in enumerate(tickers_batch):
            cfg = self._universe[ticker]
            p_long  = float(proba[idx, 2])
            p_short = float(proba[idx, 0])
            p_sum   = p_long + p_short
            if p_sum < 1e-8:
                continue
            p_long_dir  = p_long  / p_sum
            p_short_dir = p_short / p_sum

            direction = 0
            p_dir = 0.0
            if p_long_dir >= cfg.entry_threshold:
                direction, p_dir = 1, p_long_dir
            elif p_short_dir >= cfg.entry_threshold:
                direction, p_dir = -1, p_short_dir
            if direction == 0:
                continue

            ev = cfg.ev_score(p_dir)
            if ev <= 0:
                continue

            # Live news tilt: bullish catalysts boost longs / dampen shorts. The
            # multiplier is positive, so it only re-ranks — it never turns a
            # positive-EV signal negative.
            hit = self._catalyst.for_ticker(ticker) if self._catalyst is not None else None
            ev_base = ev
            ev = ev_base * catalyst_ev_multiplier(hit, direction, alpha=self._cat_alpha)
            cat_score = hit.score if hit is not None else float("nan")
            cat_count = hit.count if hit is not None else 0

            bar = self._fb.get_last_bar(ticker)
            atr = self._fb.get_atr(ticker)
            risk_features = {
                key: _jsonable(feat.get(key))
                for key in (
                    "qqq_ret_16",
                    "rel_str_qqq_4",
                    "rel_str_spy_16",
                    "stock_beta_bucket",
                    "beta_like_spy_64",
                    "daily_range_pos_20",
                    "zscore_close_64",
                    "range_pos_20",
                )
                if key in feat.index
            }
            risk_features["ev_score_base"] = float(ev_base)
            if hit is not None and hit.count:
                risk_features["catalyst_top_family"] = hit.top_family
                risk_features["catalyst_top_headline"] = hit.top_headline
            signals.append(Signal(
                ticker=ticker,
                direction=direction,
                p_dir=p_dir,
                ev_score=ev,
                entry_threshold=cfg.entry_threshold,
                atr=atr,
                ref_high=float(bar.get("high", float("nan"))) if bar else float("nan"),
                ref_low=float(bar.get("low",  float("nan"))) if bar else float("nan"),
                signal_ts=datetime.now(),
                config=cfg,
                features=risk_features,
                catalyst_score=cat_score,
                catalyst_count=cat_count,
            ))

        signals.sort(key=lambda s: s.ev_score, reverse=True)
        top = signals[:self._max_entries]
        if top:
            logger.info(
                "30m scan: %d signals → top %d  [%s]",
                len(signals), len(top),
                "  ".join(
                    f"{s.ticker}({'L' if s.direction==1 else 'S'}) "
                    f"p={s.p_dir:.2f} ev={s.ev_score:.4f}"
                    for s in top
                ),
            )
        return top

    def get_directional_p(self, ticker: str) -> tuple[float, float]:
        """
        Return (p_long_dir, p_short_dir) for any ticker with fresh features.
        Used by the SPY regime filter in the runner. Returns (nan, nan) if
        features are unavailable or model inference fails.
        """
        nan = float("nan")
        feat = self._fb._compute_latest(ticker)
        if feat is None:
            return nan, nan
        avail = [c for c in FEATURE_COLUMNS if c in feat.index]
        X = feat[avail].values.astype(np.float32).reshape(1, -1)
        try:
            proba = self._clf.predict_proba(X)[0]  # [P(short), P(neutral), P(long)]
        except Exception:
            return nan, nan
        p_long, p_short = float(proba[2]), float(proba[0])
        p_sum = p_long + p_short
        if p_sum < 1e-8:
            return nan, nan
        return p_long / p_sum, p_short / p_sum


def _jsonable(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value
