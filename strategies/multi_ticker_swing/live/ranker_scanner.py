"""
RankerSwingScanner — drop-in replacement for SwingScanner backed by the walk-forward
OOF long + short xgb_rankers (see models/oof_ranker_20260618/, data/bundle/).

Why this is a drop-in: the live runner, the EV ranking, and the SPY macro filter are all built
around two directional probabilities (p_long_dir, p_short_dir). A ranker emits one uncalibrated
score per side, so we:
  1. run the long ranker AND the short ranker on every candidate,
  2. map each raw score -> P(swing winner) via a per-side isotonic calibrator (fit on OOF preds),
  3. normalise to directional probs:  p_long_dir = P_long / (P_long + P_short), etc.
These p_*_dir are on the same 0..1 directional scale as the old classifier's, so `entry_threshold`,
`ev_score = p_dir*avg_win + (1-p_dir)*avg_loss`, and the runner's `spy_min` macro veto all work
unchanged. Direction = the stronger side, exactly as before.

Feature note: the rankers expect the 192-feature set (meta.json `feature_names`). The live feature
builder currently produces the ~123 base features; missing columns are reindexed to NaN and handled
natively by XGBoost. The only *important* missing features are bar-location (gaps / 52w distance /
S-R proximity, ~top-importance) — those are price-derived and should be added to the live builder
for full fidelity. The external context (news/theme/treasury/earnings/macro) is <15% of importance
and absent from the top-50, so NaN-serving it is acceptable for a first live run. See README_LIVE.md.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategies.multi_ticker_swing.config.pipeline_config import MODULE_ROOT
from strategies.multi_ticker_swing.live.universe import TickerConfig, load_universe
from strategies.multi_ticker_swing.live.feature_builder import LiveSwingFeatureBuilder
from strategies.multi_ticker_swing.live.catalyst_signal import (
    LiveCatalystSignal,
    catalyst_ev_multiplier,
)
from strategies.multi_ticker_swing.live.scanner import Signal, _jsonable

logger = logging.getLogger(__name__)

_BUNDLE = MODULE_ROOT / "models" / "oof_ranker_20260618"


class RankerSwingScanner:
    """OOF long+short ranker scanner. Same public surface as SwingScanner."""

    def __init__(
        self,
        feature_builder: LiveSwingFeatureBuilder,
        *,
        model_path: Path | str | None = None,   # accepted+ignored for drop-in compatibility
        bundle_dir: Path | str = _BUNDLE,
        max_entries_per_bar: int = 5,
        universe_path: Path | str | None = None,
        catalyst_signal: LiveCatalystSignal | None = None,
        catalyst_ev_alpha: float = 0.5,
        min_conviction: float = 0.30,
    ) -> None:
        import joblib

        bundle = Path(bundle_dir)
        self._long  = joblib.load(bundle / "model_ranker_long.joblib")
        self._short = joblib.load(bundle / "model_ranker_short.joblib")
        self._calib_long  = joblib.load(bundle / "calib_long.joblib")
        self._calib_short = joblib.load(bundle / "calib_short.joblib")
        self._feats: list[str] = json.loads((bundle / "meta.json").read_text())["feature_names"]
        logger.info("Loaded OOF long+short rankers (%d features) from %s", len(self._feats), bundle)

        import dataclasses
        self._fb = feature_builder
        uni = load_universe(universe_path) if universe_path else load_universe()
        # v2 policy runs the SPY macro veto OFF — this is a new model, learn live what it needs.
        # TickerConfig is frozen, so rebuild each with spy_min=0 (runner skips the veto when 0).
        self._universe = {t: dataclasses.replace(c, spy_min=0.0) for t, c in uni.items()}
        self._max_entries = max_entries_per_bar
        self._min_conviction = float(min_conviction)
        self._catalyst = catalyst_signal
        self._cat_alpha = float(catalyst_ev_alpha)
        self._warned_missing = False
        logger.info(
            "Universe: %d tradeable tickers (live catalyst tilt %s)",
            len(self._universe),
            f"on, alpha={self._cat_alpha}" if catalyst_signal is not None else "off",
        )

    @property
    def universe(self) -> dict[str, TickerConfig]:
        return self._universe

    # -- core: features -> RAW calibrated P(win) per side --------------------

    def _raw_p(self, feat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (P_long_win, P_short_win) — calibrated P(swing winner) per side.

        We rank by these RAW probabilities, NOT the directional ratio P_long/(P_long+P_short):
        on neutral bars both raw probs are ~0 so junk never makes the top-K, whereas the ratio
        flips to extremes on noise (that ratio cost ~12pp of win rate in backtest)."""
        if not self._warned_missing:
            # Two different conditions, previously reported as one. A column the
            # builder never produces is a real gap vs the training matrix; a
            # column present but all-NaN just means insufficient history for
            # this scan (e.g. gap_fill_rate_60d needs 5 gap days in 60 sessions).
            missing = [c for c in self._feats if c not in feat_df.columns]
            all_nan = [c for c in self._feats
                       if c in feat_df.columns and feat_df[c].isna().all()]
            if missing:
                logger.warning(
                    "RankerScanner: %d/%d features absent from live builder (NaN-served): %s",
                    len(missing), len(self._feats), missing[:8])
            if all_nan:
                logger.info(
                    "RankerScanner: %d/%d features present but all-NaN this scan "
                    "(insufficient history, not a builder gap): %s",
                    len(all_nan), len(self._feats), all_nan[:8])
            if missing or all_nan:
                self._warned_missing = True
        X = feat_df.reindex(columns=self._feats).to_numpy(dtype=np.float32)
        p_long  = np.clip(self._calib_long.predict(self._long.predict(X)),   1e-9, 1.0)
        p_short = np.clip(self._calib_short.predict(self._short.predict(X)), 1e-9, 1.0)
        return p_long, p_short

    def scan(
        self,
        active_tickers: list[str],
        skip_tickers: set[str] | None = None,
    ) -> list[Signal]:
        skip = skip_tickers or set()
        candidates = [t for t in active_tickers if t in self._universe and t not in skip]
        if not candidates:
            return []

        rows: list[tuple[str, pd.Series]] = []
        for ticker in candidates:
            if self._fb.get_last_bar(ticker) is None:
                continue
            feat = self._fb.compute_latest_full(ticker)
            if feat is not None:
                rows.append((ticker, feat))
        if not rows:
            return []

        tickers_batch = [r[0] for r in rows]
        feat_df = pd.DataFrame([r[1] for r in rows], index=tickers_batch)
        try:
            p_long_win, p_short_win = self._raw_p(feat_df)
        except Exception as exc:
            logger.error("Ranker inference failed: %s", exc)
            return []

        signals: list[Signal] = []
        for idx, ticker in enumerate(tickers_batch):
            cfg = self._universe[ticker]
            pl, ps = float(p_long_win[idx]), float(p_short_win[idx])

            # Direction = stronger raw side; conviction = its calibrated P(win). Skip when even
            # the best side is weak (neutral bar). max_entries sorting below does the top-K.
            p_dir = max(pl, ps)
            if p_dir < self._min_conviction:
                continue
            direction = 1 if pl >= ps else -1

            ev = cfg.ev_score(p_dir)
            if ev <= 0:
                continue

            hit = self._catalyst.for_ticker(ticker) if self._catalyst is not None else None
            ev_base = ev
            ev = ev_base * catalyst_ev_multiplier(hit, direction, alpha=self._cat_alpha)

            feat = feat_df.iloc[idx]
            bar = self._fb.get_last_bar(ticker)
            atr = self._fb.get_atr(ticker)
            risk_features = {
                key: _jsonable(feat.get(key))
                for key in ("qqq_ret_16", "rel_str_qqq_4", "rel_str_spy_16", "stock_beta_bucket",
                            "beta_like_spy_64", "daily_range_pos_20", "zscore_close_64", "range_pos_20")
                if key in feat.index
            }
            risk_features["ev_score_base"] = float(ev_base)
            risk_features["p_long_win"] = pl
            risk_features["p_short_win"] = ps
            if hit is not None and hit.count:
                risk_features["catalyst_top_family"] = hit.top_family
                risk_features["catalyst_top_headline"] = hit.top_headline

            signals.append(Signal(
                ticker=ticker, direction=direction, p_dir=p_dir, ev_score=ev,
                entry_threshold=cfg.entry_threshold, atr=atr,
                ref_high=float(bar.get("high", float("nan"))) if bar else float("nan"),
                ref_low=float(bar.get("low", float("nan"))) if bar else float("nan"),
                signal_ts=datetime.now(), config=cfg, features=risk_features,
                catalyst_score=hit.score if hit is not None else float("nan"),
                catalyst_count=hit.count if hit is not None else 0,
            ))

        signals.sort(key=lambda s: s.ev_score, reverse=True)
        top = signals[: self._max_entries]
        if top:
            logger.info(
                "30m ranker scan: %d signals → top %d  [%s]",
                len(signals), len(top),
                "  ".join(f"{s.ticker}({'L' if s.direction == 1 else 'S'}) "
                         f"p={s.p_dir:.2f} ev={s.ev_score:.4f}" for s in top),
            )
        return top

    def get_directional_p(self, ticker: str) -> tuple[float, float]:
        """(p_long_dir, p_short_dir) for one ticker — directional ratio, kept for any legacy
        caller (the SPY macro veto is off in the v2 policy)."""
        nan = float("nan")
        feat = self._fb.compute_latest_full(ticker)
        if feat is None:
            return nan, nan
        try:
            pl, ps = self._raw_p(pd.DataFrame([feat], index=[ticker]))
        except Exception:
            return nan, nan
        d = float(pl[0] + ps[0])
        return (float(pl[0] / d), float(ps[0] / d)) if d > 0 else (nan, nan)
