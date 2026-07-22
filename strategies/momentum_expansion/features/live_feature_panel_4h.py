"""
Shared live 4H feature-panel builder.

The single place every live 4H scoring call site (momentum's own ranking,
meta_ranker's shared matrix update) should go to turn a ticker list into a
complete, scoring-ready (timestamp, ticker) feature panel. Consolidates what
used to be two independently-duplicated per-ticker build loops
(update_meta_matrix.py's _build_live_features and momentum's
live/runner.py::evaluate_now) into one function, so the two batch-only
post-processing steps below can never again be wired into one live call site
and silently forgotten in the other (see LIVING_SUMMARY.md 2026-07-19: both
deployed boosters were scoring live with 5-8 missing manifest columns,
indefinitely, with zero errors, because of exactly that duplication).

Bar loading is injected (caller-supplied callables), not owned here: the two
call sites read bars through different backends (update_meta_matrix.py's raw
parquet reader vs momentum's data.load_bars + its own stale-context policy),
and this module has no opinion on that — only on what happens once bars are
in hand.

Usage:
    result = build_live_feature_panel_4h(
        tickers=tickers, load_4h_bars=load_4h, load_1d_bars=_load_daily,
        ctx_4h=ctx_4h, tail=1,
    )
    assert_manifest_coverage(scorer=ranker, panel=result.panel, label="momentum")
    scores = ranker.score(result.panel)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

from strategies.momentum_expansion.features.feature_matrix_4h import (
    _add_cross_sectional_features,
    _add_earnings_features_4h,
    build_ticker_features_4h,
)

logger = logging.getLogger(__name__)

# Raises FileNotFoundError (or any exception) if bars are uncached for a ticker.
BarLoader = Callable[[str], pd.DataFrame]
# Never raises; returns None if daily bars are uncached for a ticker.
DailyBarLoader = Callable[[str], "pd.DataFrame | None"]


@dataclass
class FeaturePanelResult:
    panel: pd.DataFrame  # (timestamp, ticker) MultiIndex; xsec_*/earnings cols ALWAYS present
    tickers_requested: int
    tickers_built: int
    tickers_skipped: list[str]
    build_seconds: float


def build_live_feature_panel_4h(
    *,
    tickers: Iterable[str],
    load_4h_bars: BarLoader,
    load_1d_bars: DailyBarLoader,
    ctx_4h: dict[str, pd.DataFrame],
    tail: int | None = None,
    since: pd.Timestamp | None = None,
    progress_every: int = 200,
) -> FeaturePanelResult:
    """Build the full live 4H feature panel for `tickers`.

    1. tickers    -- caller-supplied list
    2. raw bars   -- via the injected loader callables
    3. features   -- build_ticker_features_4h per ticker, then `tail` and/or
                     `since` trim (whichever the caller supplies; both are
                     no-ops if left None)
    4. post-process -- _add_cross_sectional_features then
                     _add_earnings_features_4h, unconditionally, over the
                     WHOLE concatenated panel. There is no parameter to skip
                     this step: every panel this function returns has every
                     xsec_*/earnings column the batch build also produces.

    Note: `ticker_meta_lookup` (sector_id/market_cap_bucket/asset_type/is_etf)
    is intentionally NOT wired here. Both prior live call sites omitted it too
    (this function doesn't regress that), and fixing it isn't a simple
    "pass the metadata through": the batch path's sector/cap/type integer
    encodings are recomputed fresh from whatever candidate metadata is loaded
    at build time (see feature_matrix_4h._build_metadata_encodings), not
    frozen anywhere a live build could reload them consistently with what a
    given trained booster actually saw. That's a separate, real gap -- flagged,
    not fixed here.
    """
    t0 = time.time()
    tickers = list(tickers)
    rows: list[pd.DataFrame] = []
    skipped: list[str] = []

    for i, t in enumerate(tickers, 1):
        try:
            df_4h = load_4h_bars(t)
        except Exception:
            skipped.append(t)
            continue
        if df_4h is None or df_4h.empty:
            skipped.append(t)
            continue
        try:
            df_1d = load_1d_bars(t)
        except Exception:
            df_1d = None

        try:
            feats = build_ticker_features_4h(ticker=t, df_4h=df_4h, df_1d=df_1d, ctx_4h=ctx_4h)
        except Exception:
            skipped.append(t)
            continue
        if feats is None or feats.empty:
            skipped.append(t)
            continue

        if tail is not None:
            feats = feats.tail(tail)
        feats = feats.reset_index().rename(columns={"index": "timestamp"})
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
        if since is not None:
            feats = feats[feats["timestamp"] > since]
        if feats.empty:
            continue
        feats["ticker"] = t
        rows.append(feats)

        if progress_every and i % progress_every == 0:
            logger.info("live feature panel: %d/%d tickers built", i, len(tickers))

    if not rows:
        return FeaturePanelResult(
            panel=pd.DataFrame(), tickers_requested=len(tickers), tickers_built=0,
            tickers_skipped=skipped, build_seconds=time.time() - t0,
        )

    panel = pd.concat(rows, ignore_index=True).set_index(["timestamp", "ticker"]).sort_index()
    panel = _add_cross_sectional_features(panel)
    panel = _add_earnings_features_4h(panel)

    n_built = len(tickers) - len(skipped)
    build_seconds = time.time() - t0
    logger.info(
        "live feature panel built: %d/%d tickers (%d skipped), %d rows, %.1fs",
        n_built, len(tickers), len(skipped), len(panel), build_seconds,
    )
    if skipped:
        logger.warning(
            "live feature panel: %d/%d tickers skipped (no bars / build failure): %s%s",
            len(skipped), len(tickers), skipped[:10], " ..." if len(skipped) > 10 else "",
        )

    return FeaturePanelResult(
        panel=panel, tickers_requested=len(tickers), tickers_built=n_built,
        tickers_skipped=skipped, build_seconds=build_seconds,
    )


def assert_manifest_coverage(
    *, scorer, panel: pd.DataFrame, label: str, raise_on_missing: bool = False,
) -> list[str]:
    """Diff `scorer.feature_columns` (ExpansionRanker | HTFSwingScorer | anything
    with that attribute) against `panel.columns`. Logs a loud warning (or raises
    ValueError if `raise_on_missing`) listing exactly what's missing, and returns
    the missing list so a caller can branch on it.

    This is the structural check the 2026-07-19 audit's bug would have tripped
    immediately: call it right before scoring at every live call site.
    """
    expected = getattr(scorer, "feature_columns", None)
    if not expected:
        return []
    missing = [c for c in expected if c not in panel.columns]
    if missing:
        msg = (f"{label}: {len(missing)}/{len(expected)} manifest feature columns "
               f"missing from the live panel: {missing}")
        if raise_on_missing:
            raise ValueError(msg)
        logger.warning(msg)
    return missing
