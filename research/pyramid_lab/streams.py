"""Per-module out-of-sample top-10 signal streams + 4H bar loading.

Module coverage and score sources deliberately match
``scripts/capstone/exit_policy_cross_module.py`` (the prior exit-policy search
this study is a delta against): each module is replayed on ITS OWN walk-forward
OOF top-10 stream. Deployed scores are not used — ``leakage_audit.md`` §4.3
documents that they partially in-sample the holdout.

The only deviation from that script is meta's source: it read
``/tmp/meta_scored.parquet``, a temp file that no longer exists. This module
rebuilds the identical leak-free ``s_combo`` via
``scripts.capstone.build_meta_scored_from_oof.build_oof_combo_scores``
(imported unchanged) — the same substitution
``scripts/capstone/reproduce_results.py::meta_exit_policy_lock`` already makes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from signals.meta_context.meta_ranker import backtest_exits as be

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
TOPK = be.TOPK  # 10

OOF_SOURCES = {
    "momentum": REPO / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet",
    "htf": REPO / "strategies/multi_ticker_swing_htf/models/oof_preds.parquet",
}
MODULES = ("momentum", "htf", "meta")


def load_module_stream(module: str, *, top_k: int = TOPK) -> pd.DataFrame:
    """``[timestamp, ticker, in_top]`` over the module's FULL OOF span.

    Mirrors ``exit_policy_cross_module.load_member`` (per-timestamp rank of the
    module's own OOF score, ``in_top = rank <= top_k``) minus its hard-coded
    val/test date window — periods are sliced later from the repo walk-forward
    fold spec instead.
    """
    if module == "meta":
        from scripts.capstone.build_meta_scored_from_oof import build_oof_combo_scores
        df = build_oof_combo_scores().dropna(subset=["s_combo"])
        score_col = "s_combo"
    else:
        df = pd.read_parquet(OOF_SOURCES[module]).reset_index()
        df = df.dropna(subset=["score"])
        score_col = "score"
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ticker"] = df["ticker"].astype(str)
    # Both momentum's and HTF's OOF carry a small number of duplicated
    # (timestamp, ticker) rows concentrated on fold-BOUNDARY bars, where two
    # adjacent walk-forward folds both scored the same bar (htf: 1,705 pairs on
    # 2023-05-22 / 2025-05-22 of 3,838 bars; momentum: 1,925 pairs). Dedupe
    # keep-first -- the same convention `build_meta_scored_from_oof._read_oof`
    # already applies to meta's OOF. Baseline and every arm see the identical
    # deduped stream, so this cannot favour one arm over another.
    n0 = len(df)
    df = df.drop_duplicates(subset=["timestamp", "ticker"], keep="first")
    if len(df) != n0:
        logger.info("[%s] dropped %d duplicated (timestamp, ticker) OOF rows (keep-first)",
                    module, n0 - len(df))
    df["rk"] = df.groupby("timestamp")[score_col].rank(ascending=False, method="first")
    top = df.loc[df["rk"] <= top_k, ["timestamp", "ticker"]].copy()
    top["in_top"] = True
    logger.info("[%s] %d top-%d rows, %d tickers, %s .. %s", module, len(top), top_k,
                top["ticker"].nunique(), top["timestamp"].min(), top["timestamp"].max())
    return top


class Bars:
    """4H OHLC cache. Uses ``backtest_exits._ticker_path`` unchanged, the same
    loader the prior exit-policy search used, so bar handling (dedupe on
    timestamp, sort, UTC) is identical."""

    def __init__(self) -> None:
        self._cache: dict[str, pd.DataFrame | None] = {}

    def get(self, ticker: str) -> pd.DataFrame | None:
        if ticker not in self._cache:
            self._cache[ticker] = be._ticker_path(ticker, None)
        return self._cache[ticker]


def build_member_masks(stream: pd.DataFrame, bars: Bars) -> dict[str, dict]:
    """Per ticker: bar arrays + a boolean top-K membership mask aligned to them.

    Reindexing membership onto the ticker's own bar index (missing -> False) is
    exactly what ``backtest_exits.simulate`` does; a signal timestamp with no
    matching bar simply never opens a position.
    """
    out: dict[str, dict] = {}
    for ticker, g in stream.groupby("ticker", sort=True):
        b = bars.get(ticker)
        if b is None or len(b) < 20:
            continue
        mask = (g.set_index("timestamp")["in_top"]
                .reindex(b.index).fillna(False).astype(bool).to_numpy())
        if not mask.any():
            continue
        out[ticker] = {
            # naive-UTC datetime64[ns]: a tz-aware DatetimeIndex.to_numpy()
            # yields an OBJECT array of Timestamps, which is both slow to
            # searchsorted and unusable as a DatetimeIndex downstream.
            "ts": b.index.tz_convert("UTC").tz_localize(None).to_numpy(),
            "close": b["close"].to_numpy(float),
            "high": b["high"].to_numpy(float),
            "low": b["low"].to_numpy(float),
            "member": mask,
        }
    return out


def master_index(masks: dict[str, dict]) -> np.ndarray:
    """Union of every involved ticker's bar timestamps, sorted — the common
    grid the per-bar P&L / deployed-capital series are accumulated onto."""
    if not masks:
        return np.array([], dtype="datetime64[ns]")
    return np.unique(np.concatenate([m["ts"] for m in masks.values()]))
