"""Regime-conditional policy replay engine.

Reuses, unchanged:
  * ``family_backtest.BarCache`` / ``family_backtest._simulate_signal`` —
    the TP/SL/time-stop exit resolution (the actual "backtester"; not
    reimplemented here).
  * ``family_backtest.select_signals`` — per-bar top-K selection.
  * ``portfolio_backtest.AdmittedTrade`` / ``BookSnapshot`` / ``compute_metrics``
    — trade/snapshot record types and the metrics table, so every rule's
    output is a drop-in for the existing metrics function.
  * ``portfolio_backtest.shares_for_notional_min1`` — whole-share rounding,
    byte-identical convention to the live fixed-notional baseline.
  * ``feature_matrix_4h._asof_join_regime`` / ``_load_daily_regime_table`` —
    the causal (``available_at``-keyed, backward as-of) regime join. Not
    reimplemented; this module only calls it.
  * ``ablation.folds.build_walk_forward_folds`` / ``train_test_masks`` — the
    repo's one walk-forward fold spec.
  * ``ablation.bootstrap.week_block_bootstrap_ci`` / ``bh_fdr`` — statistical
    machinery.

New in this module (small, and only because none of the existing sizing
policies in ``portfolio_backtest.PolicyConfig`` are regime-conditional):
  * a thin admission/sizing loop parameterized by a ``Rule`` (admit_fn +
    size_fn) instead of ``PolicyConfig``'s fixed/equal/cov branches —
    mirrors ``portfolio_backtest.run_policy``'s book/flush/dedupe pattern
    exactly (same "already held" skip, same exit-freed-by-own-exit_ts
    flush), just driven by a callable instead of a dataclass switch;
  * a variable-stop candidate resolver for H4, which calls the SAME
    ``_simulate_signal`` per row, just with a per-row ``sl_atr_mult`` drawn
    from the regime state at that row's own ``signal_ts``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from strategies.momentum_expansion.backtest.family_backtest import (
    BarCache, _simulate_signal, select_signals,
)
from strategies.momentum_expansion.config.momentum_config import MODELS_DIR
from strategies.momentum_expansion.features.feature_matrix_4h import (
    _asof_join_regime, _load_daily_regime_table,
)

from research.portfolio_lab import portfolio_backtest as pb
from strategies.momentum_expansion.ablation.bootstrap import week_block_bootstrap_ci, week_of

OOF_PREDS_PATH = MODELS_DIR / "oof_preds.parquet"

REGIME_COLS = ["risk_appetite_z", "liquidity_stress_z", "sector_dispersion_z", "spy_rv20_z", "spy_trend_state"]

BASE_TOP_K = 10
WIDE_TOP_K = 15          # superset used only by H5-breadth
BASE_TP_MULT = 5.0
BASE_SL_MULT = 5.0
BASE_MAX_HOLD = 75
BASE_NOTIONAL = 5_000.0
COST_BPS_ROUND_TRIP = 10.0
CAPITAL_REF = 100_000.0  # reporting denominator only (fixed-notional sizing is capital-independent)


# ---------------------------------------------------------------------------
# Signal stream + causal regime join
# ---------------------------------------------------------------------------

def load_signal_stream(*, oof_path=OOF_PREDS_PATH) -> pd.DataFrame:
    """The 'existing signal stream': real leak-free walk-forward OOF scores
    of the deployed momentum model, 2022-11-14 .. 2026-05-14. Not trained
    here (D6)."""
    df = pd.read_parquet(oof_path, columns=["score"]).reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ticker"] = df["ticker"].astype(str)
    return df[["timestamp", "ticker", "score"]]


def attach_regime(df: pd.DataFrame, *, ts_col: str = "signal_ts") -> pd.DataFrame:
    """Causal as-of join of REGIME_COLS onto ``df[ts_col]``. Reuses
    ``_asof_join_regime`` unchanged: each row sees only the last regime row
    whose ``available_at <= df[ts_col]``."""
    table = _load_daily_regime_table()
    joined = _asof_join_regime(pd.DatetimeIndex(df[ts_col]), table, REGIME_COLS)
    out = df.copy()
    for c in REGIME_COLS:
        out[c] = joined[c].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Candidate resolution (reuses _simulate_signal unchanged)
# ---------------------------------------------------------------------------

def resolve_fixed_stop_candidates(
    signals: pd.DataFrame, bar_cache: BarCache, *, top_k: int,
    tp_mult: float = BASE_TP_MULT, sl_mult: float = BASE_SL_MULT, max_hold: int = BASE_MAX_HOLD,
) -> pd.DataFrame:
    """Top-K selection + fixed TP/SL/hold resolution. Byte-identical
    resolution logic to ``portfolio_backtest._resolve_candidates`` (reused,
    not reimplemented) — this wrapper only adds the top_k selection step and
    a per-bar rank column so H5-breadth can admit ranks 11-15 conditionally
    from the SAME resolved stream the baseline top-10 draws from."""
    sig = select_signals(signals[["timestamp", "ticker", "score"]], top_k, allow_short=False)
    candidates = pb._resolve_candidates(sig, bar_cache, tp_mult=tp_mult, sl_mult=sl_mult, max_hold=max_hold)
    if candidates.empty:
        return candidates
    candidates["rank_in_bar"] = candidates.groupby("signal_ts")["score"].rank(ascending=False, method="first")
    return candidates


def resolve_variable_sl_candidates(
    signals_with_regime: pd.DataFrame, bar_cache: BarCache, *,
    sl_mult_fn: Callable[[pd.Series], float], tp_mult: float = BASE_TP_MULT, max_hold: int = BASE_MAX_HOLD,
) -> pd.DataFrame:
    """H4: same top-10 selection, same entry/TP logic, but ``sl_atr_mult`` is
    drawn per-row from the row's OWN regime state (already attached, causal)
    before calling the unchanged ``_simulate_signal``. This changes trade
    RESOLUTION (the exit), so it cannot reuse a single shared candidate
    stream the way the admission/sizing rules do."""
    rows = []
    for tkr, grp in signals_with_regime.groupby("ticker"):
        bars = bar_cache.get(tkr)
        if bars is None or len(bars["ts"]) < 20:
            continue
        for r in grp.itertuples():
            sl_mult = float(sl_mult_fn(r))
            res = _simulate_signal(bars, r.timestamp, 1, tp_mult, sl_mult, max_hold)
            if res is None:
                continue
            res.update(ticker=tkr, direction=1, score=r.score, signal_ts=r.timestamp, sl_mult_used=sl_mult)
            rows.append(res)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ("entry_ts", "exit_ts", "signal_ts"):
        out[col] = pd.to_datetime(out[col], utc=True)
    out = out.sort_values(["signal_ts", "score"], ascending=[True, False]).reset_index(drop=True)
    out["rank_in_bar"] = out.groupby("signal_ts")["score"].rank(ascending=False, method="first")
    return out


# ---------------------------------------------------------------------------
# Rule application: admission + sizing overlay (mirrors run_policy's
# book/flush/dedupe pattern; parameterized by callables instead of
# PolicyConfig branches, since none of the existing branches are
# regime-conditional)
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    rule_id: str
    group: str  # H1..H5
    admit: Callable[[pd.Series], bool] = lambda row: True
    size_mult: Callable[[pd.Series], float] = lambda row: 1.0


def apply_rule(
    candidates: pd.DataFrame, rule: Rule, *,
    base_notional: float = BASE_NOTIONAL, cost_bps: float = COST_BPS_ROUND_TRIP,
) -> tuple[list[pb.AdmittedTrade], list[pb.BookSnapshot]]:
    """Event-driven admission over ``candidates`` (must be sorted by
    (signal_ts, score desc) and carry the REGIME_COLS + rank_in_bar columns).
    Unconstrained concurrency (no slot cap) -- matches the fixed-notional
    baseline policy (a) this study holds as "current policy"; only WHICH
    signals are admitted and how big each is differs by rule."""
    book: dict[str, dict] = {}       # ticker -> {"exit_ts", "dollar_size"}
    trades: list[pb.AdmittedTrade] = []
    snapshots: list[pb.BookSnapshot] = []

    for signal_ts, grp in candidates.groupby("signal_ts", sort=True):
        for tkr in list(book):
            if book[tkr]["exit_ts"] <= signal_ts:
                del book[tkr]

        for row in grp.itertuples():
            ticker = row.ticker
            if ticker in book:
                continue  # already held -- same dedupe as run_policy
            if not rule.admit(row):
                continue
            mult = float(rule.size_mult(row))
            if mult <= 0:
                continue
            price = float(row.entry_price)
            notional = base_notional * mult
            shares = pb.shares_for_notional_min1(price, notional)
            dollar_size = shares * price
            pnl_pct = float(row.pnl_pct)
            pnl_gross = pnl_pct * dollar_size
            cost = (cost_bps / 1e4) * dollar_size * (2.0 + pnl_pct)
            pnl_net = pnl_gross - cost

            book[ticker] = {"exit_ts": row.exit_ts, "dollar_size": dollar_size}
            trades.append(pb.AdmittedTrade(
                ticker=ticker, signal_ts=signal_ts, entry_ts=row.entry_ts, exit_ts=row.exit_ts,
                entry_price=price, exit_price=float(row.exit_price), shares=shares, dollar_size=dollar_size,
                pnl_pct=pnl_pct, pnl_dollar_gross=pnl_gross, pnl_dollar_net=pnl_net, cost_dollar=cost,
                exit_reason=str(row.exit_reason), bars_held=int(row.bars_held), adv_dollars=None,
                sector=None, theme=None,
            ))

        weights = [p["dollar_size"] / CAPITAL_REF for p in book.values()]
        snapshots.append(pb.BookSnapshot(
            ts=signal_ts, n_open=len(book), gross_dollar=sum(p["dollar_size"] for p in book.values()),
            herfindahl=float(sum(w * w for w in weights)), max_weight=max(weights) if weights else 0.0,
        ))
    return trades, snapshots


def baseline_rule() -> Rule:
    return Rule(rule_id="baseline", group="baseline", admit=lambda r: r.rank_in_bar <= BASE_TOP_K)


# ---------------------------------------------------------------------------
# Statistic: week-block bootstrap on weekly net-$ P&L difference (rule minus
# baseline), reusing week_block_bootstrap_ci unchanged. One statistic for
# every rule type -- see HYPOTHESES.md "Statistic and FDR family".
# ---------------------------------------------------------------------------

def _weekly_pnl(trades: list[pb.AdmittedTrade]) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    t = pd.DataFrame([{"exit_ts": tr.exit_ts, "pnl": tr.pnl_dollar_net} for tr in trades])
    t["week"] = week_of(t["exit_ts"])
    return t.groupby("week")["pnl"].sum()


def weekly_diff_bootstrap(
    baseline_trades: list[pb.AdmittedTrade], rule_trades: list[pb.AdmittedTrade],
    *, n_boot: int = 1000, seed: int = 42,
) -> dict:
    """Week-block bootstrap CI + two-sided p-value on the mean weekly net-$
    P&L difference (rule - baseline), using the union of weeks either side
    traded in (missing weeks filled 0). Reuses ``week_block_bootstrap_ci``
    for the point/CI/n_weeks by treating each week's own diff as a single
    observation keyed by itself (a legitimate week-block bootstrap of the
    weekly-diff series)."""
    base_w = _weekly_pnl(baseline_trades)
    rule_w = _weekly_pnl(rule_trades)
    weeks = base_w.index.union(rule_w.index)
    diff = rule_w.reindex(weeks, fill_value=0.0) - base_w.reindex(weeks, fill_value=0.0)
    if diff.empty:
        return {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_weeks": 0,
                "excludes_zero": False, "p_value": np.nan}
    boot = week_block_bootstrap_ci(pd.Series(diff.to_numpy()), pd.Series(diff.index.to_numpy()),
                                   n_boot=n_boot, seed=seed)
    p_value = np.nan
    if boot["n_weeks"] >= 2:
        rng = np.random.default_rng(seed + 1)
        week_keys = diff.index.to_numpy()
        vals = {w: np.array([diff.loc[w]]) for w in week_keys}
        boots = np.empty(n_boot)
        for b in range(n_boot):
            draw = rng.choice(week_keys, size=len(week_keys), replace=True)
            boots[b] = np.mean(np.concatenate([vals[w] for w in draw]))
        p_ge0, p_le0 = float(np.mean(boots >= 0)), float(np.mean(boots <= 0))
        p_value = float(min(1.0, 2 * min(p_ge0, p_le0)))
    return {**boot, "p_value": p_value}
