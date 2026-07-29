"""Portfolio-level 3-way sizing comparison, built on top of the existing
``strategies/momentum_expansion/backtest/family_backtest.py`` signal + exit
engine (per the WS-D task spec: reuse the existing family backtest engine,
do not write a new one).

``family_backtest`` resolves every ``(timestamp, ticker)`` top-K signal into
an independent, FIXED-NOTIONAL trade (entry = next 4H bar open, exit = ATR
take-profit / stop-loss / time stop) under an UNCONSTRAINED-capital
assumption -- by design; its own docstring calls this "sizing-neutral signal
quality". This module does not re-derive signals and does not re-implement
the TP/SL/time-stop exit math: it imports ``select_signals`` (same top-K
selection) and ``_simulate_signal`` (same entry/exit resolution) unchanged,
then adds the one piece that engine deliberately does not have -- a
capital-constrained, event-driven PORTFOLIO overlay so three sizing policies
can be compared on the *same admitted candidate stream*:

  (a) fixed_notional     current live baseline: every admitted entry gets a
                          flat ``target_notional`` (mirrors
                          ``core.live_4h_exec.ExecPolicy.target_notional`` /
                          ``shares_for_notional``, reimplemented locally so
                          this research module does not import the live
                          execution path). No capital ceiling -- matches live,
                          which sizes off price alone, not total exposure.
  (b) equal_weight_topN  capital / target_concurrent per name, capped at
                          ``target_concurrent`` concurrent slots. No
                          covariance, no caps beyond the slot count.
  (c) cov_vol_target      Ledoit-Wolf shrunk-covariance annualized-vol
                          targeting + correlation-aware down-weight of
                          near-duplicate names + per-position/sector/theme/
                          ADV caps (research.portfolio_lab.sizing), also
                          capped at ``target_concurrent`` slots.

All three share: the same signal stream (one scored model's top-K per
timestamp), the same entry price / ATR exit resolution, the same
"skip if already held" dedupe (mirrors ``build_mixed_plan``'s
``already_held`` skip in ``core/live_4h_exec.py``), and the same per-trade
round-trip cost assumption. Only the DOLLAR SIZE of each admitted entry
differs -- isolating the effect of sizing from the effect of signal quality.

Causality: sizing decisions at signal timestamp ``ts`` use only daily bars
dated strictly before ``ts``'s own session (``DailyContext``/``covariance``
enforce this; see their module docstrings and
``tests/test_portfolio_backtest.py::test_admission_is_causal``). Position
capital is freed only once a trade's own (already-resolved) ``exit_ts`` has
passed -- never based on future information about OTHER trades.

Sector caps are effectively inert in the real-data comparison run: plan
defect D1 (see docs/superpowers/plans/2026-07-26-market-regime-and-sector-
context.md) documents that 100% of tickers currently resolve to
``sector == "Unknown"`` in the curated metadata, and the empirical
sector-ETF resolver is WS-A's deliverable, out of scope here. This module
accepts an injectable ``sector_map`` so the cap mechanism is real and tested
(``tests/test_portfolio_backtest.py``), but the real run passes ``None``
sectors throughout and reports that explicitly rather than fabricating a
mapping.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategies.momentum_expansion.backtest.family_backtest import (
    BarCache, _simulate_signal, select_signals,
)

from research.portfolio_lab import sizing as sz
from research.portfolio_lab.daily_context import DailyContext

logger = logging.getLogger(__name__)

DEFAULT_COV_WINDOW = 120
DEFAULT_COV_MIN_PERIODS = 60
DEFAULT_ADV_WINDOW = 20
DEFAULT_ADV_MIN_PERIODS = 5


def shares_for_notional_min1(price: float | None, target_notional: float) -> int:
    """Whole shares nearest ``target_notional`` at ``price``, floored at 1.

    Deliberately mirrors ``core.live_4h_exec.shares_for_notional`` byte-for-
    byte (see that module's docstring) so policy (a) is a faithful stand-in
    for the live baseline; reimplemented locally rather than imported so this
    research module has no dependency on the live execution path.
    """
    if not price or price <= 0:
        return 1
    return max(1, round(target_notional / price))


@dataclass
class PolicyConfig:
    name: str
    fixed_notional: float | None = None          # policy (a) only
    equal_weight: bool = False                    # policy (b): skip cov/caps, just cap/N
    sizing_cfg: sz.SizingConfig | None = None      # policy (c)
    target_concurrent: int | None = None           # slot cap; None = unconstrained (policy a)
    cov_window: int = DEFAULT_COV_WINDOW
    cov_min_periods: int = DEFAULT_COV_MIN_PERIODS
    adv_window: int = DEFAULT_ADV_WINDOW
    adv_min_periods: int = DEFAULT_ADV_MIN_PERIODS


@dataclass
class AdmittedTrade:
    ticker: str
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    dollar_size: float
    pnl_pct: float
    pnl_dollar_gross: float
    pnl_dollar_net: float
    cost_dollar: float
    exit_reason: str
    bars_held: int
    adv_dollars: float | None
    sector: str | None
    theme: str | None


@dataclass
class BookSnapshot:
    ts: pd.Timestamp
    n_open: int
    gross_dollar: float
    herfindahl: float
    max_weight: float


def _resolve_candidates(
    signals: pd.DataFrame, bar_cache: BarCache, *, tp_mult: float, sl_mult: float, max_hold: int,
) -> pd.DataFrame:
    """Resolve every (timestamp, ticker) top-K row to its entry/exit outcome
    using the UNCHANGED family_backtest exit machinery. This is the shared
    candidate stream every policy draws from; only admission + sizing differ
    downstream."""
    rows = []
    for tkr, grp in signals.groupby("ticker"):
        bars = bar_cache.get(tkr)
        if bars is None or len(bars["ts"]) < 20:
            continue
        for r in grp.itertuples():
            res = _simulate_signal(bars, r.timestamp, r.direction, tp_mult, sl_mult, max_hold)
            if res is None:
                continue
            res.update(ticker=tkr, direction=r.direction, score=r.score, signal_ts=r.timestamp)
            rows.append(res)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # bars["ts_dt"] (entry_ts/exit_ts) is tz-naive-but-UTC numpy datetime64
    # while signal_ts carries the scored frame's tz-aware UTC Timestamp;
    # normalize all three to tz-aware UTC so downstream comparisons
    # (_flush's exit_ts <= signal_ts) never mix naive and aware datetimes.
    for col in ("entry_ts", "exit_ts", "signal_ts"):
        out[col] = pd.to_datetime(out[col], utc=True)
    return out.sort_values(["signal_ts", "score"], ascending=[True, False]).reset_index(drop=True)


def _herfindahl(weights: list[float]) -> float:
    if not weights:
        return 0.0
    return float(sum(w * w for w in weights))


def run_policy(
    candidates: pd.DataFrame, *, policy: PolicyConfig, daily_ctx: DailyContext,
    sector_map: dict | None = None, theme_map: dict | None = None,
    cost_bps_round_trip: float = 10.0,
) -> tuple[list[AdmittedTrade], list[BookSnapshot]]:
    """Event-driven admission + sizing over the shared candidate stream.

    ``candidates`` must already be sorted by (signal_ts, score desc) --
    ``_resolve_candidates`` does this. Processes one signal_ts group at a
    time: first frees any positions whose OWN exit_ts has already passed
    (<= this signal_ts -- a trade's exit is determined at admission time and
    never depends on later signals), then evaluates each candidate in score
    order, admitting/sizing/rejecting against the book as it stands *at that
    moment* (so caps bind within a batch, not just across batches).
    """
    sector_map = sector_map or {}
    theme_map = theme_map or {}
    book: dict[str, sz.OpenPosition] = {}
    open_exit: dict[str, tuple[pd.Timestamp, dict]] = {}  # ticker -> (exit_ts, resolved_row)
    trades: list[AdmittedTrade] = []
    snapshots: list[BookSnapshot] = []
    capital = policy.sizing_cfg.capital if policy.sizing_cfg else 100_000.0

    def _flush(cutoff: pd.Timestamp) -> None:
        for tkr in list(open_exit):
            exit_ts, _row = open_exit[tkr]
            if exit_ts <= cutoff:
                book.pop(tkr, None)
                open_exit.pop(tkr, None)

    for signal_ts, grp in candidates.groupby("signal_ts", sort=True):
        _flush(signal_ts)

        slot_cap = policy.target_concurrent
        for row in grp.itertuples():
            ticker = row.ticker
            if ticker in book:
                continue  # already held -- mirrors live already_held skip
            if slot_cap is not None and len(book) >= slot_cap:
                continue

            price = float(row.entry_price)
            sector = sector_map.get(ticker)
            theme = theme_map.get(ticker)
            adv = daily_ctx.trailing_adv(
                ticker, asof=signal_ts, window=policy.adv_window, min_periods=policy.adv_min_periods,
            )

            if policy.fixed_notional is not None:
                shares = shares_for_notional_min1(price, policy.fixed_notional)
                dollar_size = shares * price
            elif policy.equal_weight:
                base = capital / max(1, policy.target_concurrent or 1)
                shares = sz.whole_shares(base, price)
                dollar_size = shares * price
            else:
                cfg = policy.sizing_cfg
                relevant = list(book.keys()) + [ticker]
                rets = daily_ctx.trailing_returns_subset(
                    relevant, asof=signal_ts, window=policy.cov_window, min_periods=policy.cov_min_periods,
                )
                cov_result = None
                if rets is not None:
                    from research.portfolio_lab import covariance as cov_mod
                    cov_result = cov_mod.ledoit_wolf_shrunk_cov(rets, asof=signal_ts)
                cov = cov_result.cov if cov_result is not None else None
                corr = cov_result.corr if cov_result is not None else None

                book_weights = sz.book_weights(book, capital)
                vol_scale = sz.portfolio_vol_scale(
                    book_weights, cov, target_vol=cfg.target_portfolio_vol_annual, max_scale=cfg.max_vol_scale,
                )
                base_dollar = (capital / max(1, cfg.target_concurrent)) * vol_scale
                result = sz.size_candidate(
                    ticker, price, cfg=cfg, book=book, sector=sector, theme=theme,
                    corr=corr, adv_dollars=adv, base_dollar_override=base_dollar,
                )
                shares, dollar_size = result.shares, result.dollar_size

            if shares <= 0 or dollar_size <= 0:
                continue

            pnl_pct = float(row.pnl_pct)
            pnl_gross = pnl_pct * dollar_size
            cost = (cost_bps_round_trip / 1e4) * dollar_size * (2.0 + pnl_pct)
            pnl_net = pnl_gross - cost

            book[ticker] = sz.OpenPosition(ticker=ticker, dollar_size=dollar_size, shares=shares,
                                           sector=sector, theme=theme)
            open_exit[ticker] = (row.exit_ts, {})
            trades.append(AdmittedTrade(
                ticker=ticker, signal_ts=signal_ts, entry_ts=row.entry_ts, exit_ts=row.exit_ts,
                entry_price=price, exit_price=float(row.exit_price), shares=shares, dollar_size=dollar_size,
                pnl_pct=pnl_pct, pnl_dollar_gross=pnl_gross, pnl_dollar_net=pnl_net, cost_dollar=cost,
                exit_reason=str(row.exit_reason), bars_held=int(row.bars_held), adv_dollars=adv,
                sector=sector, theme=theme,
            ))

        weights = list(sz.book_weights(book, capital).values())
        snapshots.append(BookSnapshot(
            ts=signal_ts, n_open=len(book), gross_dollar=sum(p.dollar_size for p in book.values()),
            herfindahl=_herfindahl(weights), max_weight=max(weights) if weights else 0.0,
        ))

    return trades, snapshots


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    trades: list[AdmittedTrade], snapshots: list[BookSnapshot], *,
    capital: float, n_candidates_seen: int,
) -> dict:
    if not trades:
        return {"trades": 0, "candidates_seen": n_candidates_seen, "fill_rate": 0.0}

    t = pd.DataFrame([{
        "exit_ts": tr.exit_ts, "pnl_pct": tr.pnl_pct, "pnl_dollar_net": tr.pnl_dollar_net,
        "pnl_dollar_gross": tr.pnl_dollar_gross, "cost_dollar": tr.cost_dollar,
        "dollar_size": tr.dollar_size, "adv_dollars": tr.adv_dollars,
    } for tr in trades]).sort_values("exit_ts").reset_index(drop=True)

    eq = capital + t["pnl_dollar_net"].cumsum()
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    total_pnl = float(t["pnl_dollar_net"].sum())
    total_ret = total_pnl / capital

    r = t["pnl_pct"].to_numpy()
    sharpe_per_trade = float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else float("nan")

    daily = t.set_index("exit_ts")["pnl_dollar_net"].resample("1D").sum()
    daily_ret = daily / capital
    n_trading_days = int((daily_ret != 0).sum()) or len(daily_ret)
    sharpe_daily = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else float("nan")
    )

    span_days = max(1, (t["exit_ts"].max() - t["exit_ts"].min()).days)
    turnover_notional = float(t["dollar_size"].sum())
    turnover_annualized = (turnover_notional / capital) / (span_days / 365.0)

    herf = [s.herfindahl for s in snapshots if s.n_open > 0]
    maxw = [s.max_weight for s in snapshots if s.n_open > 0]
    n_open = [s.n_open for s in snapshots]

    adv_part = (t["dollar_size"] / t["adv_dollars"]).replace([np.inf, -np.inf], np.nan).dropna()

    pos = t.loc[t.pnl_dollar_net > 0, "pnl_dollar_net"].sum()
    neg = -t.loc[t.pnl_dollar_net <= 0, "pnl_dollar_net"].sum()

    return {
        "trades": int(len(t)),
        "candidates_seen": int(n_candidates_seen),
        "fill_rate": round(len(t) / n_candidates_seen, 4) if n_candidates_seen else float("nan"),
        "win_rate": round(float((t.pnl_dollar_net > 0).mean()), 4),
        "profit_factor": round(float(pos / neg), 3) if neg > 0 else float("inf"),
        "avg_trade_pct": round(float(r.mean()) * 100, 4),
        "total_pnl_net": round(total_pnl, 2),
        "total_return_pct_net": round(total_ret * 100, 3),
        "max_dd_pct": round(float(dd) * 100, 3),
        "ret_over_dd": round(float(total_ret / abs(dd)), 3) if dd < 0 else float("inf"),
        "sharpe_per_trade": round(sharpe_per_trade, 3),
        "sharpe_daily": round(sharpe_daily, 3),
        "n_trading_days_with_flow": n_trading_days,
        "total_cost_dollar": round(float(t["cost_dollar"].sum()), 2),
        "cost_pct_of_gross_pnl": (
            round(float(t["cost_dollar"].sum() / t["pnl_dollar_gross"].sum()) * 100, 2)
            if t["pnl_dollar_gross"].sum() != 0 else float("nan")
        ),
        "turnover_notional_total": round(turnover_notional, 0),
        "turnover_annualized_x": round(turnover_annualized, 3),
        "avg_concurrent_positions": round(float(np.mean(n_open)), 2) if n_open else 0.0,
        "max_concurrent_positions": int(max(n_open)) if n_open else 0,
        "avg_herfindahl": round(float(np.mean(herf)), 4) if herf else float("nan"),
        "max_herfindahl": round(float(np.max(herf)), 4) if herf else float("nan"),
        "avg_max_single_weight": round(float(np.mean(maxw)), 4) if maxw else float("nan"),
        "avg_position_pct_of_adv": (
            round(float(adv_part.mean()) * 100, 3) if not adv_part.empty else float("nan")
        ),
        "p95_position_pct_of_adv": (
            round(float(adv_part.quantile(0.95)) * 100, 3) if not adv_part.empty else float("nan")
        ),
        "avg_bars_held": round(float(np.mean([tr.bars_held for tr in trades])), 2),
    }
