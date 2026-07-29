"""Lot-based position simulator with OPT-IN pyramiding (scaling INTO a held
position). See ``HYPOTHESES.md`` for the pre-registered conventions this file
implements.

Why a new simulator instead of extending an existing one
--------------------------------------------------------
Three engines already resolve 4H signals into trades, and none of them can
add notional to an open position:

* ``family_backtest._simulate_signal`` — one signal -> exactly one ATR
  TP/SL/time trade. Left BYTE-UNCHANGED by this study; it is imported here
  only so ``tests/test_engine.py`` can assert it still behaves identically,
  and ``portfolio_backtest`` / ``regime_policy`` keep calling it as before.
* ``portfolio_backtest.run_policy`` — skips a candidate whose ticker is
  already in the book.
* ``backtest_exits.simulate`` / ``exit_policy_cross_module.simulate`` — the
  percent-based engine that actually mirrors ``core.live_4h_exec.ExecPolicy``
  (stop / take-profit-trim / horizon / grace / trail, all as fractions of the
  entry price). This is the engine the live exit policy was selected with
  (``research/capstone/exit_policy_cross_module.csv``), so it is the one a
  pyramiding delta must be measured against.

This module reproduces that third engine EXACTLY when pyramiding is off
(``PyramidPolicy.trigger == "none"`` and ``cost_bps == 0``; asserted by
``tests/test_engine.py::test_pyramid_off_matches_baseline_engine``), and adds
lot-level accounting so extra notional can be layered on. It does not modify
any existing engine.

Accounting model
----------------
A position is a list of lots ``[shares, cost_price]``. Shares are CONTINUOUS
(no whole-share rounding) because the baseline percent engine it must
reproduce is continuous; ``portfolio_backtest.shares_for_notional_min1``
rounds, but parity with the baseline wins here (HYPOTHESES.md).

Two per-bar arrays are accumulated so capital deployment can be controlled for
(the whole point of the study):

* ``pnl_by_bar[j]`` — net dollar P&L attributed to bar ``j`` as
  ``(end-of-bar holdings value + sell cash - buy cash - fees) - (start-of-bar
  holdings value)``. Summed across tickers and resampled daily this is a true
  mark-to-market P&L series, so Sharpe is not an artifact of exit-date
  bucketing.
* ``deployed_by_bar[j]`` — cost basis of open lots at the END of bar ``j``.
  Its time-average is the denominator of "return per dollar deployed".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasePolicy:
    """The live 4H exit policy, as fractions of the reference (entry) price.

    Values mirror ``core.live_4h_exec.ExecPolicy`` defaults; ``max_hold``
    mirrors ``signals.meta_context.meta_ranker.backtest_exits.MAX_HOLD``, the
    outer scan cap the baseline engine applies on top of ``horizon_bars``.
    """
    take_profit: float | None = 0.30
    scale_frac: float = 0.16
    horizon_bars: int | None = 53
    stop_loss: float | None = 0.39
    trail_stop: float | None = None
    grace_bars: int | None = None
    target_notional: float = 5_000.0
    max_hold: int = 60


@dataclass(frozen=True)
class PyramidPolicy:
    """How (and whether) to add to an already-open position.

    trigger:
      ``"none"``     — no adds (baseline).
      ``"level"``    — add k fires the first time the intrabar gain from the
                       REFERENCE price reaches ``k * level`` (pyramid ladder).
      ``"reselect"`` — add k fires at the k-th later bar at which the ticker is
                       in the module's top-K, subject to ``spacing_bars`` since
                       entry or the previous add.

    add_frac: add notional as a fraction of the INITIAL entry notional.
    max_adds: hard cap on adds per position.
    basis:    ``"entry"``   — stop / take-profit / trail are keyed to the
                              ORIGINAL entry price (pre-registered primary).
              ``"blended"`` — keyed to the share-weighted average cost of open
                              lots (secondary sensitivity; this one DOES change
                              exit timing).
    """
    trigger: str = "none"
    level: float | None = None
    add_frac: float = 0.0
    max_adds: int = 0
    spacing_bars: int = 6
    basis: str = "entry"

    def __post_init__(self) -> None:
        if self.trigger not in {"none", "level", "reselect"}:
            raise ValueError(f"unknown pyramid trigger: {self.trigger!r}")
        if self.basis not in {"entry", "blended"}:
            raise ValueError(f"unknown pyramid basis: {self.basis!r}")
        if self.trigger == "level" and not self.level:
            raise ValueError("trigger='level' requires a positive level")
        if self.trigger != "none" and (self.max_adds <= 0 or self.add_frac <= 0):
            raise ValueError("a non-'none' trigger needs max_adds>0 and add_frac>0")


NO_PYRAMID = PyramidPolicy()


@dataclass
class Position:
    ticker: str
    entry_i: int
    exit_i: int
    entry_price: float
    exit_price: float
    exit_reason: str
    bars_held: int
    n_adds: int
    initial_notional: float
    added_notional: float
    peak_cost_basis: float
    fill_notional: float          # total bought+sold notional (turnover numerator)
    pnl_gross: float
    fees: float
    pnl_net: float
    ret_on_initial: float         # pnl_gross / initial_notional -- comparable to the
                                  # baseline percent engine's per-trade return


def _weighted_cost(lots: list[list[float]], shares_total: float) -> float:
    if shares_total <= 0:
        return 0.0
    return sum(s * p for s, p in lots) / shares_total


def _sell_all(lots: list[list[float]], price: float) -> tuple[float, float]:
    """Close every lot at ``price``. Returns (cash_in, realized_gross)."""
    cash = sum(s * price for s, _ in lots)
    realized = sum(s * (price - p) for s, p in lots)
    lots.clear()
    return cash, realized


def _sell_fraction(lots: list[list[float]], frac: float, price: float) -> tuple[float, float, float]:
    """Sell ``frac`` of every lot pro-rata at ``price``.

    Pro-rata (rather than FIFO/LIFO) is pre-registered: it keeps the trim a
    pure size reduction and leaves the remaining position's average cost
    unchanged, so the trim never silently interacts with which lots exist.
    Returns (shares_sold, cash_in, realized_gross).
    """
    sold = cash = realized = 0.0
    for lot in lots:
        q = lot[0] * frac
        sold += q
        cash += q * price
        realized += q * (price - lot[1])
        lot[0] -= q
    return sold, cash, realized


def simulate_ticker(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, member: np.ndarray,
    *, ticker: str = "", base: BasePolicy = BasePolicy(), pyr: PyramidPolicy = NO_PYRAMID,
    notional: float | None = None, cost_bps: float = 10.0,
    pnl_by_bar: np.ndarray | None = None, deployed_by_bar: np.ndarray | None = None,
) -> list[Position]:
    """Walk one ticker's 4H bars, opening a position whenever it is in the
    top-K and no position is open, and managing it with ``base`` (+ ``pyr``).

    Entry = the CLOSE of the bar at which top-K membership is observed, exactly
    as the baseline engine does. Scanning resumes at ``exit_bar + 1``, so
    chained RE-ENTRY after a full exit is preserved (baseline behaviour, not a
    treatment). ``pnl_by_bar`` / ``deployed_by_bar``, when supplied, are
    accumulated in place (length == len(close)).

    Causality: every decision at bar ``j`` reads only ``close[:j+1]``,
    ``high[:j+1]``, ``low[:j+1]``, ``member[:j+1]``. Nothing after ``j`` is
    consulted, and fills land at bar ``j``'s own prices.
    """
    n = len(close)
    notional = base.target_notional if notional is None else notional
    fee_rate = cost_bps / 1e4
    out: list[Position] = []
    if n < 2:
        return out

    # only bars where the name is in the top-K can start a position
    starts = np.flatnonzero(member[: n - 1])
    next_free = 0

    for i in starts:
        i = int(i)
        if i < next_free:
            continue
        entry = float(close[i])
        if not np.isfinite(entry) or entry <= 0:
            continue

        lots: list[list[float]] = [[notional / entry, entry]]
        shares_total = notional / entry
        cost_basis = notional
        peak_basis = notional
        added_notional = 0.0
        fill_notional = notional
        fees = fee_rate * notional
        realized_gross = 0.0
        n_adds = 0
        last_add_bar = i
        trimmed = False
        peak_px = entry
        bars_out = 0

        if pnl_by_bar is not None:
            pnl_by_bar[i] -= fees
        if deployed_by_bar is not None:
            deployed_by_bar[i] += cost_basis

        prev_close = entry
        prev_shares = shares_total
        exit_price: float | None = None
        exit_reason = ""
        j = i + 1

        while j < n and (j - i) <= base.max_hold:
            hi, lo, cl = float(high[j]), float(low[j]), float(close[j])
            peak_px = max(peak_px, hi)
            ref = entry if pyr.basis == "entry" else _weighted_cost(lots, shares_total)
            cash_in = cash_out = bar_fees = 0.0

            # --- 1) hard stop (full exit; pre-empts any add on this bar) ---
            if base.stop_loss is not None and (lo / ref - 1.0) <= -base.stop_loss:
                px = ref * (1.0 - base.stop_loss)
                c, r = _sell_all(lots, px)
                cash_in += c
                realized_gross += r
                bar_fees += fee_rate * c
                fill_notional += c
                shares_total = 0.0
                exit_price, exit_reason = px, "stop"

            # --- 2) trailing stop (full exit) ---
            elif base.trail_stop is not None and lo <= peak_px * (1.0 - base.trail_stop):
                px = peak_px * (1.0 - base.trail_stop)
                c, r = _sell_all(lots, px)
                cash_in += c
                realized_gross += r
                bar_fees += fee_rate * c
                fill_notional += c
                shares_total = 0.0
                exit_price, exit_reason = px, "trail"

            if exit_price is None:
                # --- 3) take-profit trim (partial), BEFORE any add on this bar ---
                if (base.take_profit is not None and not trimmed
                        and (hi / ref - 1.0) >= base.take_profit):
                    if base.scale_frac >= 1.0:
                        px = ref * (1.0 + base.take_profit)
                        c, r = _sell_all(lots, px)
                        cash_in += c
                        realized_gross += r
                        bar_fees += fee_rate * c
                        fill_notional += c
                        shares_total = 0.0
                        exit_price, exit_reason = px, "take_profit"
                    else:
                        px = ref * (1.0 + base.take_profit)
                        q, c, r = _sell_fraction(lots, base.scale_frac, px)
                        cash_in += c
                        realized_gross += r
                        bar_fees += fee_rate * c
                        fill_notional += c
                        shares_total -= q
                        trimmed = True

            if exit_price is None:
                # --- 4) rank drop-out backstop (off by default) ---
                bars_out = bars_out + 1 if not member[j] else 0
                if base.grace_bars is not None and bars_out > base.grace_bars:
                    c, r = _sell_all(lots, cl)
                    cash_in += c
                    realized_gross += r
                    bar_fees += fee_rate * c
                    fill_notional += c
                    shares_total = 0.0
                    exit_price, exit_reason = cl, "dropped_out"

            if exit_price is None:
                # --- 5) horizon hard cap ---
                if base.horizon_bars is not None and (j - i) >= base.horizon_bars:
                    c, r = _sell_all(lots, cl)
                    cash_in += c
                    realized_gross += r
                    bar_fees += fee_rate * c
                    fill_notional += c
                    shares_total = 0.0
                    exit_price, exit_reason = cl, "horizon"

            if exit_price is None and pyr.trigger != "none" and n_adds < pyr.max_adds:
                # --- 6) ADD -- evaluated LAST, so an add can only happen on a
                # bar the position SURVIVES. Every exit check above therefore
                # pre-empts the add, and no lot is ever opened on the exit bar
                # (which would burn a round-trip fee for zero exposure). The
                # trim at step 3 still executes BEFORE the add on a surviving
                # bar, so the trim always sells the pre-add share count.
                fire = False
                if pyr.trigger == "level":
                    fire = (hi / entry - 1.0) >= (n_adds + 1) * float(pyr.level)
                elif pyr.trigger == "reselect":
                    fire = bool(member[j]) and (j - last_add_bar) >= pyr.spacing_bars
                if fire and cl > 0:
                    add_dollars = pyr.add_frac * notional
                    q = add_dollars / cl
                    lots.append([q, cl])
                    shares_total += q
                    cash_out += add_dollars
                    bar_fees += fee_rate * add_dollars
                    fill_notional += add_dollars
                    added_notional += add_dollars
                    n_adds += 1
                    last_add_bar = j

            cost_basis = sum(s * p for s, p in lots)
            peak_basis = max(peak_basis, cost_basis)
            fees += bar_fees
            if pnl_by_bar is not None:
                pnl_by_bar[j] += (shares_total * cl + cash_in - cash_out
                                  - prev_shares * prev_close - bar_fees)
            if deployed_by_bar is not None:
                deployed_by_bar[j] += cost_basis

            if exit_price is not None:
                break
            prev_close, prev_shares = cl, shares_total
            j += 1

        if exit_price is None:
            # Ran off the end of the series (j == n) or past MAX_HOLD: mark out
            # at the last available close, exactly as the baseline engine does
            # (``jj = min(j, n - 1)``; ``holds = jj - i``).
            jj = min(j, n - 1)
            cl = float(close[jj])
            c, r = _sell_all(lots, cl)
            realized_gross += r
            bar_fees = fee_rate * c
            fees += bar_fees
            fill_notional += c
            if pnl_by_bar is not None:
                if jj == j:
                    # bar j was never marked inside the loop (MAX_HOLD cut it off)
                    pnl_by_bar[jj] += c - prev_shares * prev_close - bar_fees
                else:
                    # jj == j-1 was already marked to close[jj]; only the exit
                    # fee is new (the sale realizes exactly the marked value).
                    pnl_by_bar[jj] -= bar_fees
            shares_total = 0.0
            exit_price, exit_reason = cl, "max_hold"
            j = jj

        out.append(Position(
            ticker=ticker, entry_i=i, exit_i=j, entry_price=entry, exit_price=float(exit_price),
            exit_reason=exit_reason, bars_held=j - i, n_adds=n_adds,
            initial_notional=notional, added_notional=added_notional, peak_cost_basis=peak_basis,
            fill_notional=fill_notional, pnl_gross=realized_gross, fees=fees,
            pnl_net=realized_gross - fees, ret_on_initial=realized_gross / notional,
        ))
        next_free = j + 1

    return out
