"""
Walk-forward backtest for momentum_expansion.

Pipeline at each 4H bar:
  1. Read the active universe snapshot (snapshot dated <= bar — no future
     constituents).
  2. Score every name in the snapshot via the trained ranker.
  3. Take top-N. For each top-N name, evaluate 1H entry rules on the most
     recent 1H bars up to the 4H bar's timestamp.
  4. Open synthetic positions through MomentumOptionPolicy with
     submit_orders=False (paper mode).
  5. On every subsequent 4H bar: re-score, run exit predicates, mark-to-market.

P&L modes:
  - underlying_only: exit price - entry price * direction * notional shares.
  - synthetic_options: Black-Scholes pricer (synthetic_options.py) with an
    IV proxy (VIX-scaled by beta or constant).

Output: backtest/results/run_<timestamp>/{trades.csv, equity_curve.csv,
metrics.json}.
"""
from __future__ import annotations

import json
import logging
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.backtest.synthetic_options import bs_price, iv_from_vix
from strategies.momentum_expansion.config.momentum_config import (
    BACKTEST_CONFIG,
    BACKTEST_RESULTS_DIR,
    CAPITAL_CONFIG,
    FEATURES_COMBINED,
    OPTION_POLICY_CONFIG,
    RANKING_CONFIG,
)
from strategies.momentum_expansion.data.load_bars import load_1h, load_4h
from strategies.momentum_expansion.data.universe import load_snapshot_for
from strategies.momentum_expansion.inference.entry_rules import evaluate_entry
from strategies.momentum_expansion.inference.ranker import ExpansionRanker, top_n_at

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    ticker:        str
    entry_ts:      pd.Timestamp
    exit_ts:       pd.Timestamp
    direction:     int
    entry_price:   float
    exit_price:    float
    shares:        float
    contracts:     int
    entry_iv:      float
    exit_iv:       float
    entry_score:   float
    exit_reason:   str
    pnl_underlying: float
    pnl_option:    float
    bars_held:     int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, lo, c_p = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - lo), (h - c_p).abs(), (lo - c_p).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def _slice_to(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.loc[df.index <= ts]


# ---------------------------------------------------------------------------
# Position record (lighter than MomentumOptionPosition for backtest)
# ---------------------------------------------------------------------------

@dataclass
class BTPosition:
    ticker:        str
    direction:     int
    entry_ts:      pd.Timestamp
    entry_price:   float
    entry_atr:     float
    entry_score:   float
    initial_stop:  float
    trail_armed:   bool = False
    trail_high:    float = 0.0
    bars_held:     int = 0
    bars_out:      int = 0      # consecutive bars out of the top-N (simple_hold grace)
    trimmed:       bool = False # already scaled out half at take-profit (simple_hold)
    contracts:     int = 0
    entry_iv:      float = 0.30
    strike:        float = 0.0
    expiration_ts: pd.Timestamp = None
    entry_option_price: float = 0.0


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class MomentumBacktester:
    # simple_hold (shared live 4H engine) exit parameters.
    SIMPLE_TAKE_PROFIT = 0.20   # trim at +20% on the OPTION mark
    SIMPLE_SCALE_FRAC = 0.50    # sell this fraction at take-profit, ride the rest
    SIMPLE_HORIZON_BARS = 25    # full exit of the remainder after this many 4H bars
    SIMPLE_GRACE_BARS = 3       # full exit after this many bars out of the top-N

    def __init__(
        self,
        *,
        ranker: ExpansionRanker,
        cfg: dict | None = None,
        ranking_cfg: dict | None = None,
        unconstrained: bool = False,
        exit_mode: str = "momentum",
    ):
        """
        unconstrained=True: signal-evaluation mode — no capital cap, no concurrent limit,
        no ATR-based sizing. Enters every trigger that passes ranker+rules. P&L tracked
        as underlying % return on a fixed $1,000 notional per trade so results are
        interpretable without sizing noise.
        """
        self.ranker = ranker
        self.cfg = {**BACKTEST_CONFIG, **(cfg or {})}
        self.ranking_cfg = {**RANKING_CONFIG, **(ranking_cfg or {})}
        self.option_cfg = OPTION_POLICY_CONFIG
        self.capital_cfg = CAPITAL_CONFIG
        self.unconstrained = unconstrained
        if exit_mode not in ("momentum", "simple_hold"):
            raise ValueError(f"exit_mode must be 'momentum' or 'simple_hold', got {exit_mode!r}")
        self.exit_mode = exit_mode

        self.positions: dict[str, BTPosition] = {}
        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[dict] = []
        self.equity = float(self.cfg["initial_capital"])

        # Cached per-ticker frames
        self._4h_cache: dict[str, pd.DataFrame] = {}
        self._1h_cache: dict[str, pd.DataFrame] = {}

    # ---- data access (cached) ----
    def _get_4h(self, ticker: str) -> pd.DataFrame | None:
        if ticker not in self._4h_cache:
            try:
                self._4h_cache[ticker] = load_4h(ticker)
            except FileNotFoundError:
                self._4h_cache[ticker] = None
        return self._4h_cache[ticker]

    def _get_1h(self, ticker: str) -> pd.DataFrame | None:
        if ticker not in self._1h_cache:
            try:
                self._1h_cache[ticker] = load_1h(ticker)
            except FileNotFoundError:
                self._1h_cache[ticker] = None
        return self._1h_cache[ticker]

    # ---- option utilities ----
    def _entry_iv(self, vix_close: float | None, beta: float) -> float:
        mode = self.cfg["synthetic_iv_proxy"]
        if mode == "constant_30":
            return float(self.cfg["constant_iv"])
        if mode == "vix_scaled" and vix_close is not None:
            return iv_from_vix(float(vix_close), ticker_beta=beta)
        return float(self.cfg["constant_iv"])

    def _option_price(
        self,
        *,
        spot: float,
        strike: float,
        bars_to_expiry: int,
        iv: float,
        is_call: bool,
    ) -> tuple[float, float]:
        # Convert bars to fraction of year (2 4H bars per RTH day, ~252 days/yr)
        t_years = max(1e-4, bars_to_expiry / (2 * 252))
        return bs_price(spot=spot, strike=strike, t_years=t_years, iv=iv, is_call=is_call)

    # ---- entry / exit ----
    def _open_position(
        self,
        *,
        ticker: str,
        direction: int,
        bar_ts: pd.Timestamp,
        underlying: float,
        atr: float,
        score: float,
        suggested_stop_atr: float,
        vix_close: float | None,
        beta: float,
    ) -> BTPosition | None:
        if ticker in self.positions:
            return None

        atr_stop_distance = self.option_cfg["atr_stop_mult"] * suggested_stop_atr * atr

        if self.unconstrained:
            # Signal-evaluation mode: no concurrent cap, fixed 1-contract sizing.
            # P&L is tracked on underlying % returns so results aren't contaminated
            # by position-sizing choices.
            contracts = 1
            strike = round(underlying * (0.985 if direction > 0 else 1.015), 2)
            target_dte_days = (
                self.option_cfg["target_dte_min_days"] + self.option_cfg["target_dte_max_days"]
            ) // 2
            expiration_ts = bar_ts + pd.Timedelta(days=target_dte_days)
            entry_iv = self._entry_iv(vix_close, beta)
            entry_option_price, _ = self._option_price(
                spot=underlying, strike=strike,
                bars_to_expiry=target_dte_days * 2,
                iv=entry_iv, is_call=(direction > 0),
            )
        else:
            if len(self.positions) >= self.capital_cfg["max_concurrent"]:
                return None
            # Risk-based contract sizing
            risk_dollars = self.capital_cfg["sleeve_dollars"] * self.capital_cfg["per_trade_risk_pct"]
            delta = self.option_cfg["target_delta_long"]
            per_contract_risk = max(1.0, delta * atr_stop_distance * 100.0)
            contracts = max(0, int(risk_dollars // per_contract_risk))
            contracts = min(contracts, self.option_cfg["max_contracts_cap"])
            if contracts == 0:
                return None
            if direction > 0:
                strike = round(underlying * 0.985, 2)
            else:
                strike = round(underlying * 1.015, 2)
            target_dte_days = (
                self.option_cfg["target_dte_min_days"] + self.option_cfg["target_dte_max_days"]
            ) // 2
            expiration_ts = bar_ts + pd.Timedelta(days=target_dte_days)
            entry_iv = self._entry_iv(vix_close, beta)
            entry_option_price, _ = self._option_price(
                spot=underlying, strike=strike,
                bars_to_expiry=target_dte_days * 2,
                iv=entry_iv, is_call=(direction > 0),
            )
            if entry_option_price <= 0:
                return None

        pos = BTPosition(
            ticker=ticker,
            direction=direction,
            entry_ts=bar_ts,
            entry_price=underlying,
            entry_atr=atr,
            entry_score=score,
            initial_stop=underlying - direction * atr_stop_distance,
            contracts=contracts,
            entry_iv=entry_iv,
            strike=strike,
            expiration_ts=expiration_ts,
            entry_option_price=entry_option_price if entry_option_price > 0 else 0.01,
        )
        self.positions[ticker] = pos
        return pos

    def _check_exit(
        self,
        *,
        pos: BTPosition,
        underlying: float,
        atr: float,
        ema_slow: float,
        score: float,
    ) -> str | None:
        pos.bars_held += 1
        favorable = (underlying - pos.entry_price) * pos.direction
        if not pos.trail_armed and favorable >= self.option_cfg["atr_trail_arm"] * pos.entry_atr:
            pos.trail_armed = True
            pos.trail_high = underlying
        if pos.trail_armed:
            if pos.direction > 0:
                pos.trail_high = max(pos.trail_high, underlying)
                stop = pos.trail_high - self.option_cfg["atr_trail_distance"] * pos.entry_atr
                if underlying <= stop:
                    return "trail_stop"
            else:
                pos.trail_high = min(pos.trail_high or underlying, underlying)
                stop = pos.trail_high + self.option_cfg["atr_trail_distance"] * pos.entry_atr
                if underlying >= stop:
                    return "trail_stop"
        if pos.direction > 0 and underlying <= pos.initial_stop:
            return "initial_stop"
        if pos.direction < 0 and underlying >= pos.initial_stop:
            return "initial_stop"
        trend_break_atr = self.option_cfg.get("trend_break_atr")
        if trend_break_atr is not None:
            if pos.direction > 0 and underlying < ema_slow - trend_break_atr * atr:
                return "trend_break"
            if pos.direction < 0 and underlying > ema_slow + trend_break_atr * atr:
                return "trend_break"
        if score < self.option_cfg["score_decay_exit"]:
            return "score_decay"
        if pos.bars_held >= self.option_cfg["max_holding_4h_bars"]:
            return "time_stop"
        return None

    def _current_option_price(self, pos, underlying, bar_ts, vix_close) -> float | None:
        try:
            bars_to_expiry = max(1, int((pos.expiration_ts - bar_ts).total_seconds() / (4 * 3600)))
            iv = self._entry_iv(vix_close, beta=1.0)
            price, _ = self._option_price(
                spot=underlying, strike=pos.strike, bars_to_expiry=bars_to_expiry,
                iv=iv, is_call=(pos.direction > 0),
            )
            return price
        except Exception:
            return None

    def _check_exit_simple(self, *, pos, underlying, bar_ts, vix_close, in_topk) -> str | None:
        """Shared live 4H engine exit: TP +20% option -> trim half -> ride the rest
        to a 25-bar horizon / 3-bar drop-out grace. No trailing / stop / trend exits."""
        pos.bars_held += 1
        pos.bars_out = 0 if in_topk else pos.bars_out + 1
        if pos.bars_held >= self.SIMPLE_HORIZON_BARS:
            return "horizon"
        if pos.bars_out > self.SIMPLE_GRACE_BARS:
            return "grace"
        if not pos.trimmed:
            opt = self._current_option_price(pos, underlying, bar_ts, vix_close)
            if opt is not None and pos.entry_option_price > 0 \
                    and (opt / pos.entry_option_price - 1.0) >= self.SIMPLE_TAKE_PROFIT:
                return "trim"
        return None

    def _close_position(
        self,
        *,
        pos: BTPosition,
        underlying: float,
        bar_ts: pd.Timestamp,
        reason: str,
        vix_close: float | None,
        notional_frac: float = 1.0,
    ) -> BacktestTrade:
        # Underlying P&L — in unconstrained mode, use a fixed $1,000 notional per trade
        # so equity-curve is a meaningful signal-quality measure rather than a sizing
        # artifact. `notional_frac` < 1 closes only part of the position (scale-out).
        if self.unconstrained:
            notional_dollars = 1_000.0 * notional_frac
            shares_eq = notional_dollars / max(pos.entry_price, 0.01)
        else:
            shares_eq = pos.contracts * 100.0 * notional_frac
        pnl_underlying = (underlying - pos.entry_price) * pos.direction * shares_eq

        # Synthetic option P&L
        bars_to_expiry = max(1, int((pos.expiration_ts - bar_ts).total_seconds() / (4 * 3600)))
        exit_iv = self._entry_iv(vix_close, beta=1.0)
        exit_option_price, _ = self._option_price(
            spot=underlying, strike=pos.strike,
            bars_to_expiry=bars_to_expiry, iv=exit_iv,
            is_call=(pos.direction > 0),
        )
        contracts_frac = pos.contracts * notional_frac
        pnl_option = (exit_option_price - pos.entry_option_price) * contracts_frac * 100.0
        # Apply commissions on entry+exit
        notional_opt = pos.entry_option_price * contracts_frac * 100.0
        commission = self.cfg["commission_pct"] * (notional_opt + exit_option_price * contracts_frac * 100.0)
        pnl_option -= commission

        trade = BacktestTrade(
            ticker=pos.ticker,
            entry_ts=pos.entry_ts,
            exit_ts=bar_ts,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=underlying,
            shares=shares_eq,
            contracts=pos.contracts,
            entry_iv=pos.entry_iv,
            exit_iv=exit_iv,
            entry_score=pos.entry_score,
            exit_reason=reason,
            pnl_underlying=pnl_underlying,
            pnl_option=pnl_option,
            bars_held=pos.bars_held,
        )
        return trade

    # ---- main loop ----
    def run(
        self,
        *,
        features: pd.DataFrame,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> dict:
        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("features must have (timestamp, ticker) MultiIndex")
        ts_level = features.index.names[0]
        all_ts = features.index.get_level_values(ts_level).unique().sort_values()
        mask = (all_ts >= pd.Timestamp(start)) & (all_ts <= pd.Timestamp(end))
        bar_timestamps = all_ts[mask]

        vix_4h = self._get_4h("VIXY")  # context VIX proxy
        spy_4h = self._get_4h("SPY")

        for bi, bar_ts in enumerate(bar_timestamps):
            vix_close = None
            if vix_4h is not None and len(vix_4h) > 0:
                idx = vix_4h.index.searchsorted(bar_ts, side="right") - 1
                if 0 <= idx < len(vix_4h):
                    vix_close = float(vix_4h["close"].iloc[idx])

            # Rank once per bar: used for entries and (simple_hold) drop-out grace.
            ranked = top_n_at(
                bar_ts=bar_ts, features=features,
                ranker=self.ranker, cfg=self.ranking_cfg,
            )
            topk_set = set(ranked["ticker"]) if not ranked.empty else set()

            # ---- exit checks first (so freed slots can be re-entered)
            for ticker in list(self.positions.keys()):
                df_4h = self._get_4h(ticker)
                if df_4h is None:
                    continue
                idx = df_4h.index.searchsorted(bar_ts, side="right") - 1
                if idx < 0:
                    continue
                row = df_4h.iloc[idx]
                close = float(row["close"])
                ema_slow = float(_ema(df_4h["close"], 20).iloc[idx])
                atr = float(_atr(df_4h, 14).iloc[idx])
                if not np.isfinite(atr) or atr <= 0:
                    continue

                # Score cached on row's feature vector — use the ranker
                feat_idx = (bar_ts, ticker)
                if feat_idx in features.index:
                    feat_row = features.loc[feat_idx:feat_idx]
                    if len(feat_row) > 0:
                        score = float(self.ranker.score(feat_row).iloc[0])
                    else:
                        score = self.positions[ticker].entry_score
                else:
                    score = self.positions[ticker].entry_score

                if self.exit_mode == "simple_hold":
                    pos = self.positions[ticker]
                    reason = self._check_exit_simple(
                        pos=pos, underlying=close, bar_ts=bar_ts,
                        vix_close=vix_close, in_topk=(ticker in topk_set),
                    )
                    if reason == "trim":
                        # Sell half at +20% option; keep riding the remainder.
                        trade = self._close_position(
                            pos=pos, underlying=close, bar_ts=bar_ts, reason="tp_trim",
                            vix_close=vix_close, notional_frac=self.SIMPLE_SCALE_FRAC,
                        )
                        pos.trimmed = True
                        self.trades.append(trade)
                        self.equity += trade.pnl_underlying if self.unconstrained else trade.pnl_option
                    elif reason is not None:  # horizon / grace -> close the remainder
                        pos = self.positions.pop(ticker)
                        frac = (1.0 - self.SIMPLE_SCALE_FRAC) if pos.trimmed else 1.0
                        trade = self._close_position(
                            pos=pos, underlying=close, bar_ts=bar_ts, reason=reason,
                            vix_close=vix_close, notional_frac=frac,
                        )
                        self.trades.append(trade)
                        self.equity += trade.pnl_underlying if self.unconstrained else trade.pnl_option
                    continue

                reason = self._check_exit(
                    pos=self.positions[ticker],
                    underlying=close, atr=atr, ema_slow=ema_slow, score=score,
                )
                if reason is not None:
                    pos = self.positions.pop(ticker)
                    trade = self._close_position(
                        pos=pos, underlying=close, bar_ts=bar_ts,
                        reason=reason, vix_close=vix_close,
                    )
                    self.trades.append(trade)
                    # In unconstrained mode equity tracks signal quality via underlying P&L
                    self.equity += trade.pnl_underlying if self.unconstrained else trade.pnl_option

            # ---- entry path (ranked computed at top of bar)
            for _, row in ranked.iterrows():
                ticker = row["ticker"]
                if ticker in self.positions:
                    continue
                df_1h = self._get_1h(ticker)
                if df_1h is None:
                    continue
                df_1h_view = _slice_to(df_1h, bar_ts)
                if df_1h_view is None or df_1h_view.empty:
                    continue
                triggers = evaluate_entry(df_1h=df_1h_view.tail(60))
                if not triggers:
                    continue
                trig = triggers[0]
                df_4h = self._get_4h(ticker)
                if df_4h is None:
                    continue
                idx = df_4h.index.searchsorted(bar_ts, side="right") - 1
                if idx < 0:
                    continue
                close = float(df_4h["close"].iloc[idx])
                atr = float(_atr(df_4h, 14).iloc[idx])
                if not np.isfinite(atr) or atr <= 0:
                    continue
                self._open_position(
                    ticker=ticker, direction=+1,
                    bar_ts=bar_ts, underlying=close,
                    atr=atr, score=float(row["expansion_score"]),
                    suggested_stop_atr=trig.suggested_stop_atr,
                    vix_close=vix_close, beta=1.0,
                )

            # ---- equity curve point
            self.equity_curve.append({
                "ts":      bar_ts,
                "equity":  self.equity,
                "open_n":  len(self.positions),
                "trades":  len(self.trades),
            })

            if (bi + 1) % 200 == 0:
                logger.info("(%d/%d) equity=%.2f open=%d trades=%d",
                            bi + 1, len(bar_timestamps),
                            self.equity, len(self.positions), len(self.trades))

        return self._summary()

    def _summary(self) -> dict:
        eq = pd.DataFrame(self.equity_curve)
        if eq.empty:
            return {"trades": 0, "metrics": {}, "unconstrained": self.unconstrained}
        eq["ret"] = eq["equity"].pct_change().fillna(0.0)
        cagr = self._cagr(eq)
        max_dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
        sharpe = float(eq["ret"].mean() / eq["ret"].std() * np.sqrt(2 * 252)) if eq["ret"].std() > 0 else float("nan")
        pnl_key = "pnl_underlying" if self.unconstrained else "pnl_option"
        pnls = [getattr(t, pnl_key) for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        winrate = len(wins) / len(pnls) if pnls else float("nan")
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        right_tail = float(np.percentile(pnls, 95)) if pnls else 0.0
        pct_returns = [(t.exit_price - t.entry_price) / t.entry_price * t.direction for t in self.trades]
        return {
            "trades": len(self.trades),
            "unconstrained": self.unconstrained,
            "pnl_basis": pnl_key,
            "metrics": {
                "cagr": float(cagr),
                "max_drawdown": float(max_dd),
                "sharpe": float(sharpe),
                "winrate": float(winrate),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "right_tail_p95": float(right_tail),
                "final_equity": float(self.equity),
                "avg_trade_pct_return": float(np.mean(pct_returns)) if pct_returns else 0.0,
                "median_trade_pct_return": float(np.median(pct_returns)) if pct_returns else 0.0,
            },
        }

    def _cagr(self, eq: pd.DataFrame) -> float:
        if len(eq) < 2:
            return float("nan")
        years = (eq["ts"].iloc[-1] - eq["ts"].iloc[0]).total_seconds() / (365.25 * 86400)
        if years <= 0:
            return float("nan")
        ratio = eq["equity"].iloc[-1] / eq["equity"].iloc[0]
        return float(ratio ** (1.0 / years) - 1.0)

    def write_results(self, *, run_dir: Path | None = None) -> Path:
        BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if run_dir is None:
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            run_dir = BACKTEST_RESULTS_DIR / f"run_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.trades:
            pd.DataFrame([t.__dict__ for t in self.trades]).to_csv(run_dir / "trades.csv", index=False)
        pd.DataFrame(self.equity_curve).to_csv(run_dir / "equity_curve.csv", index=False)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(self._summary(), f, indent=2, default=str)
        logger.info("Backtest results written to %s", run_dir)
        return run_dir
