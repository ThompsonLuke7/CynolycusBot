"""
MomentumOptionPolicy — options execution layer for momentum_expansion.

Standalone class. Does NOT touch Policy/order_policy.py (the SPY pivot
policy). Reuses `API.Alpaca_API.options.options_api.AlpacaOptionsClient`
for chain queries + order submission.

Per-trade flow:
  1. Caller passes (ticker, direction, suggested_stop_atr, current_atr,
     current_underlying_price).
  2. Chain liquidity gate — query option snapshots for the target DTE band,
     filter by OI / volume / spread, target_delta. Reject if no contract
     passes.
  3. IV regime gate — reject if proxy IV-rank is too high.
  4. Sizing — risk per trade is per_trade_risk_pct of sleeve_dollars;
     contracts = floor(risk_$ / (atr_stop * 100 * delta)) bounded by
     max_contracts_cap and min_per_trade_dollars.
  5. Submit (or skip if submit_orders=False).

Per-position management:
  - track entry underlying price + ATR for trailing
  - exit if 4H expansion score decays below score_decay_exit
  - exit if underlying breaks ema_slow by trend_break_atr × ATR
  - hard time stop after max_holding_4h_bars
"""
from __future__ import annotations

import logging
import math
import time as time_mod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

try:
    from API.Alpaca_API.options.options_api import AlpacaOptionsClient
except Exception:
    AlpacaOptionsClient = None  # type: ignore

from momentum_expansion.config.momentum_config import (
    CAMPAIGN_CONFIG,
    CAPITAL_CONFIG,
    OPTION_POLICY_CONFIG,
)

logger = logging.getLogger(__name__)


@dataclass
class MomentumOptionConfig:
    sleeve_dollars: float = CAPITAL_CONFIG["sleeve_dollars"]
    max_concurrent: int = CAPITAL_CONFIG["max_concurrent"]
    per_trade_risk_pct: float = CAPITAL_CONFIG["per_trade_risk_pct"]
    min_per_trade_dollars: float = CAPITAL_CONFIG["min_per_trade_dollars"]

    target_dte_min_days: int = OPTION_POLICY_CONFIG["target_dte_min_days"]
    target_dte_max_days: int = OPTION_POLICY_CONFIG["target_dte_max_days"]
    target_delta_long: float = OPTION_POLICY_CONFIG["target_delta_long"]
    delta_tolerance: float = OPTION_POLICY_CONFIG["delta_tolerance"]
    adaptive_delta_enabled: bool = OPTION_POLICY_CONFIG["adaptive_delta_enabled"]
    adaptive_delta_high_score: float = OPTION_POLICY_CONFIG["adaptive_delta_high_score"]
    adaptive_delta_mid_score: float = OPTION_POLICY_CONFIG["adaptive_delta_mid_score"]
    adaptive_delta_high: float = OPTION_POLICY_CONFIG["adaptive_delta_high"]
    adaptive_delta_mid: float = OPTION_POLICY_CONFIG["adaptive_delta_mid"]
    adaptive_delta_low: float = OPTION_POLICY_CONFIG["adaptive_delta_low"]
    adaptive_delta_late_campaign_entry: int = OPTION_POLICY_CONFIG["adaptive_delta_late_campaign_entry"]

    min_open_interest: int = OPTION_POLICY_CONFIG["min_open_interest"]
    min_chain_volume: int = OPTION_POLICY_CONFIG["min_chain_volume"]
    max_bid_ask_spread_pct: float = OPTION_POLICY_CONFIG["max_bid_ask_spread_pct"]

    price_mode: str = OPTION_POLICY_CONFIG["price_mode"]
    max_contracts_cap: int = OPTION_POLICY_CONFIG["max_contracts_cap"]

    max_ivr: float = OPTION_POLICY_CONFIG["max_ivr"]

    atr_stop_mult: float = OPTION_POLICY_CONFIG["atr_stop_mult"]
    atr_trail_arm: float = OPTION_POLICY_CONFIG["atr_trail_arm"]
    atr_trail_distance: float = OPTION_POLICY_CONFIG["atr_trail_distance"]
    score_decay_exit: float = OPTION_POLICY_CONFIG["score_decay_exit"]
    trend_break_atr: float | None = OPTION_POLICY_CONFIG["trend_break_atr"]
    max_holding_4h_bars: int = OPTION_POLICY_CONFIG["max_holding_4h_bars"]

    campaign_enabled: bool = CAMPAIGN_CONFIG["enabled"]
    campaign_reset_gap_days: int = CAMPAIGN_CONFIG["reset_gap_days"]
    campaign_max_entries: int = CAMPAIGN_CONFIG["max_entries"]
    campaign_entry_1_min_score: float = CAMPAIGN_CONFIG["entry_1_min_score"]
    campaign_entry_2_min_score: float = CAMPAIGN_CONFIG["entry_2_min_score"]
    campaign_entry_3_min_score: float = CAMPAIGN_CONFIG["entry_3_min_score"]
    campaign_entry_4_min_score: float = CAMPAIGN_CONFIG["entry_4_min_score"]
    campaign_entry_3_min_cumulative_return: float = CAMPAIGN_CONFIG["entry_3_min_cumulative_return"]

    submit_orders: bool = False
    paper: bool = True


@dataclass
class ContractCandidate:
    occ_symbol: str
    underlying: str
    side: str            # "call" | "put"
    expiration: str      # YYYY-MM-DD
    strike: float
    delta: float
    iv: float
    bid: float
    ask: float
    open_interest: int
    volume: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if (self.bid > 0 and self.ask > 0) else 0.0

    @property
    def spread_pct(self) -> float:
        if self.mid <= 0:
            return float("inf")
        return (self.ask - self.bid) / self.mid


@dataclass
class MomentumOptionPosition:
    underlying:        str
    direction:         int          # +1 long_call, -1 long_put
    contract:          ContractCandidate
    qty:               int
    entry_underlying:  float
    entry_atr:         float
    entry_score:       float
    entry_time:        pd.Timestamp
    initial_stop:      float        # underlying price level
    trail_armed:       bool = False
    trail_high:        float = 0.0  # highest favorable underlying since entry
    bars_held:         int = 0
    extra:             dict = field(default_factory=dict)


@dataclass
class MomentumCampaignState:
    entries: int = 0
    profitable_legs: int = 0
    cumulative_return: float = 0.0
    last_exit_time: pd.Timestamp | None = None


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------

def _normalize_snapshot_entry(occ_symbol: str, snap: dict) -> ContractCandidate | None:
    try:
        greeks = snap.get("greeks") or {}
        latest_quote = snap.get("latestQuote") or {}
        bid = float(latest_quote.get("bp") or latest_quote.get("bid_price") or 0.0)
        ask = float(latest_quote.get("ap") or latest_quote.get("ask_price") or 0.0)
        delta = float(greeks.get("delta") or 0.0)
        iv = float(snap.get("impliedVolatility") or 0.0)

        # Open interest / volume not always present on snapshot; allow 0.
        oi = int(snap.get("openInterest") or 0)
        vol = int(snap.get("dailyVolume") or 0)

        # Parse OCC symbol -> underlying / side / strike / expiration
        # Format: <UNDER><YYMMDD><C|P><STRIKE_x1000>
        if len(occ_symbol) < 15:
            return None
        # Strike is the last 8 chars
        strike_str = occ_symbol[-8:]
        side_char = occ_symbol[-9]
        date_str = occ_symbol[-15:-9]
        underlying = occ_symbol[:-15]
        try:
            strike = int(strike_str) / 1000.0
            exp = datetime.strptime(date_str, "%y%m%d").date().isoformat()
        except ValueError:
            return None
        side = "call" if side_char == "C" else "put"

        return ContractCandidate(
            occ_symbol=occ_symbol,
            underlying=underlying,
            side=side,
            expiration=exp,
            strike=strike,
            delta=delta,
            iv=iv,
            bid=bid,
            ask=ask,
            open_interest=oi,
            volume=vol,
        )
    except Exception as exc:
        logger.debug("snapshot parse failed for %s: %s", occ_symbol, exc)
        return None


def _select_contract(
    *,
    candidates: list[ContractCandidate],
    cfg: MomentumOptionConfig,
    direction: int,
    target_delta_abs: float | None = None,
) -> ContractCandidate | None:
    """Pick the contract with delta closest to the target_delta_long, with
    OI / volume / spread gates."""
    if not candidates:
        return None
    target_delta_abs = float(target_delta_abs or cfg.target_delta_long)
    target_delta = target_delta_abs if direction > 0 else -target_delta_abs
    pool = []
    for c in candidates:
        if direction > 0 and c.side != "call":
            continue
        if direction < 0 and c.side != "put":
            continue
        if c.open_interest < cfg.min_open_interest:
            continue
        if c.volume < cfg.min_chain_volume:
            continue
        if c.spread_pct > cfg.max_bid_ask_spread_pct:
            continue
        if abs(c.delta - target_delta) > cfg.delta_tolerance + 0.05:
            continue
        pool.append(c)
    if not pool:
        return None
    pool.sort(key=lambda x: abs(x.delta - target_delta))
    return pool[0]


# ---------------------------------------------------------------------------
# IV regime
# ---------------------------------------------------------------------------

def _ivr_proxy(history_iv: list[float], current_iv: float) -> float:
    """Approximate IV-rank using the history list. Falls back to 0.5 if empty."""
    if not history_iv:
        return 0.5
    lo, hi = min(history_iv), max(history_iv)
    if hi <= lo:
        return 0.5
    return float((current_iv - lo) / (hi - lo))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class MomentumOptionPolicy:
    def __init__(
        self,
        *,
        cfg: MomentumOptionConfig | None = None,
        client: Any = None,
    ):
        self.cfg = cfg or MomentumOptionConfig()
        if client is None and AlpacaOptionsClient is not None:
            try:
                client = AlpacaOptionsClient()
            except Exception as exc:
                logger.warning("AlpacaOptionsClient init failed: %s", exc)
                client = None
        self.client = client
        self.positions: dict[str, MomentumOptionPosition] = {}
        self.campaigns: dict[str, MomentumCampaignState] = {}

    # ---- campaign controls ----
    def _campaign_state_for(self, ticker: str, as_of: pd.Timestamp) -> MomentumCampaignState:
        state = self.campaigns.get(ticker)
        if state is None:
            state = MomentumCampaignState()
            self.campaigns[ticker] = state
            return state
        if state.last_exit_time is not None:
            gap_days = (as_of - state.last_exit_time).total_seconds() / 86400.0
            if gap_days >= float(self.cfg.campaign_reset_gap_days):
                state = MomentumCampaignState()
                self.campaigns[ticker] = state
        return state

    def _campaign_entry_allowed(
        self,
        *,
        ticker: str,
        expansion_score: float,
        as_of: pd.Timestamp,
    ) -> tuple[bool, str, MomentumCampaignState]:
        state = self._campaign_state_for(ticker, as_of)
        if not self.cfg.campaign_enabled:
            return True, "campaign_disabled", state

        next_entry = int(state.entries) + 1
        if next_entry > int(self.cfg.campaign_max_entries):
            return False, f"campaign_max_entries_{self.cfg.campaign_max_entries}", state

        min_score = float(self.cfg.campaign_entry_1_min_score)
        if next_entry == 2:
            min_score = float(self.cfg.campaign_entry_2_min_score)
        elif next_entry == 3:
            if float(state.cumulative_return) < float(self.cfg.campaign_entry_3_min_cumulative_return):
                return False, "campaign_entry3_needs_profit", state
            min_score = float(self.cfg.campaign_entry_3_min_score)
        elif next_entry >= 4:
            min_score = float(self.cfg.campaign_entry_4_min_score)

        if float(expansion_score) < min_score:
            return False, f"campaign_score_below_{min_score:.2f}", state
        return True, f"campaign_entry_{next_entry}", state

    def _target_delta_for_entry(
        self,
        *,
        expansion_score: float,
        campaign_entry: int,
    ) -> float:
        if not self.cfg.adaptive_delta_enabled:
            return float(self.cfg.target_delta_long)
        if int(campaign_entry) >= int(self.cfg.adaptive_delta_late_campaign_entry):
            return float(self.cfg.adaptive_delta_low)
        if float(expansion_score) >= float(self.cfg.adaptive_delta_high_score):
            return float(self.cfg.adaptive_delta_high)
        if float(expansion_score) >= float(self.cfg.adaptive_delta_mid_score):
            return float(self.cfg.adaptive_delta_mid)
        return float(self.cfg.adaptive_delta_low)

    def record_campaign_exit(
        self,
        *,
        ticker: str,
        exit_underlying: float,
        as_of: pd.Timestamp,
    ) -> None:
        pos = self.positions.get(ticker)
        if pos is None or pos.entry_underlying <= 0:
            return
        state = self._campaign_state_for(ticker, as_of)
        ret = pos.direction * (float(exit_underlying) / float(pos.entry_underlying) - 1.0)
        state.cumulative_return += float(ret)
        if ret > 0:
            state.profitable_legs += 1
        state.last_exit_time = as_of

    # ---- chain queries ----
    def _list_contract_candidates(
        self,
        *,
        underlying: str,
        as_of: date,
        direction: int,
    ) -> list[ContractCandidate]:
        if self.client is None:
            return []
        dte_min = as_of + timedelta(days=self.cfg.target_dte_min_days)
        dte_max = as_of + timedelta(days=self.cfg.target_dte_max_days)
        opt_type = "call" if direction > 0 else "put"
        try:
            snap = self.client.get_option_snapshots(
                underlying,
                expiration_date_gte=dte_min.isoformat(),
                expiration_date_lte=dte_max.isoformat(),
                type=opt_type,
            )
        except Exception as exc:
            logger.warning("[%s] chain fetch failed: %s", underlying, exc)
            return []

        out: list[ContractCandidate] = []
        for sym, body in (snap or {}).items():
            cand = _normalize_snapshot_entry(sym, body if isinstance(body, dict) else {})
            if cand is not None:
                out.append(cand)
        return out

    # ---- entry ----
    def consider_entry(
        self,
        *,
        ticker: str,
        direction: int,
        underlying_price: float,
        atr: float,
        expansion_score: float,
        suggested_stop_atr: float,
        as_of: pd.Timestamp,
        history_iv: list[float] | None = None,
    ) -> MomentumOptionPosition | None:
        if len(self.positions) >= self.cfg.max_concurrent:
            logger.info("[%s] max_concurrent reached — skip", ticker)
            return None
        if ticker in self.positions:
            logger.debug("[%s] already long — skip", ticker)
            return None
        if atr <= 0 or underlying_price <= 0:
            return None
        campaign_ok, campaign_reason, campaign_state = self._campaign_entry_allowed(
            ticker=ticker,
            expansion_score=expansion_score,
            as_of=as_of,
        )
        if not campaign_ok:
            logger.info("[%s] campaign gate rejected entry: %s", ticker, campaign_reason)
            return None
        campaign_entry = int(campaign_state.entries) + 1
        target_delta_abs = self._target_delta_for_entry(
            expansion_score=expansion_score,
            campaign_entry=campaign_entry,
        )

        # Chain selection
        candidates = self._list_contract_candidates(
            underlying=ticker, as_of=as_of.date(), direction=direction
        )
        contract = _select_contract(
            candidates=candidates,
            cfg=self.cfg,
            direction=direction,
            target_delta_abs=target_delta_abs,
        )
        if contract is None:
            logger.info("[%s] no qualifying contract — skip", ticker)
            return None

        # IV regime gate
        ivr = _ivr_proxy(history_iv or [], contract.iv)
        if ivr > self.cfg.max_ivr:
            logger.info("[%s] IVR=%.2f > %.2f — skip", ticker, ivr, self.cfg.max_ivr)
            return None

        # Sizing — risk = sleeve * per_trade_risk_pct, stop = atr_stop_mult × ATR
        atr_stop_distance = self.cfg.atr_stop_mult * suggested_stop_atr * atr
        risk_dollars = self.cfg.sleeve_dollars * self.cfg.per_trade_risk_pct
        # Approximate option $ loss per 1× ATR underlying drop = abs(delta) * atr_stop_distance * 100
        per_contract_risk = max(0.01, abs(contract.delta) * atr_stop_distance * 100.0)
        raw_qty = math.floor(risk_dollars / per_contract_risk)
        qty = max(0, min(raw_qty, self.cfg.max_contracts_cap))
        if qty == 0:
            logger.info("[%s] sizing -> 0 contracts (risk=%.0f / per=%.0f)", ticker, risk_dollars, per_contract_risk)
            return None
        # Min dollar floor
        if qty * contract.mid * 100 < self.cfg.min_per_trade_dollars:
            qty = max(qty, math.ceil(self.cfg.min_per_trade_dollars / max(contract.mid * 100, 1.0)))
            qty = min(qty, self.cfg.max_contracts_cap)
            if qty == 0:
                return None

        initial_stop = (
            underlying_price - direction * atr_stop_distance
        )

        position = MomentumOptionPosition(
            underlying=ticker,
            direction=direction,
            contract=contract,
            qty=qty,
            entry_underlying=underlying_price,
            entry_atr=atr,
            entry_score=expansion_score,
            entry_time=as_of,
            initial_stop=initial_stop,
            extra={
                "campaign_entry": campaign_entry,
                "campaign_reason": campaign_reason,
                "campaign_cumulative_return_before": float(campaign_state.cumulative_return),
                "target_delta_abs": float(target_delta_abs),
            },
        )
        self.positions[ticker] = position
        campaign_state.entries += 1

        if self.cfg.submit_orders and self.client is not None:
            try:
                self.client.submit_option_order(
                    symbol=contract.occ_symbol,
                    qty=qty,
                    side="buy",
                    order_type="market",
                    time_in_force="day",
                )
                logger.info("[%s] submitted BUY %d × %s", ticker, qty, contract.occ_symbol)
            except Exception as exc:
                logger.warning("[%s] order submit failed: %s", ticker, exc)
        else:
            logger.info("[%s] (paper) BUY %d × %s @ %.2f (delta=%.2f target=%.2f, IVR=%.2f)",
                        ticker, qty, contract.occ_symbol, contract.mid, contract.delta, target_delta_abs, ivr)
        return position

    # ---- per-bar management ----
    def update_and_check_exit(
        self,
        *,
        ticker: str,
        underlying_price: float,
        atr: float,
        ema_slow: float,
        expansion_score: float,
    ) -> tuple[bool, str]:
        """
        Apply trailing logic + exit predicates for an open position.
        Returns (exit_now, reason). Caller is responsible for actually
        submitting the close order.
        """
        pos = self.positions.get(ticker)
        if pos is None:
            return False, ""
        pos.bars_held += 1

        # Trail arm + advance
        favorable_move = (underlying_price - pos.entry_underlying) * pos.direction
        if not pos.trail_armed and favorable_move >= self.cfg.atr_trail_arm * pos.entry_atr:
            pos.trail_armed = True
            pos.trail_high = underlying_price
        if pos.trail_armed:
            if pos.direction > 0:
                pos.trail_high = max(pos.trail_high, underlying_price)
                trail_stop = pos.trail_high - self.cfg.atr_trail_distance * pos.entry_atr
                if underlying_price <= trail_stop:
                    return True, "trail_stop"
            else:
                pos.trail_high = min(pos.trail_high or underlying_price, underlying_price)
                trail_stop = pos.trail_high + self.cfg.atr_trail_distance * pos.entry_atr
                if underlying_price >= trail_stop:
                    return True, "trail_stop"

        # Hard initial stop
        if pos.direction > 0 and underlying_price <= pos.initial_stop:
            return True, "initial_stop"
        if pos.direction < 0 and underlying_price >= pos.initial_stop:
            return True, "initial_stop"

        # Trend break vs ema_slow
        if self.cfg.trend_break_atr is not None:
            if pos.direction > 0 and underlying_price < ema_slow - self.cfg.trend_break_atr * atr:
                return True, "trend_break"
            if pos.direction < 0 and underlying_price > ema_slow + self.cfg.trend_break_atr * atr:
                return True, "trend_break"

        # Score decay
        if expansion_score < self.cfg.score_decay_exit:
            return True, "score_decay"

        # Time stop
        if pos.bars_held >= self.cfg.max_holding_4h_bars:
            return True, "time_stop"

        return False, ""

    def close(self, ticker: str, reason: str = "manual") -> MomentumOptionPosition | None:
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return None
        if self.cfg.submit_orders and self.client is not None:
            try:
                self.client.submit_option_order(
                    symbol=pos.contract.occ_symbol,
                    qty=pos.qty,
                    side="sell",
                    order_type="market",
                    time_in_force="day",
                )
                logger.info("[%s] submitted SELL %d × %s (%s)",
                            ticker, pos.qty, pos.contract.occ_symbol, reason)
            except Exception as exc:
                logger.warning("[%s] close failed: %s", ticker, exc)
        else:
            logger.info("[%s] (paper) CLOSE %d × %s reason=%s",
                        ticker, pos.qty, pos.contract.occ_symbol, reason)
        return pos
