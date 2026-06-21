from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from strategies.dealer_positioning.config import DealerPositioningConfig
from strategies.dealer_positioning.models import DealerSignal, SimTrade
from strategies.dealer_positioning.signals import PriceCandle


def _parse_iso(raw: str) -> datetime:
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _contract_expiry(contract: dict[str, Any]) -> datetime | None:
    raw = contract.get("expiration_date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _contract_symbol(contract: dict[str, Any]) -> str:
    return str(contract.get("symbol") or "").strip().upper()


def _is_standard_100_contract(contract: dict[str, Any]) -> bool:
    multiplier = contract.get("multiplier")
    size = contract.get("size")
    if multiplier is not None and _as_float(multiplier) not in {100.0, None}:
        return False
    if size is not None and _as_float(size) not in {100.0, None}:
        return False
    return True


def _extract_latest_quote_payload(payload: Any, symbol: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        if symbol in payload and isinstance(payload[symbol], dict):
            return payload[symbol]
        quotes = payload.get("quotes")
        if isinstance(quotes, dict) and symbol in quotes and isinstance(quotes[symbol], dict):
            return quotes[symbol]
    return {}


def _quote_price(quote: dict[str, Any], *, mode: str) -> float | None:
    bid = _as_float(quote.get("bid_price") or quote.get("bp"))
    ask = _as_float(quote.get("ask_price") or quote.get("ap"))
    last = _as_float(quote.get("last_price") or quote.get("last") or quote.get("price"))
    mid = ((bid + ask) / 2.0) if bid is not None and ask is not None else None
    modes = {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "last": last,
    }
    preferred = modes.get(mode)
    if preferred is not None and preferred > 0:
        return preferred
    for fallback in (ask, mid, bid, last):
        if fallback is not None and fallback > 0:
            return fallback
    return None


def _quote_metrics(quote: dict[str, Any], *, mode: str) -> dict[str, float | None]:
    bid = _as_float(quote.get("bid_price") or quote.get("bp"))
    ask = _as_float(quote.get("ask_price") or quote.get("ap"))
    last = _as_float(quote.get("last_price") or quote.get("last") or quote.get("price"))
    mid = ((bid + ask) / 2.0) if bid is not None and ask is not None else None
    spread_pct = ((ask - bid) / mid) if bid is not None and ask is not None and mid and mid > 0 else None
    return {
        "price": _quote_price(quote, mode=mode),
        "bid": bid,
        "ask": ask,
        "last": last,
        "mid": mid,
        "spread_pct": spread_pct,
    }


class DealerTradeExecutor:
    """Optional Alpaca paper/live order layer for dealer-positioning signals."""

    def __init__(self, config: DealerPositioningConfig) -> None:
        self._config = config
        self._open: dict[str, list[SimTrade]] = defaultdict(list)
        self._closed: list[SimTrade] = []
        self._counter = 0
        self._client = AlpacaOptionsClient(env_file=config.alpaca_env_file)

    @property
    def open_trades(self) -> list[SimTrade]:
        return [trade for trades in self._open.values() for trade in trades]

    @property
    def closed_trades(self) -> list[SimTrade]:
        return list(self._closed)

    def on_signal(self, signal: DealerSignal) -> SimTrade | None:
        open_for_symbol = [t for t in self._open[signal.symbol] if t.status == "open"]
        if len(open_for_symbol) >= self._config.max_trades_per_symbol:
            return None

        current_price = float(signal.spot)
        target_strike = float(signal.target if signal.target is not None else signal.level)
        direction = 1 if signal.direction == "long" else -1
        option_type = "call" if direction == 1 else "put"
        contract = self._select_contract(
            ticker=signal.symbol,
            option_type=option_type,
            target_strike=target_strike,
            current_price=current_price,
            ref_dt=_parse_iso(signal.timestamp),
        )
        if contract is None:
            return None

        contract_symbol = _contract_symbol(contract)
        expiration = str(contract.get("expiration_date") or "")
        contract_strike = _as_float(contract.get("strike_price"))
        entry_quote = self._latest_contract_quote(contract_symbol, mode="ask")
        entry_contract_price = entry_quote["price"]
        open_order_id = None
        if self._config.submit_orders:
            order_resp = self._client.submit_option_order(
                symbol=contract_symbol,
                qty=self._config.option_order_qty,
                side="buy",
                order_type=self._config.option_order_type,
                time_in_force=self._config.option_time_in_force,
                limit_price=round(entry_contract_price, 2) if self._config.option_order_type == "limit" and entry_contract_price else None,
            )
            open_order_id = str(order_resp.get("id") or "") if isinstance(order_resp, dict) else None

        self._counter += 1
        trade = SimTrade(
            trade_id=f"{signal.symbol}-{self._counter:06d}",
            opened_at=signal.timestamp,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            direction=signal.direction,
            entry=float(signal.spot),
            target=signal.target,
            stop=signal.stop,
            order_mode="alpaca" if self._config.submit_orders else "alpaca_simulated",
            contract_symbol=contract_symbol,
            contract_expiration=expiration,
            contract_strike=contract_strike,
            contract_side="C" if option_type == "call" else "P",
            entry_contract_price=entry_contract_price,
            entry_contract_bid=entry_quote["bid"],
            entry_contract_ask=entry_quote["ask"],
            entry_contract_spread_pct=entry_quote["spread_pct"],
            open_order_id=open_order_id,
        )
        self._open[signal.symbol].append(trade)
        return trade

    def on_candle(self, symbol: str, candle: PriceCandle) -> list[SimTrade]:
        symbol = symbol.upper()
        closed: list[SimTrade] = []
        still_open: list[SimTrade] = []
        for trade in self._open.get(symbol, []):
            exit_reason = None
            exit_underlying = None
            if trade.direction == "long":
                if trade.stop is not None and candle.low <= trade.stop:
                    exit_underlying, exit_reason = float(trade.stop), "stop"
                elif trade.target is not None and candle.high >= trade.target:
                    exit_underlying, exit_reason = float(trade.target), "target"
            else:
                if trade.stop is not None and candle.high >= trade.stop:
                    exit_underlying, exit_reason = float(trade.stop), "stop"
                elif trade.target is not None and candle.low <= trade.target:
                    exit_underlying, exit_reason = float(trade.target), "target"

            if exit_reason is None:
                still_open.append(trade)
                continue

            exit_quote = self._latest_contract_quote(str(trade.contract_symbol or ""), mode="bid")
            exit_contract_price = exit_quote["price"]
            close_order_id = None
            if self._config.submit_orders and trade.contract_symbol:
                order_resp = self._client.submit_option_order(
                    symbol=trade.contract_symbol,
                    qty=self._config.option_order_qty,
                    side="sell",
                    order_type=self._config.option_order_type,
                    time_in_force=self._config.option_time_in_force,
                    limit_price=round(exit_contract_price, 2) if self._config.option_order_type == "limit" and exit_contract_price else None,
                )
                close_order_id = str(order_resp.get("id") or "") if isinstance(order_resp, dict) else None

            trade.status = "closed"
            trade.closed_at = candle.timestamp.astimezone(timezone.utc).isoformat()
            trade.exit_price = exit_underlying
            trade.exit_reason = exit_reason
            trade.exit_contract_price = exit_contract_price
            trade.exit_contract_bid = exit_quote["bid"]
            trade.exit_contract_ask = exit_quote["ask"]
            trade.exit_contract_spread_pct = exit_quote["spread_pct"]
            trade.close_order_id = close_order_id
            mult = 1.0 if trade.direction == "long" else -1.0
            trade.pnl_points = ((exit_underlying or trade.entry) - trade.entry) * mult
            if trade.entry_contract_price is not None and exit_contract_price is not None:
                trade.pnl_dollars = (exit_contract_price - trade.entry_contract_price) * 100.0 * float(self._config.option_order_qty)
            closed.append(trade)
            self._closed.append(trade)

        self._open[symbol] = still_open
        return closed

    def finalize_expired(self, now: datetime, last_prices: dict[str, float]) -> list[SimTrade]:
        """Move expired contracts out of the open ledger using intrinsic value."""
        now_utc = now.astimezone(timezone.utc)
        closed: list[SimTrade] = []
        for symbol, trades in list(self._open.items()):
            still_open: list[SimTrade] = []
            spot = _as_float(last_prices.get(symbol))
            for trade in trades:
                try:
                    expiration = datetime.fromisoformat(str(trade.contract_expiration)).date()
                except (TypeError, ValueError):
                    still_open.append(trade)
                    continue
                if expiration > now_utc.date():
                    still_open.append(trade)
                    continue

                strike = _as_float(trade.contract_strike)
                intrinsic = 0.0
                if spot is not None and strike is not None:
                    if str(trade.contract_side).upper() == "C":
                        intrinsic = max(0.0, spot - strike)
                    elif str(trade.contract_side).upper() == "P":
                        intrinsic = max(0.0, strike - spot)
                trade.status = "closed"
                trade.closed_at = now_utc.isoformat()
                trade.exit_price = spot
                trade.exit_reason = "expiration"
                trade.exit_contract_price = intrinsic
                if trade.entry_contract_price is not None:
                    trade.pnl_dollars = (
                        intrinsic - float(trade.entry_contract_price)
                    ) * 100.0 * float(self._config.option_order_qty)
                if spot is not None:
                    mult = 1.0 if trade.direction == "long" else -1.0
                    trade.pnl_points = (spot - trade.entry) * mult
                closed.append(trade)
                self._closed.append(trade)
            self._open[symbol] = still_open
        return closed

    def _select_contract(
        self,
        *,
        ticker: str,
        option_type: str,
        target_strike: float,
        current_price: float,
        ref_dt: datetime,
    ) -> dict[str, Any] | None:
        start = ref_dt.date() + timedelta(days=min(self._config.dte_offsets))
        end = ref_dt.date() + timedelta(days=max(self._config.dte_offsets))
        window = max(1.0, abs(float(target_strike) - float(current_price)) * 2.0, float(current_price) * 0.01)
        params = {
            "underlying_symbol": ticker,
            "expiration_date_gte": start.isoformat(),
            "expiration_date_lte": end.isoformat(),
            "type": option_type,
            "status": "active",
            "strike_price_gte": max(0.01, float(target_strike) - window),
            "strike_price_lte": float(target_strike) + window,
        }
        resp = self._client.get_option_contracts(**params)
        contracts = self._extract_contracts(resp)
        if not contracts:
            params.pop("strike_price_gte", None)
            params.pop("strike_price_lte", None)
            resp = self._client.get_option_contracts(**params)
            contracts = self._extract_contracts(resp)
        if not contracts:
            return None

        candidates: list[tuple[float, float, dict[str, Any]]] = []
        for contract in contracts:
            if not isinstance(contract, dict) or not _is_standard_100_contract(contract):
                continue
            exp = _contract_expiry(contract)
            strike = _as_float(contract.get("strike_price"))
            symbol = _contract_symbol(contract)
            if exp is None or strike is None or not symbol:
                continue
            expiry_score = max(0.0, (exp.date() - start).days)
            strike_score = abs(strike - float(target_strike))
            candidates.append((expiry_score, strike_score, contract))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    @staticmethod
    def _extract_contracts(resp: Any) -> list[dict[str, Any]]:
        if isinstance(resp, dict):
            page = resp.get("option_contracts")
            if isinstance(page, list):
                return [c for c in page if isinstance(c, dict)]
        if isinstance(resp, list):
            return [c for c in resp if isinstance(c, dict)]
        return []

    def _latest_contract_price(self, symbol: str, *, mode: str) -> float | None:
        return self._latest_contract_quote(symbol, mode=mode)["price"]

    def _latest_contract_quote(self, symbol: str, *, mode: str) -> dict[str, float | None]:
        if not symbol:
            return {"price": None, "bid": None, "ask": None, "last": None, "mid": None, "spread_pct": None}
        payload = self._client.get_option_quotes(symbols=symbol, limit=1)
        quote = _extract_latest_quote_payload(payload, symbol)
        return _quote_metrics(quote, mode=mode)
