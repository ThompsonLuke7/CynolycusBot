from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from ..core.config import AlpacaConfig, _profile_env_value, _read_env_file, _split_env_file_profile


@dataclass(frozen=True)
class OptionsClientConfig:
    trading_base_url: str
    data_base_url: str
    timeout_sec: int = 30

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "OptionsClientConfig":
        env_path, profile = _split_env_file_profile(env_file)
        file_values = _read_env_file(env_path) if env_path else {}
        trading = (
            _profile_env_value(file_values, profile, "APCA_API_BASE_URL", "ALPACA_TRADING_API_BASE_URL")
            or "https://paper-api.alpaca.markets"
        )
        data = (
            _profile_env_value(file_values, profile, "ALPACA_DATA_API_BASE_URL")
            or "https://data.alpaca.markets"
        )
        return cls(trading_base_url=trading, data_base_url=data)


class AlpacaOptionsClient:
    """
    Minimal REST client for Alpaca Options API.

    Core endpoints:
      - GET /v2/options/contracts
      - GET /v2/options/quotes
      - POST /v2/orders (with option symbol)
    """

    def __init__(
        self,
        *,
        env_file: str | None = ".env",
        trading_base_url: str | None = None,
        data_base_url: str | None = None,
        timeout_sec: int = 30,
    ) -> None:
        cfg = AlpacaConfig.from_env(env_file)
        defaults = OptionsClientConfig.from_env(env_file)
        self._key = cfg.key_id
        self._secret = cfg.secret_key
        self._trading_base = (trading_base_url or defaults.trading_base_url).rstrip("/")
        self._data_base = (data_base_url or defaults.data_base_url).rstrip("/")
        self._timeout = int(timeout_sec)

    @staticmethod
    def format_option_symbol(
        *,
        underlying: str,
        expiration: str,
        call_put: str,
        strike: float,
    ) -> str:
        """
        Format OCC-style option symbol (e.g., SPY240216C00475000).

        expiration: YYYYMMDD or YYMMDD (string)
        call_put: "C" or "P"
        strike: strike price (e.g., 475.0)
        """
        u = underlying.strip().upper()
        exp = expiration.strip()
        if len(exp) == 8:
            exp = exp[2:]
        cp = call_put.strip().upper()
        strike_int = int(round(float(strike) * 1000))
        return f"{u}{exp}{cp}{strike_int:08d}"

    def _redact(self, text: Any) -> Any:
        """Strip API credentials out of anything that may reach a log or raise."""

        if not isinstance(text, str):
            return text
        for secret in (self._key, self._secret):
            if secret:
                text = text.replace(secret, "***redacted***")
        return text

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> Any:
        if params:
            query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            url = f"{url}?{query}"
        data = None
        headers = {
            "APCA-API-KEY-ID": self._key,
            "APCA-API-SECRET-KEY": self._secret,
        }
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        # Retry transient rate-limit (429) / server (5xx) errors with backoff.
        # These were causing ~100 hard skips/day (no_quote, delta->ATM fallback).
        # On persistent failure we re-raise so callers keep their existing
        # skip-and-move-on behavior.
        # POST is not idempotent here (no client_order_id dedupe on
        # /v2/orders), so a 5xx — which may mean the order was actually
        # accepted and only the response delivery failed — must NOT be
        # retried or it can double-submit a live order. 429 means the
        # request was rejected before any processing, so it is always safe
        # to retry regardless of method.
        # PATCH carries the same hazard: a replace whose response was lost may
        # already have been applied, so it is never retried either.
        is_post = method.upper() in {"POST", "PATCH"}
        backoffs = (0.5, 1.0, 2.0)
        for attempt in range(len(backoffs) + 1):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    if not raw:
                        return None
                    return json.loads(raw)
            except urllib.error.HTTPError as exc:
                transient = exc.code == 429 or (500 <= exc.code < 600 and not is_post)
                if not transient or attempt >= len(backoffs):
                    # Surface Alpaca's actual rejection reason (the body) — otherwise
                    # a 403/422 stringifies to a bare "Forbidden"/"Unprocessable
                    # Entity" and we can't tell buying-power vs after-hours-OPG vs
                    # no-position. The body is only readable once, so fold it into
                    # the re-raised error's message while preserving .code.
                    body = ""
                    try:
                        body = (exc.read() or b"").decode("utf-8", "replace")[:400].strip()
                    except Exception:
                        body = ""
                    if body:
                        raise urllib.error.HTTPError(
                            self._redact(exc.url),
                            exc.code,
                            self._redact(f"{exc.reason}: {body}"),
                            exc.headers,
                            None,
                        ) from exc
                    raise
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else backoffs[attempt]
                except (TypeError, ValueError):
                    delay = backoffs[attempt]
                time.sleep(min(delay, 5.0))

    def get_option_contracts(self, **params: Any) -> Any:
        """
        GET /v2/options/contracts

        Example params:
          underlying_symbol="SPY", expiration_date="2024-02-16",
          type="call", strike_price_gte=470, strike_price_lte=480
        """
        url = f"{self._trading_base}/v2/options/contracts"
        return self._request("GET", url, params=params)

    def get_option_snapshots(self, underlying_symbol: str, **params: Any) -> dict:
        """
        GET /v1beta1/options/snapshots/{underlying_symbol}

        Returns a dict keyed by OCC symbol, each value containing:
          greeks: {delta, gamma, theta, vega, rho}
          impliedVolatility, latestQuote, latestTrade

        Example params:
          expiration_date="2026-06-20", type="call",
          strike_price_gte=500, strike_price_lte=560
        """
        sym = underlying_symbol.strip().upper()
        url = f"{self._data_base}/v1beta1/options/snapshots/{sym}"
        result = self._request("GET", url, params=params)
        if isinstance(result, dict):
            return result.get("snapshots", result)
        return {}

    def get_positions(self, **params: Any) -> Any:
        """
        GET /v2/positions

        Optional params are passed through as query params.
        """
        url = f"{self._trading_base}/v2/positions"
        return self._request("GET", url, params=params)

    def get_orders(self, **params: Any) -> Any:
        """
        GET /v2/orders

        Example params:
          status="open", limit=100, direction="desc"
        """
        url = f"{self._trading_base}/v2/orders"
        return self._request("GET", url, params=params)

    def get_order(self, order_id: str) -> Any:
        """
        GET /v2/orders/{order_id}
        """
        oid = str(order_id).strip()
        if not oid:
            raise ValueError("order_id is required")
        url = f"{self._trading_base}/v2/orders/{oid}"
        return self._request("GET", url)

    def cancel_order(self, order_id: str) -> Any:
        """
        DELETE /v2/orders/{order_id}
        """
        oid = str(order_id).strip()
        if not oid:
            raise ValueError("order_id is required")
        url = f"{self._trading_base}/v2/orders/{oid}"
        return self._request("DELETE", url)

    def get_account(self) -> Any:
        """
        GET /v2/account
        """
        url = f"{self._trading_base}/v2/account"
        return self._request("GET", url)

    def get_option_quotes(self, **params: Any) -> Any:
        """
        GET options quotes.

        Example params:
          symbols="SPY240216C00475000" or symbol_or_symbols="..."
        """
        attempts: list[tuple[str, dict[str, Any]]] = []
        clean_params = {k: v for k, v in params.items() if v is not None}
        symbols = clean_params.get("symbols") or clean_params.get("symbol_or_symbols")
        base_params = dict(clean_params)
        latest_base_params = {k: v for k, v in base_params.items() if k != "limit"}
        if symbols is not None:
            params_symbols = dict(base_params)
            params_symbols["symbols"] = symbols
            params_symbols.pop("symbol_or_symbols", None)
            params_symbol_or_symbols = dict(base_params)
            params_symbol_or_symbols["symbol_or_symbols"] = symbols
            params_symbol_or_symbols.pop("symbols", None)
            latest_params_symbols = dict(latest_base_params)
            latest_params_symbols["symbols"] = symbols
            latest_params_symbols.pop("symbol_or_symbols", None)
            latest_params_symbol_or_symbols = dict(latest_base_params)
            latest_params_symbol_or_symbols["symbol_or_symbols"] = symbols
            latest_params_symbol_or_symbols.pop("symbols", None)
        else:
            params_symbols = dict(base_params)
            params_symbol_or_symbols = dict(base_params)
            latest_params_symbols = dict(latest_base_params)
            latest_params_symbol_or_symbols = dict(latest_base_params)

        attempts.extend(
            [
                (f"{self._data_base}/v1beta1/options/quotes/latest", latest_params_symbols),
                (f"{self._data_base}/v1beta1/options/quotes/latest", latest_params_symbol_or_symbols),
                (f"{self._data_base}/v2/options/quotes", params_symbols),
                (f"{self._data_base}/v2/options/quotes", params_symbol_or_symbols),
            ]
        )

        last_exc: Exception | None = None
        for url, req_params in attempts:
            try:
                return self._request("GET", url, params=req_params)
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code == 404:
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("option_quote_request_failed")

    def submit_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        position_intent: str | None = None,
        client_order_id: str | None = None,
    ) -> Any:
        """
        POST /v2/orders.

        ``client_order_id`` is optional and omitted from the payload unless a
        caller supplies one, so existing call sites are unaffected. Supplying
        it makes the submission idempotent at the broker.
        """
        url = f"{self._trading_base}/v2/orders"
        payload = {
            "symbol": symbol,
            "qty": int(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            payload["limit_price"] = float(limit_price)
        if stop_price is not None:
            payload["stop_price"] = float(stop_price)
        if position_intent is not None:
            payload["position_intent"] = position_intent
        if client_order_id is not None:
            payload["client_order_id"] = str(client_order_id)
        return self._request("POST", url, json_body=payload)

    def submit_option_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        position_intent: str | None = None,
        client_order_id: str | None = None,
    ) -> Any:
        """
        POST /v2/orders with an option symbol.

        ``position_intent`` (e.g. "sell_to_close", "buy_to_open") disambiguates
        options orders explicitly instead of relying on Alpaca to infer intent
        from current position state. Optional and omitted from the payload
        unless a caller passes it, so existing call sites are unaffected.
        """
        return self.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            position_intent=position_intent,
            client_order_id=client_order_id,
        )

    def submit_multileg_order(
        self,
        *,
        legs: "Sequence[Mapping[str, Any]]",
        qty: int,
        order_type: str,
        time_in_force: str,
        limit_price: float | Decimal | None,
        client_order_id: str,
    ) -> Any:
        """
        POST /v2/orders with ``order_class="mleg"``.

        A multi-leg request omits the parent ``symbol`` and ``side``; each leg
        carries its own ``symbol``, ``ratio_qty``, ``side``, and
        ``position_intent``. ``limit_price`` is positive for a net debit and
        negative for a net credit.
        """
        if not legs:
            raise ValueError("a multi-leg order requires at least one leg")
        if len(legs) > 4:
            raise ValueError("a multi-leg order accepts at most four legs")
        parent_qty = int(qty)
        if parent_qty <= 0:
            raise ValueError("qty must be a positive whole number of contracts")
        if order_type not in {"market", "limit"}:
            raise ValueError(f"unsupported multi-leg order type: {order_type!r}")
        if order_type == "limit" and limit_price is None:
            raise ValueError("a limit multi-leg order requires limit_price")
        if order_type == "market" and limit_price is not None:
            raise ValueError("a market multi-leg order cannot carry limit_price")
        if not str(client_order_id).strip():
            raise ValueError("client_order_id is required for multi-leg orders")

        normalized: list[dict[str, Any]] = []
        for index, leg in enumerate(legs, start=1):
            ratio = int(leg["ratio_qty"])
            if ratio <= 0:
                raise ValueError(f"leg {index} ratio_qty must be positive")
            normalized.append(
                {
                    "symbol": str(leg["symbol"]),
                    "ratio_qty": ratio,
                    "side": str(leg["side"]),
                    "position_intent": str(leg["position_intent"]),
                }
            )
        divisor = 0
        for leg in normalized:
            divisor = math.gcd(divisor, int(leg["ratio_qty"]))
        if divisor > 1:
            # Alpaca requires reduced ratios (greatest common divisor of one).
            for leg in normalized:
                leg["ratio_qty"] = int(leg["ratio_qty"]) // divisor

        payload: dict[str, Any] = {
            "order_class": "mleg",
            "qty": parent_qty,
            "type": order_type,
            "time_in_force": time_in_force,
            "client_order_id": str(client_order_id),
            "legs": normalized,
        }
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        url = f"{self._trading_base}/v2/orders"
        return self._request("POST", url, json_body=payload)

    def get_order_by_client_order_id(self, client_order_id: str) -> Any:
        """
        GET /v2/orders:by_client_order_id?client_order_id=...

        Returns ``None`` when the broker has no such order. This is the
        documented lookup; never scan the order list to find one.
        """
        cid = str(client_order_id).strip()
        if not cid:
            raise ValueError("client_order_id is required")
        url = f"{self._trading_base}/v2/orders:by_client_order_id"
        try:
            return self._request("GET", url, params={"client_order_id": cid})
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def replace_order(self, order_id: str, payload: Mapping[str, Any]) -> Any:
        """
        PATCH /v2/orders/{order_id}

        Supports ``qty``, ``limit_price``, ``stop_price``, ``time_in_force``,
        and a new ``client_order_id``. A structural change is a new linked
        order request, never a PATCH.
        """
        oid = str(order_id).strip()
        if not oid:
            raise ValueError("order_id is required")
        allowed = {"qty", "limit_price", "stop_price", "time_in_force", "client_order_id"}
        unsupported = sorted(set(payload) - allowed)
        if unsupported:
            raise ValueError(f"replace does not support fields: {unsupported}")
        if not payload:
            raise ValueError("replace requires at least one field")
        url = f"{self._trading_base}/v2/orders/{oid}"
        return self._request("PATCH", url, json_body=dict(payload))
