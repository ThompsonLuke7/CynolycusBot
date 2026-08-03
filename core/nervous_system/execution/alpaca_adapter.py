"""Alpaca paper broker adapter.

This adapter is paper-only by construction.  ``PRODUCTION_LIVE`` fails in the
constructor, before any HTTP transport exists, so a live submission cannot be
reached even by a caller that ignores every other guard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlparse
import urllib.error

from core.nervous_system.contracts.enums import (
    AssetClass,
    DebitCredit,
    ExecutionStatus,
    InstrumentFamily,
    OrderSide,
)
from core.nervous_system.contracts.orders import OrderRequest
from core.nervous_system.contracts.enums import RuntimeEnvironment

from .broker import (
    ALPACA_STATUS_MAP,
    BrokerAccount,
    BrokerAmbiguousSubmission,
    BrokerAuthenticationError,
    BrokerContractError,
    BrokerOrder,
    BrokerOrderLeg,
    BrokerPosition,
    BrokerRejected,
    BrokerUnavailable,
    OrderReplacement,
)


PAPER_HOSTS = frozenset({"paper-api.alpaca.markets"})
PAPER_ACCOUNT_ALIASES = frozenset({"paper"})
CLIENT_ORDER_ID_LENGTH = 48

_SECRET_KEY_HINTS = ("key", "secret", "token", "password", "authorization", "apca")
_SIDE_MAP = {OrderSide.BUY: "buy", OrderSide.SELL: "sell"}
_ZERO = Decimal("0")


def client_order_id_for(request: OrderRequest) -> str:
    """Deterministic 48-character broker idempotency key."""

    key = request.idempotency_key.strip()
    if not key:
        raise BrokerContractError("order request carries no idempotency key")
    return key[:CLIENT_ORDER_ID_LENGTH]


def sanitize(value: Any) -> Any:
    """Recursively drop credential-looking values from a broker payload."""

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if any(hint in name.lower() for hint in _SECRET_KEY_HINTS) and (
                "order" not in name.lower()
            ):
                cleaned[name] = "***redacted***"
            else:
                cleaned[name] = sanitize(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    return value


def _decimal(value: Any, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BrokerContractError(f"{field} is not a number: {value!r}") from exc


def _optional_decimal(value: Any, field: str) -> Decimal | None:
    if value is None or value == "":
        return None
    return _decimal(value, field)


def _timestamp(value: Any, field: str) -> datetime | None:
    if value is None or value == "":
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise BrokerContractError(f"{field} is not an ISO timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise BrokerContractError(f"{field} must be timezone-aware: {value!r}")
    return parsed.astimezone(timezone.utc)


class AlpacaPaperAdapter:
    """Translate between nervous-system contracts and the Alpaca paper API."""

    def __init__(
        self,
        client: Any,
        *,
        environment: RuntimeEnvironment,
        account_alias: str,
        trading_base_url: str,
        clock: Any = None,
    ) -> None:
        if environment is RuntimeEnvironment.PRODUCTION_LIVE:
            raise BrokerAuthenticationError(
                "PRODUCTION_LIVE is refused: this adapter is paper-only"
            )
        if environment is RuntimeEnvironment.QA_PAPER:
            if account_alias.strip().lower() not in PAPER_ACCOUNT_ALIASES:
                raise BrokerAuthenticationError(
                    "QA_PAPER requires the paper account alias"
                )
            host = urlparse(trading_base_url).hostname or ""
            if host.lower() not in PAPER_HOSTS:
                raise BrokerAuthenticationError(
                    f"QA_PAPER requires the paper host, not {host!r}"
                )
        self._client = client
        self._environment = environment
        self._account_alias = account_alias
        self._trading_base_url = trading_base_url
        self._clock = clock

    # -- reads --------------------------------------------------------------

    def _now(self) -> datetime:
        if self._clock is not None:
            return self._clock()
        return datetime.now(timezone.utc)

    def account(self) -> BrokerAccount:
        payload = self._call(self._client.get_account)
        if not isinstance(payload, Mapping):
            raise BrokerContractError("account response is not an object")
        return BrokerAccount(
            account_id=str(payload.get("id", "")),
            account_alias=self._account_alias,
            status=str(payload.get("status", "UNKNOWN")),
            equity=_decimal(payload.get("equity", "0"), "equity"),
            cash=_decimal(payload.get("cash", "0"), "cash"),
            buying_power=_decimal(payload.get("buying_power", "0"), "buying_power"),
            observed_at=self._now(),
            raw=sanitize(dict(payload)),
        )

    def positions(self) -> tuple[BrokerPosition, ...]:
        payload = self._call(self._client.get_positions)
        if payload is None:
            return ()
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise BrokerContractError("positions response is not a list")
        observed_at = self._now()
        return tuple(
            BrokerPosition(
                symbol=str(item.get("symbol", "")),
                # Alpaca reports "us_option"/"us_equity", so match on the
                # substring: a prefix check would silently class every option
                # position as equity and corrupt exposure.
                asset_class=(
                    AssetClass.OPTION
                    if "option" in str(item.get("asset_class", "")).lower()
                    else AssetClass.EQUITY
                ),
                quantity=_decimal(item.get("qty", "0"), "position qty"),
                average_entry_price=_optional_decimal(
                    item.get("avg_entry_price"), "avg_entry_price"
                ),
                market_value=_optional_decimal(item.get("market_value"), "market_value"),
                observed_at=observed_at,
                raw=sanitize(dict(item)),
            )
            for item in payload
        )

    def orders(self, *, status: str = "all") -> tuple[BrokerOrder, ...]:
        payload = self._call(lambda: self._client.get_orders(status=status))
        if payload is None:
            return ()
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise BrokerContractError("orders response is not a list")
        return tuple(self._to_order(item) for item in payload)

    def find_by_client_order_id(self, client_order_id: str) -> BrokerOrder | None:
        payload = self._call(
            lambda: self._client.get_order_by_client_order_id(client_order_id)
        )
        if payload is None:
            return None
        return self._to_order(payload)

    # -- writes -------------------------------------------------------------

    def submit(self, request: OrderRequest) -> BrokerOrder:
        self._require_paper_submission(request)
        client_order_id = client_order_id_for(request)

        if request.instrument_family is InstrumentFamily.EQUITY:
            if request.equity_symbol is None or request.equity_side is None:
                raise BrokerContractError("an equity request requires symbol and side")
            payload = self._submit_equity(request, client_order_id)
        else:
            payload = self._submit_multileg(request, client_order_id)

        if not isinstance(payload, Mapping):
            raise BrokerContractError("submit response is not an object")
        return self._to_order(payload)

    def _submit_equity(self, request: OrderRequest, client_order_id: str) -> Any:
        return self._call_write(
            lambda: self._client.submit_order(
                symbol=request.equity_symbol,
                qty=int(request.parent_quantity),
                side=_SIDE_MAP[request.equity_side],
                order_type=request.order_type,
                time_in_force=request.time_in_force,
                limit_price=(
                    float(request.net_limit_price)
                    if request.net_limit_price is not None
                    else None
                ),
                client_order_id=client_order_id,
            )
        )

    def _submit_multileg(self, request: OrderRequest, client_order_id: str) -> Any:
        legs = [
            {
                "symbol": leg.symbol,
                "ratio_qty": leg.ratio,
                "side": _SIDE_MAP[leg.side],
                "position_intent": leg.position_intent.value.lower(),
            }
            for leg in request.legs
        ]
        limit_price: Decimal | None = None
        if request.net_limit_price is not None:
            magnitude = request.net_limit_price
            # A multi-leg limit is signed: positive pays a debit, negative
            # receives a credit.
            limit_price = (
                magnitude
                if request.debit_credit is DebitCredit.DEBIT
                else -magnitude
            )
        return self._call_write(
            lambda: self._client.submit_multileg_order(
                legs=legs,
                qty=int(request.parent_quantity),
                order_type=request.order_type,
                time_in_force=request.time_in_force,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
        )

    def cancel(self, broker_order_id: str) -> BrokerOrder:
        self._call_write(lambda: self._client.cancel_order(broker_order_id))
        # DELETE answers 204 with no body.  Refetch rather than fabricating a
        # canceled terminal state the broker never confirmed.
        payload = self._call(lambda: self._client.get_order(broker_order_id))
        if not isinstance(payload, Mapping):
            raise BrokerContractError("cancel refetch did not return an order")
        return self._to_order(payload)

    def replace(
        self,
        broker_order_id: str,
        replacement: OrderReplacement,
    ) -> BrokerOrder:
        payload = self._call_write(
            lambda: self._client.replace_order(
                broker_order_id, replacement.to_payload()
            )
        )
        if not isinstance(payload, Mapping):
            raise BrokerContractError("replace response is not an object")
        # A successful PATCH is not proof the replacement beat a fill; the
        # returned status is the broker's answer and is preserved verbatim.
        return self._to_order(payload)

    # -- translation --------------------------------------------------------

    def _to_order(self, payload: Mapping[str, Any]) -> BrokerOrder:
        if not isinstance(payload, Mapping):
            raise BrokerContractError("order payload is not an object")
        broker_order_id = str(payload.get("id", "")).strip()
        if not broker_order_id:
            raise BrokerContractError("order payload has no broker id")
        raw_status = str(payload.get("status", "")).strip()
        if not raw_status:
            raise BrokerContractError("order payload has no status")
        status = ALPACA_STATUS_MAP.get(raw_status.lower(), ExecutionStatus.UNKNOWN)

        filled_quantity = _decimal(payload.get("filled_qty", "0") or "0", "filled_qty")
        average_fill_price = _optional_decimal(
            payload.get("filled_avg_price"), "filled_avg_price"
        )
        legs_payload = payload.get("legs") or ()
        legs = tuple(
            BrokerOrderLeg(
                symbol=str(leg.get("symbol", "")),
                ratio_quantity=int(leg.get("ratio_qty", 1) or 1),
                side=OrderSide.BUY
                if str(leg.get("side", "buy")).lower() == "buy"
                else OrderSide.SELL,
                position_intent=(
                    str(leg["position_intent"])
                    if leg.get("position_intent") is not None
                    else None
                ),
                raw_status=str(leg.get("status", raw_status)),
                filled_quantity=_decimal(leg.get("filled_qty", "0") or "0", "leg filled_qty"),
                average_fill_price=_optional_decimal(
                    leg.get("filled_avg_price"), "leg filled_avg_price"
                ),
                broker_order_id=str(leg["id"]) if leg.get("id") is not None else None,
            )
            for leg in legs_payload
        )
        return BrokerOrder(
            broker_order_id=broker_order_id,
            client_order_id=str(payload.get("client_order_id", "")),
            status=status,
            raw_status=raw_status,
            submitted_at=_timestamp(payload.get("submitted_at"), "submitted_at"),
            updated_at=_timestamp(payload.get("updated_at"), "updated_at"),
            filled_at=_timestamp(payload.get("filled_at"), "filled_at"),
            filled_quantity=filled_quantity,
            average_fill_price=average_fill_price,
            legs=legs,
            observed_at=self._now(),
            raw=sanitize(dict(payload)),
        )

    # -- transport boundary -------------------------------------------------

    def _require_paper_submission(self, request: OrderRequest) -> None:
        if self._environment is RuntimeEnvironment.PRODUCTION_LIVE:
            raise BrokerAuthenticationError("PRODUCTION_LIVE submissions are refused")
        if request.environment is RuntimeEnvironment.PRODUCTION_LIVE:
            raise BrokerAuthenticationError(
                "order request is marked PRODUCTION_LIVE and cannot be submitted"
            )
        if request.account_alias != self._account_alias:
            raise BrokerAuthenticationError(
                "order request account alias does not match the adapter account"
            )

    def _call(self, operation: Any) -> Any:
        """Run a read, translating transport failures into typed errors."""

        try:
            return operation()
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc, ambiguous=False) from exc
        except urllib.error.URLError as exc:
            raise BrokerUnavailable(f"broker unreachable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise BrokerUnavailable("broker read timed out") from exc

    def _call_write(self, operation: Any) -> Any:
        """Run a mutating call, where a lost response is ambiguous, not failed."""

        try:
            return operation()
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc, ambiguous=True) from exc
        except urllib.error.URLError as exc:
            raise BrokerAmbiguousSubmission(
                f"broker connection lost during a write: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise BrokerAmbiguousSubmission("broker write timed out") from exc

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError, *, ambiguous: bool) -> Exception:
        code = exc.code
        if code in {401, 403}:
            return BrokerAuthenticationError(f"broker refused credentials ({code})")
        if 500 <= code < 600:
            if ambiguous:
                # The order may already exist; reconcile by client order ID
                # rather than resubmitting.
                return BrokerAmbiguousSubmission(f"broker server error ({code})")
            return BrokerUnavailable(f"broker server error ({code})")
        if code == 429:
            return BrokerUnavailable("broker rate limited (429)")
        return BrokerRejected(f"broker rejected the request ({code}): {exc.reason}")


__all__ = [
    "CLIENT_ORDER_ID_LENGTH",
    "PAPER_ACCOUNT_ALIASES",
    "PAPER_HOSTS",
    "AlpacaPaperAdapter",
    "client_order_id_for",
    "sanitize",
]
