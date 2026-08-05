"""Route a Meta Ranker order plan through the governed execution path.

This is the Meta side of the Task 23 cutover. The runner no longer decides what
reaches the broker; it produces a plan, and every row of that plan becomes a
context snapshot, a ``TradeIntent``, a ``PolicyDecision``, and — only when the
policy permits it — an ``OrderRequest`` handed to
``DecisionCoordinator -> ExecutionGateway``.

Two properties matter more than anything else here:

* Nothing reaches a broker that the policy did not approve, and production-live
  is refused before any broker object is constructed at all.
* Every plan row is accounted for. A row that is silently dropped is a position
  nobody is managing, which is worse than a row that is loudly refused.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import math
from typing import Any
from uuid import UUID

from core.live_4h_exec import shares_for_notional
from core.nervous_system.contracts.enums import (
    DebitCredit,
    DecisionKind,
    InstrumentFamily,
    OrderSide,
    PolicyAction,
    PolicyMode,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.contracts.orders import OrderRequest
from core.nervous_system.execution.gateway import order_request_id_for
from signals.meta_context.meta_ranker.nervous_system_adapter import (
    MetaIntentConfig,
    build_plan_intents,
    underlying_for,
)


class RouterRefusal(str, Enum):
    OFF_MODE_SUBMIT = "OFF_MODE_CANNOT_SUBMIT"
    POLICY_VETO = "POLICY_VETO"
    NO_REFERENCE_PRICE = "NO_REFERENCE_PRICE"
    BELOW_ONE_SHARE = "BUDGET_BELOW_ONE_SHARE"
    OPTION_ROUTE_NOT_GOVERNED = "OPTION_ROUTE_NOT_GOVERNED"


_VETO_ACTIONS = frozenset({PolicyAction.REJECT, PolicyAction.DEFER})

# How long a planned order stays valid. Matches the MVP policy entry window, so
# a queued order cannot be submitted against a stale decision.
DEFAULT_ORDER_TTL = timedelta(minutes=20)


@dataclass(frozen=True)
class RoutedRow:
    """One plan row and everything the governed path decided about it."""

    symbol: str
    side: str
    quantity: Decimal
    intent: TradeIntent
    policy_action: PolicyAction | None = None
    order_request: OrderRequest | None = None
    outcome: Any = None
    refusal: RouterRefusal | None = None

    @property
    def submitted(self) -> bool:
        return bool(getattr(self.outcome, "submitted", False))


def equity_order_request(
    *,
    decision_id: UUID,
    policy_decision_id: UUID,
    environment: RuntimeEnvironment,
    account_alias: str,
    decision_kind: DecisionKind,
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    risk_reducing: bool,
    broker_position_key: str | None,
    maximum_loss: Decimal,
    buying_power_required: Decimal,
    idempotency_key: str,
    created_at: datetime,
    expires_at: datetime,
) -> OrderRequest:
    """Build one equity market order with a content-derived identity.

    ``DebitCredit.DEBIT`` is used for both sides: CREDIT is reserved for option
    structures that collect a premium, and the contract requires a positive
    limit magnitude alongside it, which a market order does not have.
    """

    if not isinstance(quantity, Decimal) or not quantity.is_finite() or quantity <= 0:
        raise ValueError("equity order quantity must be a positive finite Decimal")

    fields: dict[str, Any] = {
        "decision_id": decision_id,
        "policy_decision_id": policy_decision_id,
        "environment": environment,
        "account_alias": account_alias,
        "decision_kind": decision_kind,
        "risk_reducing": risk_reducing,
        "broker_position_key": broker_position_key,
        "instrument_family": InstrumentFamily.EQUITY,
        "equity_symbol": symbol,
        "equity_side": side,
        "parent_quantity": quantity,
        "debit_credit": DebitCredit.DEBIT,
        "net_limit_price": None,
        "maximum_loss": maximum_loss,
        "buying_power_required": buying_power_required,
        "time_in_force": "day",
        "order_type": "market",
        "idempotency_key": idempotency_key,
        "created_at": created_at,
        "expires_at": expires_at,
    }
    # Build once to obtain the content hash, then rebuild with the identity
    # derived from it, so the same order content always has the same ID.
    provisional = OrderRequest.create(order_request_id=UUID(int=0), **fields)
    return OrderRequest.create(
        order_request_id=order_request_id_for(decision_id, provisional.request_hash),
        **fields,
    )


class MetaGatewayRouter:
    """Turn one Meta order plan into governed decisions."""

    def __init__(
        self,
        *,
        coordinator: Any,
        snapshot_builder: Any,
        policy_evaluator: Callable[[TradeIntent, Any, Any], Any],
        policy_config: Any,
        freshness_profile: Any,
        environment: RuntimeEnvironment,
        account_alias: str,
        intent_config: MetaIntentConfig,
        clock: Callable[[], datetime],
        order_ttl: timedelta = DEFAULT_ORDER_TTL,
    ) -> None:
        if environment is RuntimeEnvironment.PRODUCTION_LIVE:
            # Refused here, in the constructor, so that no snapshot, intent,
            # order, or broker object is ever created for a live account.
            raise ValueError(
                "PRODUCTION_LIVE is refused: the Meta path has no live route"
            )
        self._coordinator = coordinator
        self._snapshots = snapshot_builder
        self._evaluate = policy_evaluator
        self._policy_config = policy_config
        self._profile = freshness_profile
        self._environment = environment
        self._account_alias = account_alias
        self._intent_config = intent_config
        self._clock = clock
        self._order_ttl = order_ttl

    def route(
        self,
        plan: Sequence[Sequence[Any]],
        *,
        exit_context: Mapping[str, Any],
        ticker_by_symbol: Mapping[str, str],
        scores_by_ticker: Mapping[str, Mapping[str, object]],
        decision_bar: datetime,
        reference_prices: Mapping[str, float],
        position_keys: Mapping[str, str],
        policy_mode: PolicyMode,
        submit: bool,
    ) -> tuple[RoutedRow, ...]:
        """Plan and, where permitted, submit every row of one order plan."""

        now = self._clock()
        snapshots = self._build_snapshots(
            plan,
            ticker_by_symbol=ticker_by_symbol,
            decision_bar=decision_bar,
            now=now,
        )
        intents = build_plan_intents(
            plan,
            exit_context=exit_context,
            ticker_by_symbol=ticker_by_symbol,
            scores_by_ticker=scores_by_ticker,
            decision_time=now,
            decision_bar=decision_bar,
            snapshot_id_by_ticker={
                ticker: snapshot.snapshot_id for ticker, snapshot in snapshots.items()
            },
            config=self._intent_config,
        )

        rows: list[RoutedRow] = []
        for plan_row, intent in zip(plan, intents):
            rows.append(
                self._route_one(
                    plan_row,
                    intent=intent,
                    snapshot=snapshots[intent.ticker],
                    reference_prices=reference_prices,
                    position_keys=position_keys,
                    policy_mode=policy_mode,
                    submit=submit,
                    decision_bar=decision_bar,
                )
            )
        return tuple(rows)

    # -- internals ----------------------------------------------------------

    def _build_snapshots(
        self,
        plan: Sequence[Sequence[Any]],
        *,
        ticker_by_symbol: Mapping[str, str],
        decision_bar: datetime,
        now: datetime,
    ) -> dict[str, Any]:
        snapshots: dict[str, Any] = {}
        for row in plan:
            ticker = ticker_by_symbol.get(row[0]) or underlying_for(row[0])
            if ticker in snapshots:
                continue
            snapshots[ticker] = self._snapshots.build(
                strategy_id=self._intent_config.strategy_id,
                entity_id=ticker,
                decision_time=now,
                decision_bar=decision_bar,
                profile=self._profile,
            )
        return snapshots

    def _route_one(
        self,
        plan_row: Sequence[Any],
        *,
        intent: TradeIntent,
        snapshot: Any,
        reference_prices: Mapping[str, float],
        position_keys: Mapping[str, str],
        policy_mode: PolicyMode,
        submit: bool,
        decision_bar: datetime,
    ) -> RoutedRow:
        symbol, side, quantity = plan_row[0], str(plan_row[1]).lower(), plan_row[2]
        route = plan_row[4] if len(plan_row) > 4 else "equity"
        base = {
            "symbol": symbol,
            "side": side,
            "quantity": Decimal(str(quantity)),
            "intent": intent,
        }

        if policy_mode is PolicyMode.OFF and submit:
            # OFF records a baseline and never submits; asking it to is a
            # caller error, not a silent downgrade.
            return RoutedRow(**base, refusal=RouterRefusal.OFF_MODE_SUBMIT)

        if route == "option":
            # options_exec computes bid/ask but does not return them, and never
            # records a quote timestamp, so an OptionLeg cannot be built. The
            # old direct-submit path would bypass the gateway entirely, so the
            # row is refused instead of quietly ungoverned.
            return RoutedRow(**base, refusal=RouterRefusal.OPTION_ROUTE_NOT_GOVERNED)

        policy_decision = self._evaluate(intent, snapshot, self._policy_config)
        base["policy_action"] = policy_decision.action
        if policy_decision.action in _VETO_ACTIONS:
            return RoutedRow(**base, refusal=RouterRefusal.POLICY_VETO)

        risk_reducing = intent.decision_kind is not DecisionKind.ENTRY
        if risk_reducing:
            order_quantity = intent.position_size_requested
        else:
            price = reference_prices.get(intent.ticker)
            if not price or not math.isfinite(float(price)) or float(price) <= 0:
                # Sizing needs the exact decision-bar price. Guessing one would
                # put a fabricated quantity on a real order.
                return RoutedRow(**base, refusal=RouterRefusal.NO_REFERENCE_PRICE)
            order_quantity = Decimal(
                str(
                    shares_for_notional(
                        float(price), float(policy_decision.final_risk_budget)
                    )
                )
            )
        if order_quantity <= 0:
            return RoutedRow(**base, refusal=RouterRefusal.BELOW_ONE_SHARE)

        order_request = equity_order_request(
            decision_id=self._decision_id(intent, policy_decision),
            policy_decision_id=policy_decision.policy_decision_id,
            environment=self._environment,
            account_alias=self._account_alias,
            decision_kind=intent.decision_kind,
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            quantity=order_quantity,
            risk_reducing=risk_reducing,
            broker_position_key=position_keys.get(symbol) if risk_reducing else None,
            maximum_loss=(
                Decimal("0") if risk_reducing else policy_decision.final_risk_budget
            ),
            buying_power_required=(
                Decimal("0") if risk_reducing else policy_decision.final_risk_budget
            ),
            idempotency_key=intent.idempotency_key,
            # Anchored to the decision bar, not the wall clock. If the order's
            # timestamps moved with each attempt its content hash would move
            # too, and a retried 4H pass would mint a second client order ID
            # for the same decision — a duplicate order. Anchoring also makes
            # the gateway's expiry check fail closed on a stale replay.
            created_at=decision_bar,
            expires_at=decision_bar + self._order_ttl,
        )
        base["order_request"] = order_request

        outcome = self._coordinator.process_intent(
            intent,
            policy_mode=policy_mode,
            submit=submit,
            snapshot=snapshot,
            policy_decision=policy_decision,
            order_request=order_request,
        )
        return RoutedRow(**base, outcome=outcome)

    @staticmethod
    def _decision_id(intent: TradeIntent, policy_decision: Any) -> UUID:
        from core.nervous_system.orchestration.coordinator import DecisionCoordinator

        return DecisionCoordinator._decision_id(intent, policy_decision)


class GovernedPathUnavailable(RuntimeError):
    """The governed execution path could not be constructed.

    Raised, never swallowed. A caller that wanted to submit must stop: falling
    back to a direct broker call is exactly the bypass this module exists to
    prevent.
    """


def build_router(
    *,
    intent_config: MetaIntentConfig,
    environ: Mapping[str, str] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> MetaGatewayRouter:
    """Assemble the governed path from the documented environment.

    Every dependency is constructed here so the runner never touches a broker
    client, a session, or a journal directly.
    """

    from datetime import timezone

    from core.nervous_system.config.freshness import get_snapshot_profile
    from core.nervous_system.config.policy import MVP_POLICY_CONFIG
    from core.nervous_system.config.runtime import NervousSystemSettings
    from core.nervous_system.context.snapshot_builder import SnapshotBuilder
    from core.nervous_system.orchestration.coordinator import DecisionCoordinator
    from core.nervous_system.persistence.database import (
        create_database_engine,
        create_session_factory,
    )
    from core.nervous_system.persistence.repositories.state import StateRepository
    from core.nervous_system.persistence.uow import UnitOfWork
    from core.nervous_system.policy.engine import evaluate_policy

    try:
        settings = NervousSystemSettings.from_env(environ)
    except Exception as exc:  # noqa: BLE001 - any misconfiguration must fail closed
        raise GovernedPathUnavailable(
            "the nervous-system environment is not configured; refusing to submit"
        ) from exc

    if settings.environment is RuntimeEnvironment.PRODUCTION_LIVE:
        raise GovernedPathUnavailable(
            "PRODUCTION_LIVE is refused: the Meta path has no live route"
        )

    tick = clock or (lambda: datetime.now(timezone.utc))
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    def uow_factory() -> UnitOfWork:
        return UnitOfWork(session_factory)

    policy_config = replace(
        MVP_POLICY_CONFIG,
        mode=settings.policy_mode,
        environment=settings.environment,
        account_alias=settings.account_alias,
    )
    return MetaGatewayRouter(
        coordinator=DecisionCoordinator(
            environment=settings.environment,
            unit_of_work_factory=uow_factory,
            clock=tick,
            gateway_factory=lambda: _build_gateway(settings, session_factory, tick),
        ),
        snapshot_builder=SnapshotBuilder(StateRepository(session_factory())),
        policy_evaluator=evaluate_policy,
        policy_config=policy_config,
        freshness_profile=get_snapshot_profile(policy_config.required_snapshot_profile),
        environment=settings.environment,
        account_alias=settings.account_alias,
        intent_config=intent_config,
        clock=tick,
    )


def _build_gateway(settings: Any, session_factory: Any, clock: Callable[[], datetime]) -> Any:
    """Construct the broker-facing gateway. Only ever called to submit."""

    from core.nervous_system.execution.alpaca_adapter import AlpacaPaperAdapter
    from core.nervous_system.execution.gateway import ExecutionGateway
    from core.nervous_system.execution.journal import LocalAtomicJournal
    from core.nervous_system.persistence.uow import UnitOfWork

    return ExecutionGateway(
        broker=AlpacaPaperAdapter(environment=settings.environment),
        journal=LocalAtomicJournal(settings.operational_root / "execution_journal"),
        unit_of_work_factory=lambda: UnitOfWork(session_factory),
        environment=settings.environment,
        account_alias=settings.account_alias,
        worker_id=f"meta-ranker@{settings.account_alias}",
        clock=clock,
    )


__all__ = [
    "DEFAULT_ORDER_TTL",
    "GovernedPathUnavailable",
    "MetaGatewayRouter",
    "RoutedRow",
    "RouterRefusal",
    "build_router",
    "equity_order_request",
]
