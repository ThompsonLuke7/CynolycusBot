"""Shared fixtures for nervous-system tests."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.decisions import (
    DecisionRecord,
    HashedDecisionArtifact,
)
from core.nervous_system.contracts.enums import (
    DebitCredit,
    DecisionKind,
    Direction,
    InstrumentFamily,
    ModifierOperation,
    OrderSide,
    PolicyAction,
    PolicyMode,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.contracts.orders import OrderRequest
from core.nervous_system.contracts.policy import PolicyDecision, PolicyModifier
from core.nervous_system.contracts.base import content_hash
from core.nervous_system.persistence.repositories.decision import CompleteDecisionChain


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """Return the disposable migration-test database URL or skip.

    The migration integration test upgrades, downgrades, and recreates the
    schema.  ``NERVOUS_SYSTEM_TEST_DATABASE_URL`` must therefore point to a
    dedicated disposable test database, never a development, QA, or
    production database.
    """

    value = os.environ.get("NERVOUS_SYSTEM_TEST_DATABASE_URL")
    if not value:
        pytest.skip("set NERVOUS_SYSTEM_TEST_DATABASE_URL for postgres tests")
    return value


@pytest.fixture(scope="session")
def postgres_engine(postgres_url: str):
    """Upgrade only the explicitly supplied disposable PostgreSQL database."""

    engine = create_engine(postgres_url)
    config_path = Path("core/nervous_system/persistence/alembic.ini")
    cfg = Config(str(config_path))
    cfg.set_main_option("sqlalchemy.url", postgres_url)
    with engine.connect() as connection:
        version_row = connection.execute(
            text(
                # Pinned to the current head: a database left at an earlier
                # revision must still be upgraded, or new tables silently
                # never appear.
                "SELECT version_num FROM public.alembic_version "
                "WHERE version_num = '0004_audit_observability'"
            )
        ).first()
    if version_row is None:
        command.upgrade(cfg, "head")
    yield engine
    engine.dispose()


@pytest.fixture
def pg_session(postgres_engine):
    connection = postgres_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, autoflush=False, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


@pytest.fixture
def session_factory(postgres_engine):
    return sessionmaker(bind=postgres_engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def complete_decision_chain() -> CompleteDecisionChain:
    now = datetime(2026, 7, 30, 18, 20, tzinfo=timezone.utc)
    snapshot = ContextSnapshot.from_states(
        snapshot_id=uuid4(),
        decision_time=now,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(),
        freshness_profile="qa",
    )
    intent = TradeIntent(
        intent_id=uuid4(),
        strategy_id="meta_ranker",
        ticker="AMD",
        direction=Direction.LONG,
        decision_kind=DecisionKind.ENTRY,
        raw_score=0.97,
        raw_probability=None,
        expected_return=None,
        expected_holding_period="53x4h",
        snapshot_id=snapshot.snapshot_id,
        selected_bar=None,
        entry_window="current-or-next-open",
        preferred_entry=None,
        invalidation=None,
        target=None,
        stop=None,
        position_size_requested=Decimal("100"),
        instrument_preferences=(InstrumentFamily.EQUITY,),
        feature_timestamp=now,
        created_at=now,
        model_version="meta@1",
        feature_version="matrix@1",
        reason_codes=("TEST",),
    )
    modifier = PolicyModifier(
        rule_id="risk.cap",
        rule_version="risk.cap@1",
        operation=ModifierOperation.CAP,
        input_value="100",
        configured_condition="max_position_risk",
        configured_value=Decimal("100"),
        budget_before=Decimal("100"),
        budget_after=Decimal("100"),
        reason_code="TEST",
        source_state_id=None,
        config_version="policy@1",
    )
    policy = PolicyDecision(
        policy_decision_id=uuid4(),
        intent_id=intent.intent_id,
        snapshot_id=snapshot.snapshot_id,
        environment=RuntimeEnvironment.QA_PAPER,
        mode=PolicyMode.ENFORCE,
        action=PolicyAction.APPROVE,
        approved_direction=Direction.LONG,
        base_risk_budget=Decimal("100"),
        final_risk_budget=Decimal("100"),
        allowed_instruments=frozenset({InstrumentFamily.EQUITY}),
        hard_vetoes=(),
        modifiers=(modifier,),
        stop_adjustment=None,
        target_adjustment=None,
        holding_period_adjustment=None,
        hedge_requirement=None,
        reason_codes=("TEST",),
        policy_version="policy@1",
        config_version="policy@1",
        created_at=now,
        expires_at=now + timedelta(minutes=20),
    )
    record_id = uuid4()
    order = OrderRequest.create(
        order_request_id=uuid4(),
        decision_id=record_id,
        policy_decision_id=policy.policy_decision_id,
        environment=RuntimeEnvironment.QA_PAPER,
        account_alias="paper",
        decision_kind=DecisionKind.ENTRY,
        risk_reducing=False,
        broker_position_key=None,
        instrument_family=InstrumentFamily.EQUITY,
        equity_symbol="AMD",
        equity_side=OrderSide.BUY,
        parent_quantity=Decimal("1"),
        debit_credit=DebitCredit.DEBIT,
        net_limit_price=None,
        maximum_loss=Decimal("100"),
        buying_power_required=Decimal("100"),
        time_in_force="day",
        order_type="market",
        idempotency_key=f"task-7-{uuid4()}",
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    artifacts = tuple(
        HashedDecisionArtifact.from_payload(
            stage,
            1,
            {"stage": stage, "status": "RUN", "test": True},
        )
        for stage in (
            "RAW_STRATEGY_OUTPUT",
            "EXPOSURE_REPORT",
            "INSTRUMENT_CANDIDATES",
            "INSTRUMENT_SELECTION",
        )
    )
    record = DecisionRecord(
        decision_record_id=record_id,
        decision_time=now,
        snapshot_id=snapshot.snapshot_id,
        intent_id=intent.intent_id,
        policy_decision_id=policy.policy_decision_id,
        order_request_ids=(order.order_request_id,),
        source_manifest_hash="1" * 64,
        snapshot_hash=snapshot.content_hash,
        intent_hash=content_hash(intent, exclude={"intent_id"}),
        policy_hash=content_hash(policy, exclude={"policy_decision_id"}),
        raw_strategy_output=artifacts[0],
        exposure_report=artifacts[1],
        instrument_candidates=artifacts[2],
        instrument_selection=artifacts[3],
        order_hashes=(order.request_hash,),
        config_hash="2" * 64,
        model_versions={"meta": "meta@1"},
        feature_versions={"matrix": "matrix@1"},
        schema_version=1,
    )
    return CompleteDecisionChain(
        snapshot=snapshot,
        intent=intent,
        policy_decision=policy,
        record=record,
        order_requests=(order,),
    )
