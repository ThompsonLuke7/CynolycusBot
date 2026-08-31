"""MVP acceptance (Task 27).

What this file asserts is narrow on purpose: the properties that must hold
before anyone considers pointing this at money, and nothing more. Where a
requirement can only be met by operating the system over time — a shadow soak,
a controlled paper-submit subset — this file says so rather than simulating it
and calling the simulation evidence.

The direct-submit boundary is scoped to Meta. Other strategies still submit
directly and are inventoried here as post-MVP migrations. Claiming a
repository-wide cutover would be false.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.nervous_system.config.runtime import (
    PRODUCTION_LIVE_VETO,
    NervousSystemSettings,
)
from core.nervous_system.contracts.enums import RuntimeEnvironment


REPO = Path(__file__).resolve().parents[3]

SUBMIT_CALLS = {"submit_order", "submit_option_order", "submit_multileg_order"}

# The only place a POST to a broker may originate. Everything else is either
# governed or inventoried below.
INWARD_ADAPTER = "core/nervous_system/execution/alpaca_adapter.py"
BROKER_CLIENT = "core/API/Alpaca_API/options/options_api.py"

# Legacy bypasses, tracked deliberately. These are NOT part of the MVP cutover;
# they are the post-MVP migration list, and pinning the counts means a new one
# cannot appear unnoticed.
LEGACY_DIRECT_SUBMIT = {
    # 9, not 6: main added an option ENTRY ladder (2 sites, Dealer Ranker only)
    # and a pre-open exit flush (1 site) while this branch was open. See
    # test_meta_no_bypass for the per-function breakdown and for which of these
    # Meta can still reach (none).
    "core/live_4h_exec.py": 9,
    "core/startup_queue.py": 1,
    "core/API/Alpaca_API/cli/options_cli.py": 1,
    "strategies/spy_intraday/Policy/order_policy.py": 3,
    "strategies/multi_ticker_swing/live/position_manager.py": 3,
    "strategies/multi_ticker_swing/live/runner.py": 1,
    "strategies/multi_ticker_swing_htf/live/runner.py": 2,
    "strategies/momentum_expansion/policy/momentum_option_policy.py": 2,
    "strategies/dealer_positioning/execution.py": 2,
    # Added 2026-08-29 with the intraday structure engine's first real paper
    # orders. Deliberately on the legacy path, not the governed one: the
    # nervous system is wired to Meta only, and routing a brand-new execution
    # surface through a gateway no other 4H module uses would couple this
    # module's first live data to that migration. It is paper-only by config
    # (`load_config` raises otherwise), capped at a few concurrent positions,
    # and belongs on the post-MVP migration list alongside the other four.
    "strategies/intraday_structure/execution.py": 2,
}

# Meta's own surface, which must be zero.
META_OWNED = (
    "signals/meta_context/meta_ranker/live_runner.py",
    "signals/meta_context/meta_ranker/options_exec.py",
    "signals/meta_context/meta_ranker/gateway_execution.py",
    "signals/meta_context/meta_ranker/nervous_system_adapter.py",
    "UI/meta_ranker_dashboard.py",
)


def _submit_calls(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) in SUBMIT_CALLS
    )


# ---------------------------------------------------------------------------
# The direct-submit boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relative", META_OWNED)
def test_meta_owns_no_direct_broker_submission(relative: str) -> None:
    """The MVP claim, and the whole point of the cutover."""

    assert _submit_calls(REPO / relative) == 0


@pytest.mark.parametrize("relative", sorted(LEGACY_DIRECT_SUBMIT))
def test_legacy_bypasses_are_inventoried_not_forgotten(relative: str) -> None:
    """These are post-MVP migrations. Pinning the count means a new bypass
    cannot appear without this failing, and it keeps the MVP claim honest:
    Meta is cut over, the repository is not.
    """

    assert _submit_calls(REPO / relative) == LEGACY_DIRECT_SUBMIT[relative]


def test_no_unknown_file_submits_directly() -> None:
    """Catches a bypass appearing somewhere nobody thought to look."""

    known = set(LEGACY_DIRECT_SUBMIT) | {INWARD_ADAPTER, BROKER_CLIENT}
    offenders = {}
    for root in ("core", "signals", "strategies", "UI", "scripts"):
        for path in (REPO / root).rglob("*.py"):
            if "test" in path.parts or path.name.startswith("test_"):
                continue
            relative = path.relative_to(REPO).as_posix()
            if relative in known:
                continue
            count = _submit_calls(path)
            if count:
                offenders[relative] = count

    assert offenders == {}, f"unexpected direct submission: {offenders}"


def test_only_the_inward_adapter_posts_on_behalf_of_the_nervous_system() -> None:
    nervous_system = REPO / "core/nervous_system"
    submitters = {
        path.relative_to(REPO).as_posix()
        for path in nervous_system.rglob("*.py")
        if "tests" not in path.parts and _submit_calls(path)
    }

    assert submitters == {INWARD_ADAPTER}


# ---------------------------------------------------------------------------
# The environment x mode x submit veto matrix
# ---------------------------------------------------------------------------


def _env(**updates: str) -> dict[str, str]:
    payload = {
        "CYNOLYCUS_ENVIRONMENT": "QA_PAPER",
        "CYNOLYCUS_NERVOUS_SYSTEM_MODE": "SHADOW",
        "CYNOLYCUS_DATABASE_URL": (
            "postgresql+psycopg://u:p@/cynolycus?host=/cloudsql/p:us-east5:i"
        ),
        "CYNOLYCUS_OPERATIONAL_ROOT": "/tmp/cynolycus",
        "CYNOLYCUS_EXECUTION_JOURNAL": "gcs",
        "CYNOLYCUS_EXECUTION_JOURNAL_BUCKET": "b",
        "CYNOLYCUS_ACCOUNT_ALIAS": "paper",
        "CYNOLYCUS_GCP_PROJECT": "p",
        "CYNOLYCUS_CLOUD_SQL_INSTANCE": "p:us-east5:i",
        "CYNOLYCUS_ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        "CYNOLYCUS_ALPACA_ACCOUNT_ID": "PA1",
        "CYNOLYCUS_SECRET_BINDING": "projects/p/secrets/s",
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize("mode", ["OFF", "SHADOW", "ENFORCE"])
@pytest.mark.parametrize("submit", ["true", "false"])
def test_production_live_never_submits_in_any_mode(mode: str, submit: str) -> None:
    """Six cells of the matrix, and the answer is the same in all of them."""

    settings = NervousSystemSettings.from_env(
        _env(
            CYNOLYCUS_ENVIRONMENT="PRODUCTION_LIVE",
            CYNOLYCUS_ACCOUNT_ALIAS="live",
            CYNOLYCUS_NERVOUS_SYSTEM_MODE=mode,
            CYNOLYCUS_SUBMIT_ENABLED=submit,
        )
    )

    assert settings.submit_enabled is False
    assert settings.execution_veto() == PRODUCTION_LIVE_VETO


@pytest.mark.parametrize("mode", ["OFF", "SHADOW", "ENFORCE"])
def test_qa_paper_does_not_submit_without_explicit_enablement(mode: str) -> None:
    settings = NervousSystemSettings.from_env(
        _env(CYNOLYCUS_NERVOUS_SYSTEM_MODE=mode)
    )

    assert settings.submit_enabled is False
    assert settings.execution_veto() is None


def test_qa_paper_submits_only_when_told_to_explicitly() -> None:
    settings = NervousSystemSettings.from_env(
        _env(CYNOLYCUS_NERVOUS_SYSTEM_MODE="ENFORCE", CYNOLYCUS_SUBMIT_ENABLED="true")
    )

    assert settings.submit_enabled is True


def test_the_gateway_refuses_to_be_constructed_for_production_live() -> None:
    """Belt and braces: even handed a live environment directly, the gateway
    will not exist to be called.
    """

    from core.nervous_system.execution.gateway import (
        ExecutionGateway,
        GatewayPreflightError,
    )

    with pytest.raises(GatewayPreflightError):
        ExecutionGateway(
            broker=object(), journal=object(), unit_of_work_factory=lambda: None,
            environment=RuntimeEnvironment.PRODUCTION_LIVE,
            account_alias="live", worker_id="w",
        )


def test_the_meta_router_refuses_production_live_before_building_anything() -> None:
    from signals.meta_context.meta_ranker.gateway_execution import MetaGatewayRouter

    with pytest.raises(ValueError, match="PRODUCTION_LIVE"):
        MetaGatewayRouter(
            coordinator=object(), snapshot_builder=object(),
            policy_evaluator=lambda *_: None, policy_config=object(),
            freshness_profile=object(),
            environment=RuntimeEnvironment.PRODUCTION_LIVE,
            account_alias="live", intent_config=object(), clock=lambda: None,
        )


# ---------------------------------------------------------------------------
# No live API, credential, or network construction in the governed path
# ---------------------------------------------------------------------------


NETWORK_MODULES = {"requests", "httpx", "urllib3", "socket"}


@pytest.mark.parametrize(
    "relative",
    [
        "core/nervous_system/policy/engine.py",
        "core/nervous_system/portfolio/exposure.py",
        "core/nervous_system/replay/fitness.py",
        "core/nervous_system/replay/attribution.py",
        "core/nervous_system/replay/evidence.py",
        "core/nervous_system/replay/provider.py",
        "core/nervous_system/execution/options/payoff.py",
        "core/nervous_system/execution/options/close_ladder.py",
    ],
)
def test_the_pure_decision_path_reaches_no_network(relative: str) -> None:
    """These modules decide things. A decision that depends on a socket is not
    reproducible, and cannot be replayed.
    """

    tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not (imported & NETWORK_MODULES)


def test_the_policy_engine_reads_no_clock() -> None:
    """Every policy input is passed in, so the same decision replays the same
    way tomorrow.
    """

    source = (REPO / "core/nervous_system/policy/engine.py").read_text()

    assert "datetime.now(" not in source
    assert "time.time(" not in source


# ---------------------------------------------------------------------------
# Acceptance artifacts exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "docs/nervous_system/MVP_ACCEPTANCE.md",
        "docs/nervous_system/OPERATIONS_RUNBOOK.md",
        "scripts/validate_nervous_system_mvp.py",
        "scripts/cloud/nervous_system_db.py",
    ],
)
def test_the_acceptance_artifacts_are_committed(relative: str) -> None:
    assert (REPO / relative).exists(), f"{relative} is missing"


def test_the_acceptance_document_states_what_is_not_proven() -> None:
    """An acceptance document that lists only successes is a sales document.

    Asserted on the section and its contents, not on the phrase appearing
    somewhere: the phrase also occurs in the introduction, so a looser check
    would pass with the section deleted.
    """

    text = (REPO / "docs/nervous_system/MVP_ACCEPTANCE.md").read_text()

    assert "## NOT PROVEN" in text, "the section itself must exist"
    body = text.split("## NOT PROVEN", 1)[1].split("## ", 1)[0]
    for requirement in ("shadow soak", "paper-submit", "entitlement"):
        assert requirement in body.lower(), f"{requirement} is not listed as outstanding"


def test_the_acceptance_document_does_not_claim_a_repository_wide_cutover() -> None:
    """The single easiest way for this document to become false."""

    text = (REPO / "docs/nervous_system/MVP_ACCEPTANCE.md").read_text().lower()

    assert "not a repository-wide one" in text
    for strategy in ("htf", "momentum", "swing", "dealer", "spy"):
        assert strategy in text, f"{strategy} must be named as still bypassing"
