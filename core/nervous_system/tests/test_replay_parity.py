"""Replay determinism, live/replay parity, and the no-network guarantee.

Three acceptance properties from Task 24, each of which fails silently if it
is not tested.

*Determinism.* A replay whose answer depends on input order, or on the wall
clock, cannot be used to evaluate anything — two runs disagreeing is
indistinguishable from a real change in behaviour.

*Parity.* Replay is only evidence about live if both build the same
pre-execution artifacts from the same inputs. Parity is asserted on hashes
rather than on eyeballed numbers.

*No network.* A "replay" that reaches a live API is not a replay. This is
asserted statically, because a runtime test only proves the paths it happened
to execute.
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.replay import (
    MarkType,
    Observation,
    ObservationKind,
    SideFitnessMetrics,
    SourceFitnessThresholds,
)
from core.nervous_system.replay.evidence import select_causal_observations
from core.nervous_system.replay.fitness import evaluate_source_fitness
from core.nervous_system.replay.provider import ReplayEvidenceProvider


REPLAY_PACKAGE = Path(__file__).resolve().parents[1] / "replay"
BAR = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
DECIDE = BAR + timedelta(minutes=5)


def _obs(name: str, **updates) -> Observation:
    payload = {
        "observation_id": uuid5(NAMESPACE_URL, name),
        "kind": ObservationKind.BAR,
        "instrument": "AMD",
        "as_of": BAR,
        "available_at": BAR + timedelta(minutes=1),
        "valid_until": BAR + timedelta(days=30),
        "generated_at": BAR + timedelta(minutes=1),
        "artifact_hash": "a" * 64,
        "record_locator": f"locator/{name}",
        "provider": "alpaca",
        "feed": "sip",
        "tier": "verified",
        "schema_version": 1,
        "producer": "shared_bars",
        "mark_type": MarkType.QUOTE_BID_ASK,
        "bar_bound": True,
    }
    payload.update(updates)
    return Observation(**payload)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_a_repeat_run_selects_byte_identical_evidence() -> None:
    corpus = [_obs("a"), _obs("b", available_at=BAR + timedelta(minutes=2))]

    first = select_causal_observations(corpus, decision_time=DECIDE, decision_bar=BAR)
    second = select_causal_observations(corpus, decision_time=DECIDE, decision_bar=BAR)

    assert [content_hash(o) for o in first] == [content_hash(o) for o in second]


def test_input_permutation_does_not_change_the_selection() -> None:
    """Replay must not depend on the order a provider happened to yield rows."""

    a, b, c = _obs("a"), _obs("b", available_at=BAR + timedelta(minutes=2)), _obs("c")

    forward = select_causal_observations([a, b, c], decision_time=DECIDE, decision_bar=BAR)
    reverse = select_causal_observations([c, b, a], decision_time=DECIDE, decision_bar=BAR)

    assert [content_hash(o) for o in forward] == [content_hash(o) for o in reverse]


def test_a_fitness_verdict_is_reproducible_from_its_inputs() -> None:
    def _report():
        return evaluate_source_fitness(
            sides=(
                SideFitnessMetrics(
                    option_type="CALL", mark_type=MarkType.QUOTE_BID_ASK,
                    matched_positions=40, sessions=15,
                    valid_quote_fraction=Decimal("0.99"),
                    identical_mark_fraction=Decimal("0.01"),
                    pearson=Decimal("0.91"), spearman=Decimal("0.88"),
                    max_quote_age_seconds=Decimal("30"), entitlement_verified=True,
                ),
            ),
            thresholds=SourceFitnessThresholds(),
            source="alpaca", feed="opra", tier="indicative",
        )

    assert content_hash(_report()) == content_hash(_report())


def test_appending_later_evidence_cannot_change_an_earlier_decision() -> None:
    """Future-append invariance: re-running an old decision after more data
    arrived must reproduce exactly what it produced at the time.
    """

    known = _obs("known")
    later = _obs(
        "later",
        available_at=DECIDE + timedelta(days=1),
        valid_until=DECIDE + timedelta(days=30),
    )

    before = ReplayEvidenceProvider([known]).bars(decision_time=DECIDE, decision_bar=BAR)
    after = ReplayEvidenceProvider([known, later]).bars(
        decision_time=DECIDE, decision_bar=BAR
    )

    assert [content_hash(o) for o in before] == [content_hash(o) for o in after]


# ---------------------------------------------------------------------------
# Live / replay parity
# ---------------------------------------------------------------------------


def test_live_and_replay_agree_on_the_pre_execution_hash() -> None:
    """Parity is asserted on the artifact hash, not on inspected numbers: two
    pipelines that merely look similar are not evidence about each other.
    """

    corpus = [_obs("a"), _obs("b", available_at=BAR + timedelta(minutes=2))]

    # "Live" saw evidence arrive one at a time, in arrival order.
    live = select_causal_observations(
        sorted(corpus, key=lambda o: o.available_at),
        decision_time=DECIDE, decision_bar=BAR,
    )
    # "Replay" reads the same evidence back from storage, in whatever order.
    replay = ReplayEvidenceProvider(list(reversed(corpus))).bars(
        decision_time=DECIDE, decision_bar=BAR
    )

    assert content_hash_sequence(live) == content_hash_sequence(replay)


def content_hash_sequence(observations) -> str:
    return "|".join(content_hash(observation) for observation in observations)


def test_parity_breaks_visibly_when_the_evidence_differs() -> None:
    """A parity check that cannot fail proves nothing."""

    live = [_obs("a")]
    replay = [_obs("a", record_locator="somewhere/else")]

    assert content_hash_sequence(live) != content_hash_sequence(replay)


# ---------------------------------------------------------------------------
# No network, no broker, no credentials
# ---------------------------------------------------------------------------


FORBIDDEN_IMPORTS = {
    "requests", "httpx", "urllib", "urllib3", "socket", "http",
    "alpaca", "boto3", "google",
}

FORBIDDEN_NAMES = {
    "AlpacaOptionsClient", "AlpacaPaperAdapter", "ExecutionGateway",
}


def _replay_modules() -> list[Path]:
    return sorted(REPLAY_PACKAGE.glob("*.py"))


def test_the_replay_package_has_modules_to_scan() -> None:
    """Guards the scans below: an empty glob would pass them vacuously."""

    assert len(_replay_modules()) >= 4


@pytest.mark.parametrize("path", _replay_modules(), ids=lambda p: p.name)
def test_replay_code_imports_no_network_or_broker_library(path: Path) -> None:
    """Asserted statically. A runtime test only proves the paths it executed;
    a source scan proves the capability is absent.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not (imported & FORBIDDEN_IMPORTS), f"{path.name} reaches the network"


@pytest.mark.parametrize("path", _replay_modules(), ids=lambda p: p.name)
def test_replay_code_constructs_no_broker_or_gateway(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constructed = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert not (constructed & FORBIDDEN_NAMES), f"{path.name} builds a broker path"


@pytest.mark.parametrize("path", _replay_modules(), ids=lambda p: p.name)
def test_replay_code_never_reads_a_wall_clock(path: Path) -> None:
    """A replay that reads the clock is not reproducible: the same inputs give
    a different answer tomorrow.
    """

    source = path.read_text(encoding="utf-8")

    assert "datetime.now(" not in source, f"{path.name} reads the wall clock"
    assert "time.time(" not in source, f"{path.name} reads the wall clock"


@pytest.mark.parametrize("path", _replay_modules(), ids=lambda p: p.name)
def test_replay_code_reads_no_credentials(path: Path) -> None:
    source = path.read_text(encoding="utf-8")

    assert "os.environ" not in source, f"{path.name} reads the environment"
    assert "getenv" not in source, f"{path.name} reads the environment"
