"""An unreachable governed path queues the plan; it never loses it.

Regression for 2026-08-20 14:20 ET. The Meta runner built a nine-order plan,
called `_submit_via_gateway`, and died with SystemExit out of the gateway's
`_build_snapshots` when the nervous-system Postgres refused the connection.
Everything after the submit call in `_execute` — the order-plan audit append,
`state["managed"] = new_managed`, `_save_state`, and the deferral files — was
therefore skipped, so the plan left no trace at all: the audit log has no
`order_plan` row for that bar, and nothing was queued for retry.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import core.live_4h_exec as exec_mod
import signals.meta_context.meta_ranker.live_runner as lr
from signals.meta_context.meta_ranker.gateway_execution import GovernedPathUnavailable

BAR = pd.Timestamp("2026-08-20 14:00:00+00:00")


def _args(tmp_path):
    return SimpleNamespace(
        submit=True,
        mode="options",
        signal_audit_log=str(tmp_path / "audit.jsonl"),
        matrix="signals/meta_context/meta_ranker/meta_ranker_matrix.parquet",
    )


@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    """Point every pending-queue path at a temp dir, and neutralise the
    readiness gate.

    The gate reads Data/readiness/latest_success.json from the repo root. That
    file exists in a live working tree and does NOT exist in a fresh checkout,
    so without this these tests passed locally and failed in any clean worktree
    — every order was gated out before reaching the code under test, which is
    deferral and audit behaviour, not readiness.
    """

    monkeypatch.setattr(
        exec_mod, "pending_open_path",
        lambda module, root=None: tmp_path / f"{module}_pending_open.json",
    )
    monkeypatch.setattr(
        exec_mod, "pending_exit_path",
        lambda module, root=None: tmp_path / f"{module}_pending_exit.json",
    )
    monkeypatch.setattr(
        lr, "filter_entry_orders_for_readiness",
        lambda plan, **_kwargs: (plan, [], "readiness stubbed for this test"),
    )
    return tmp_path


def test_governed_path_outage_queues_entries_and_writes_audit(
    tmp_path, monkeypatch, isolated_ledger
):
    saved = {}
    monkeypatch.setattr(lr, "_save_state", lambda state: saved.update(state))
    # Market open, so the plan would otherwise be submitted right now.
    monkeypatch.setattr(exec_mod, "is_market_open_now", lambda now=None: True, raising=False)
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)

    def boom(*_args, **_kwargs):
        raise GovernedPathUnavailable("context snapshot unavailable for CRWD: OperationalError")

    monkeypatch.setattr(lr, "_submit_via_gateway", boom)

    plan = [
        ("CRWD", "buy", 26, "entry", "equity"),
        ("PSIG", "buy", 1845, "entry", "equity"),
    ]
    new_managed = {
        "CRWD": {"route": "equity", "symbol": "CRWD", "shares": 26},
        "PSIG": {"route": "equity", "symbol": "PSIG", "shares": 1845},
    }
    state = {"managed": {}, "history": []}

    # The call must return normally rather than propagating.
    lr._execute(
        _args(tmp_path), client=None, plan=list(plan), state=state,
        new_managed=new_managed, bar=BAR, targets=["CRWD", "PSIG"],
        is_option=False, module="meta_ranker",
    )

    # 1. The plan was queued for the next flush, not lost.
    queued = json.loads((isolated_ledger / "meta_ranker_pending_open.json").read_text())
    assert {e["order_symbol"] for e in queued["entries"]} == {"CRWD", "PSIG"}

    # 2. Managed state was saved, and the unplaced entries were pruned from it —
    #    nothing reached the broker, so nothing may be claimed as held.
    assert saved.get("managed") == {}

    # 3. The audit records the bar. The 8/20 failure's signature was an audit
    #    log with no row at all for the bar that produced the plan.
    rows = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [r["event"] for r in rows] == ["order_plan"]
    assert rows[0]["targets"] == ["CRWD", "PSIG"]


def test_snapshot_failure_is_raised_as_governed_path_unavailable(monkeypatch):
    """A store outage surfaces as the one exception the runner contains."""

    from signals.meta_context.meta_ranker.gateway_execution import MetaGatewayRouter
    from signals.meta_context.meta_ranker.nervous_system_adapter import MetaIntentConfig

    class DeadStore:
        def build(self, **_kwargs):
            raise RuntimeError("connection to server at 127.0.0.1, port 55432 failed")

    router = MetaGatewayRouter.__new__(MetaGatewayRouter)
    router._snapshots = DeadStore()
    router._intent_config = MetaIntentConfig(strategy_id="meta_ranker")
    router._profile = None

    with pytest.raises(GovernedPathUnavailable) as excinfo:
        router._build_snapshots(
            [("CRWD", "buy", 26, "entry", "equity")],
            ticker_by_symbol={"CRWD": "CRWD"},
            decision_bar=BAR.to_pydatetime(),
            now=BAR.to_pydatetime(),
        )
    # Names the ticker and the failure class, not the connection string.
    assert "CRWD" in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)
    assert "55432" not in str(excinfo.value)


def test_audit_records_the_planned_rows_and_their_disposition(
    tmp_path, monkeypatch, isolated_ledger
):
    """`plan` is the residue; `planned` is the decision.

    Both the 2026-08-20 and 2026-08-21 16:20 ET runs logged `plan: []` while
    deferring 5 and 8 orders respectively, so the audit said the module had
    decided nothing on bars where it had decided a great deal.
    """
    monkeypatch.setattr(lr, "_save_state", lambda state: None)
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)

    plan = [
        ("CRWD", "buy", 26, "entry", "equity"),
        ("PSIG", "buy", 1845, "entry", "equity"),
    ]
    new_managed = {
        "CRWD": {"route": "equity", "symbol": "CRWD", "shares": 26},
        "PSIG": {"route": "equity", "symbol": "PSIG", "shares": 1845},
    }

    lr._execute(
        _args(tmp_path), client=None, plan=list(plan), state={"managed": {}, "history": []},
        new_managed=new_managed, bar=BAR, targets=["CRWD", "PSIG"],
        is_option=False, module="meta_ranker",
    )

    row = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert row["plan"] == []          # nothing was submitted, as before
    assert {p["symbol"] for p in row["planned"]} == {"CRWD", "PSIG"}
    assert {p["disposition"] for p in row["planned"]} == {"deferred_entry_market_closed"}


def test_a_submitted_row_is_recorded_as_submitted(tmp_path, monkeypatch, isolated_ledger):
    monkeypatch.setattr(lr, "_save_state", lambda state: None)
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)
    monkeypatch.setattr(lr, "_submit_via_gateway", lambda *a, **k: None)

    lr._execute(
        _args(tmp_path), client=None,
        plan=[("CRWD", "buy", 26, "entry", "equity")],
        state={"managed": {}, "history": []},
        new_managed={"CRWD": {"route": "equity", "symbol": "CRWD", "shares": 26}},
        bar=BAR, targets=["CRWD"], is_option=False, module="meta_ranker",
    )

    row = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert [p["disposition"] for p in row["planned"]] == ["submitted"]
