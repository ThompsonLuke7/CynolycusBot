"""Offline second-review evidence, NOT production regression tests or fixes.

Observation tests intentionally describe current behavior, including defects.
Control tests delimit each claim; prototype tests check isolated proposed math.
All mutable state is redirected to pytest temporary directories. No real broker.
Run explicitly: .venv/bin/python -m pytest -q research/system_review_2026-09-16/test_reassessment.py
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "prior_review", REPO / "research/system_review_2026-09-13/reproduce_findings.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
execution, risk = prior.execution, prior.risk


def extract(path, names, namespace, transform=None):
    """Use actual function bodies without importing training/runner bootstraps."""
    tree = ast.parse((REPO / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=nodes, type_ignores=[])
    if transform:
        module = transform.visit(module)
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("status,filled,price,owned,booked", [
    ("accepted", 0, None, True, None),
    ("filled", 10, "2.0", False, 1000),
    ("partially_filled", 2, "2.0", False, 1000),  # defective: should own 8, book 200
])
def test_f01_full_partial_and_unfilled_control(tmp_path, monkeypatch, status, filled, price, owned, booked):
    broker = prior.FakeBroker(status=status, filled_qty=filled, fill=price)
    managed = {}
    monkeypatch.setattr(execution, "defer_exits_if_opg_unavailable", lambda m, b, p, l, **kw: p)
    execution.execute_plan(
        broker, plan=[(prior.OCC, "sell", 10, "horizon", "option")], limits={},
        submit=True, equity_tif_fn=lambda: "day", new_managed=managed,
        exit_context={prior.OCC: ("ABC", prior.state())}, module="fills",
        pos_lookup={prior.OCC: {"avg_entry": 1.}}, bar=prior.NOW, ledger_root=str(tmp_path))
    assert bool(managed) is owned
    rows = prior.rows(tmp_path, "fills")
    assert ([r["realized_pnl"] for r in rows] if rows else []) == ([] if booked is None else [booked])


@pytest.mark.parametrize("outcome,owned,rows_expected", [
    ("SUBMITTED", False, 1), ("REJECTED", True, 0), ("AMBIGUOUS", True, 0),
])
def test_f02_actual_meta_callback_and_refusal_controls(tmp_path, outcome, owned, rows_expected):
    from core.nervous_system.execution.gateway import ExecutionOutcome

    result = SimpleNamespace(outcome=ExecutionOutcome[outcome], broker_order_id="oid")
    row = SimpleNamespace(symbol="ABC", side="sell", quantity=10, refusal=None,
                          outcome=SimpleNamespace(execution_result=result),
                          intent=SimpleNamespace(reason_codes=["horizon"]))
    router = SimpleNamespace(route=lambda *a, **kw: kw["on_row"](row))
    ns = extract("signals/meta_context/meta_ranker/live_runner.py",
                 {"gateway_verdict", "_submit_via_gateway"}, {
        "build_router": lambda **kw: router, "intent_config": lambda args: None,
        "logger": logging.getLogger("review"), "PolicyMode": SimpleNamespace(ENFORCE="enforce"),
        "_save_state": lambda state: None,
        "record_exit_realized_pnl": lambda *a, **kw: execution.record_exit_realized_pnl(
            *a, **kw, ledger_root=str(tmp_path)),
    })
    managed = {}
    ns["_submit_via_gateway"](
        None, [("ABC", "sell", 10, "horizon", "equity")], {}, managed, prior.NOW,
        is_option=False, limits={}, exit_context={"ABC": ("ABC", {"route": "equity"})},
        module="meta", pos_lookup={"ABC": {"avg_entry": 100}}, client=prior.FakeBroker())
    assert bool(managed) is owned
    rows = prior.rows(tmp_path, "meta")
    assert len(rows) == rows_expected
    if rows:
        assert rows[0]["realized_pnl"] is None


def test_f02_deferred_owner_survives_but_order_tracking_does_not(tmp_path):
    assert prior.deferred_acceptance(tmp_path)["pending_order_tracked"] is False


def test_f02_intraday_claim_really_is_removed(tmp_path):
    assert prior.intraday_acceptance(tmp_path)["owner_removed"] is True


def test_f02_htf_independent_submit_loop_has_same_gap(tmp_path):
    broker = prior.FakeBroker()
    managed = {}
    ns = extract("strategies/multi_ticker_swing_htf/live/runner.py", {"_execute"}, {
        "AUDIT_MODULE": "htf", "STATE_PATH": tmp_path / "state.json",
        "init_dispositions": execution.init_dispositions,
        "mark_plan_gone": execution.mark_plan_gone,
        "defer_entries_if_market_closed": lambda m, b, p, nm, lim: p,
        "defer_exits_if_opg_unavailable": lambda m, b, p, lim, **kw: p,
        "filter_entry_orders_for_readiness": lambda p, **kw: (p, [], ""),
        "submit_option_exit_with_ladder": execution.submit_option_exit_with_ladder,
        "record_exit_realized_pnl": lambda *a, **kw: execution.record_exit_realized_pnl(
            *a, **kw, ledger_root=str(tmp_path)),
        "_save_state": lambda _: None, "_append_order_plan_audit": lambda *a, **kw: None,
        "_now": lambda: prior.NOW.isoformat(), "logger": logging.getLogger("review"),
    })
    ns["_execute"](
        SimpleNamespace(submit=True, mode="options"), broker,
        [(prior.OCC, "sell", 10, "horizon", "option")], {}, managed, prior.NOW, [],
        is_option=True, exit_context={prior.OCC: ("ABC", prior.state())},
        pos_lookup={prior.OCC: {"avg_entry": 1.}})
    assert managed == {}
    assert prior.rows(tmp_path, "htf")[0]["realized_pnl"] is None


@pytest.mark.parametrize("price,expected_exit", [(100., False), (90., True), (None, True)])
def test_f03_current_price_vs_unavailable_premium_fallback(tmp_path, price, expected_exit):
    res = risk.evaluate_risk_exits(
        prior.FakeBroker(), module="risk", managed={"ABC": prior.state()},
        pos_info={prior.OCC: {"qty": 10, "avg_entry": 1., "current": .2}},
        policy=execution.ExecPolicy(), now_et=prior.NOW,
        underlying_fn=lambda _: (price, 2.), ledger_root=str(tmp_path))
    assert bool(res.plan) is expected_exit


def test_f03_default_accepts_old_observation(tmp_path):
    assert prior.stale_risk(tmp_path)["cached_orders"] == 0


@pytest.mark.parametrize("status", ["accepted", "partially_filled", "canceled", "expired"])
def test_f04_risk_pass_does_not_distinguish_live_and_dead_orders(tmp_path, status):
    class Broker(prior.FakeBroker):
        reads = 0

        def get_order(self, oid):
            self.reads += 1
            return {"id": oid, "status": status, "filled_qty": "0"}

    client = Broker()
    st = prior.state()
    st["exit_pending"] = {"order_id": "oid"}
    res = risk.evaluate_risk_exits(
        client, module="risk", managed={"ABC": st},
        pos_info={prior.OCC: {"qty": 10, "avg_entry": 1., "current": .2}},
        policy=execution.ExecPolicy(), now_et=prior.NOW,
        underlying_fn=lambda _: (90., 2.), ledger_root=str(tmp_path))
    assert client.reads == 0 and res.plan == []


def test_f04_existing_slow_helper_is_not_a_safe_drop_in():
    class Broker:
        def get_order(self, oid):
            raise TimeoutError("offline")
    assert execution._order_is_working(Broker(), "oid") is False


def test_f05_actual_writer_ignores_lock_while_cooperating_writer_is_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(risk, "LOCK_ROOT", tmp_path / "locks")
    with risk.module_state_lock("control") as first:
        with risk.module_state_lock("control") as second:
            assert first and not second
    assert prior.unshared_lock(tmp_path)["new_position_lost_by_stale_save"]


@pytest.mark.parametrize("mode,expected_orders", [("filled", 1), ("cancel_timeout", 3), ("pending_cancel", 3)])
def test_f06_first_fill_control_and_unconfirmed_cancel(tmp_path, mode, expected_orders):
    class Broker(prior.FakeBroker):
        def get_option_quotes(self, symbols=None, **kw):
            return {"quotes": {symbols: {"bp": 1., "ap": 2.}}}

        def cancel_order(self, oid):
            if mode == "cancel_timeout":
                raise TimeoutError("offline")
            return {"status": "pending_cancel"}

    broker = Broker(status="filled" if mode == "filled" else "accepted")
    execution.submit_option_entry_with_ladder(
        broker, symbol=prior.OCC, qty=10, ask=2., sleep_fn=lambda _: None,
        poll_fn=lambda resp: resp["status"] == "filled")
    assert len(broker.orders) == expected_orders


def test_f06_confirmed_cancel_still_requires_residual_sizing():
    class Broker(prior.FakeBroker):
        def get_option_quotes(self, symbols=None, **kw):
            return {"quotes": {symbols: {"bp": 1., "ap": 2.}}}

        def cancel_order(self, oid):
            return {"id": oid, "status": "canceled", "filled_qty": "2"}

    broker = Broker(status="partially_filled", filled_qty=2, fill="1.5")
    execution.submit_option_entry_with_ladder(
        broker, symbol=prior.OCC, qty=10, ask=2., sleep_fn=lambda _: None,
        poll_fn=lambda _: False)
    assert [o["qty"] for o in broker.orders] == [10, 10, 10]  # intended: 10, 8, 6


def test_f07_missing_equity_order_does_not_prove_loss(tmp_path):
    assert prior.missing_order_loss(tmp_path)["invented_pnl"] == -1000


def test_f07_returning_unresolved_alone_would_still_lose_slow_runner_state(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "resolve_settled_exit", lambda *a, **kw: False)
    monkeypatch.setattr(risk, "resolve_settled_exit", lambda *a, **kw: False)
    st = prior.state()
    st["exit_pending"] = {"order_id": "oid"}
    slow = execution.build_mixed_plan(
        prior.FakeBroker(), targets=[], managed={"ABC": copy.deepcopy(st)}, pos_info={},
        bar=prior.NOW, signal_audits={}, policy=execution.ExecPolicy(),
        route_fn=lambda *a, **kw: None, ref_price_fn=lambda _: None,
        module="review", ledger_root=str(tmp_path), verbose=False)
    fast = risk.evaluate_risk_exits(
        prior.FakeBroker(), module="review", managed={"ABC": copy.deepcopy(st)},
        pos_info={}, policy=execution.ExecPolicy(), now_et=prior.NOW, ledger_root=str(tmp_path))
    assert slow.new_managed == {} and "ABC" in fast.new_managed


def test_f08_old_harness_overlap_is_not_new_harness_overlap(tmp_path):
    assert prior.embargo_overlap(tmp_path)["actual_boundary_bar_distance"] < 25
    ns = extract("scripts/horizon_thesis/run_horizon_grid.py", {"split_bars"},
                 {"np": np, "pd": pd, "EMBARGO_BARS": 70})
    bars = pd.date_range("2020-01-01", periods=1000, freq="12h", tz="UTC")
    # Inspect the newer helper's actual masks rather than assuming its docstring.
    frame = pd.DataFrame({"x": 1}, index=pd.MultiIndex.from_product([bars, ["ABC"]]))
    result = ns["split_bars"](frame)
    tr, va, te = result[:3]
    assert bars.get_loc(va.index[0][0]) - bars.get_loc(tr.index[-1][0]) >= 70
    assert bars.get_loc(te.index[0][0]) - bars.get_loc(va.index[-1][0]) >= 70


class RemoveLeadingForwardShift(ast.NodeTransformer):
    """Isolated F09 prototype; leaves the repository builder untouched."""
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "shift"
                and len(node.args) == 1 and ast.dump(node.args[0]) == ast.dump(ast.parse("-1", mode="eval").body)):
            return node.func.value
        return node


def test_f09_window_correction_prototype_and_future_perturbation(tmp_path):
    daily = pd.DataFrame({"timestamp": pd.bdate_range("2025-01-01", periods=210, tz="UTC")
                        + pd.Timedelta(hours=14), "open": 100., "high": 101.,
                        "low": 99., "close": 100., "volume": np.arange(210) + 1000.})
    path = tmp_path / "ABC.parquet"
    daily.to_parquet(path)
    ns = {"pd": pd, "np": np, "BARS_1D": tmp_path, "MAX_H": 30,
          "HOLDS": [5, 10, 15, 20, 30], "CHECKPOINT": 5, "BETA_WINDOW": 60}
    original = extract("scripts/horizon_thesis/build_decision_panel.py", {"ticker_frame"}, ns.copy())
    fixed = extract("scripts/horizon_thesis/build_decision_panel.py", {"ticker_frame"}, ns.copy(),
                    RemoveLeadingForwardShift())
    before, fixed_before = original["ticker_frame"]("ABC"), fixed["ticker_frame"]("ABC")
    daily.loc[105, ["volume", "close"]] = [1_000_000., 200.]
    daily.to_parquet(path)
    after, fixed_after = original["ticker_frame"]("ABC"), fixed["ticker_frame"]("ABC")
    for col in ("cp_volume_ratio", "cp_up_day_share"):
        assert before.loc[100, col] != after.loc[100, col]
        assert fixed_before.loc[100, col] == fixed_after.loc[100, col]
    assert fixed_before.loc[100, "cp_volume_ratio"] == pytest.approx(
        daily.loc[100:104, "volume"].mean() / daily.loc[80:99, "volume"].mean())
    for col in ("cp_ret", "cp_mfe", "cp_mae", "fwdret_5", "fwdret_30"):
        pd.testing.assert_series_equal(before[col], fixed_before[col])


@pytest.mark.parametrize("path", [
    "strategies/model_training/colab_competition.py",
    "signals/meta_context/meta_ranker/colab_competition.py",
    "strategies/multi_ticker_swing_htf/data/training_export/colab_competition.py",
])
def test_f10_canonical_and_exported_selection_respond_to_test_labels(path):
    ns = extract(path, {"primary_metric_name", "choose_best"}, {"pd": pd})
    df = pd.DataFrame({"family": ["A", "B"], "val_ndcg_at_10": [.9, .1],
                       "test_ndcg_at_10": [.8, .2]})
    first = ns["choose_best"](df)["family"]
    df["test_ndcg_at_10"] = [.2, .8]
    assert ns["choose_best"](df)["family"] != first
    assert df.sort_values("val_ndcg_at_10", ascending=False).iloc[0]["family"] == "A"


def test_f11_local_artifact_family_and_provenance_limit():
    import pyarrow.parquet as pq
    root = REPO / "strategies/multi_ticker_swing_htf/models"
    meta = json.loads((root / "eval_metrics.json").read_text())
    native = json.loads((root / "htf_swing_xgb.json").read_text())
    assert meta["winner_family"] == "lgbm_classifier"
    assert native["learner"]["objective"]["name"] == "binary:logistic"
    # No self-identifying family/seed in this OOF: historical provenance is inferred,
    # not independently authenticated from the parquet itself.
    assert not {"family", "seed", "model_family"} & set(pq.ParquetFile(root / "oof_preds.parquet").schema.names)
