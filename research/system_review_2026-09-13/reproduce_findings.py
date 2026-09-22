"""Offline review demonstrations; asserts observed defects, NOT correct behavior.

Run: .venv/bin/python research/system_review_2026-09-13/reproduce_findings.py
All broker clients are fakes; all mutable artifacts go to a TemporaryDirectory.
No training, network, real orders, or production state writes are performed.
An assertion failure after a fix means the corresponding demonstration is stale.
"""
from __future__ import annotations

import ast
import copy
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import core.live_4h_exec as execution
import core.live_risk_pass as risk
from strategies.intraday_structure.config import ExecutionPolicy
from strategies.intraday_structure.execution import IntradayOptionExecutor

NOW = datetime(2026, 9, 11, 11, 0, tzinfo=ZoneInfo("America/New_York"))
OCC = "ABC261016C00100000"


class FakeBroker:
    def __init__(self, *, status="accepted", filled_qty=0, fill=None):
        self.status, self.filled_qty, self.fill = status, filled_qty, fill
        self.orders = []

    def submit_option_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"id": str(len(self.orders)), "status": self.status,
                "filled_qty": str(self.filled_qty), "filled_avg_price": self.fill}

    def get_positions(self):
        return [{"symbol": OCC, "qty": "10", "asset_class": "us_option"}]

    def get_option_quotes(self, **kwargs):
        return {}


def state():
    return {"ticker": "ABC", "route": "option", "occ": OCC, "contracts": 10,
            "entry_avg_price": 1.0, "entry_bar": "2026-09-10T14:00:00Z",
            "runs_held": 2, "last_mark_price": 1.0, "u_entry": 100., "u_atr": 2.}


def rows(root, module):
    path = root / module / "closed_trades.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stale_risk(root):
    bars = root / "bars"
    bars.mkdir()
    index = pd.date_range("2026-09-01", periods=20, freq="4h", tz="UTC")
    pd.DataFrame({"high": 101., "low": 99., "close": 100.}, index=index).to_parquet(bars / "ABC.parquet")
    kwargs = dict(client=FakeBroker(), module="review", managed={"ABC": state()},
                  pos_info={OCC: {"qty": 10, "avg_entry": 1., "current": .2}},
                  policy=execution.ExecPolicy(), now_et=NOW, ledger_root=str(root))
    with patch.object(execution, "DEFAULT_BARS_4H_DIR", bars):
        cached = risk.evaluate_risk_exits(**copy.deepcopy(kwargs))
    fresh = risk.evaluate_risk_exits(**kwargs, underlying_fn=lambda ticker: (90., 2.))
    assert cached.plan == [] and fresh.plan[0][3] == "underlying_stop_-1.5atr"
    return {"cache_last": str(index[-1]), "decision_time": str(NOW),
            "cached_close": 100, "fresh_counterfactual_close": 90,
            "cached_orders": len(cached.plan), "fresh_orders": len(fresh.plan)}


def partial_exit(root):
    broker = FakeBroker(status="partially_filled", filled_qty=2, fill="2.0")
    managed = {}
    with patch.object(execution, "defer_exits_if_opg_unavailable", lambda module, bar, plan, limits, **kw: plan):
        execution.execute_plan(broker, plan=[(OCC, "sell", 10, "horizon", "option")],
                               limits={}, submit=True, equity_tif_fn=lambda: "day",
                               new_managed=managed, exit_context={OCC: ("ABC", state())},
                               module="partial", pos_lookup={OCC: {"avg_entry": 1.}},
                               bar=NOW, ledger_root=str(root))
    record = rows(root, "partial")[0]
    assert not managed and record["qty"] == 10 and record["realized_pnl"] == 1000
    return {"actual_filled_qty": 2, "booked_qty": record["qty"],
            "actual_realized_pnl": 200, "booked_realized_pnl": record["realized_pnl"],
            "remaining_8_contracts_owned": bool(managed)}


def deferred_acceptance(root):
    module = "deferred"
    queue = execution.pending_exit_path(module, str(root))
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(json.dumps({"entries": [{"order_symbol": OCC, "qty": 10,
        "side": "sell", "route": "option", "reason": "horizon", "full_exit": True}]}))
    managed = {"ABC": state()}
    execution.submit_pending_exit_orders(FakeBroker(), module, equity_tif_fn=lambda: "day",
        pos_lookup={OCC: {"qty": 10, "avg_entry": 1.}}, managed=managed,
        now=NOW, ledger_root=str(root))
    record = rows(root, module)[0]
    assert json.loads(queue.read_text())["entries"] == []
    assert "exit_pending" not in managed["ABC"] and record["realized_pnl"] is None
    return {"queue_entries": 0, "pending_order_tracked": False, "phantom_close_pnl": None}


def intraday_acceptance(root):
    ex = IntradayOptionExecutor(FakeBroker(), ExecutionPolicy(state_path=str(root / "intraday.json")),
                               select_option_fn=lambda *a, **kw: None, ledger_root=str(root), now_fn=lambda: NOW)
    pos = {**state(), "qty": 10, "entry_filled_qty": 10., "entry_fill_price": 1.}
    ex._open["setup"] = pos
    record = ex._close_position("setup", pos, exit_reason="invalidated", u_exit=99.)
    assert ex.open_positions == {} and record["exit_fill_price"] is None
    return {"accepted_but_unfilled": True, "owner_removed": True, "close_written": record is not None}


def missing_order_loss(root):
    class Unavailable:
        def get_order(self, oid):
            raise TimeoutError("synthetic unavailable order read")
    st = {"exit_pending": {"order_id": "equity-order", "route": "equity",
                          "qty": 10, "entry_avg_price": 100.}}
    execution.resolve_settled_exit(Unavailable(), module="missing", ticker="ABC", symbol="ABC",
                                  state=st, bar=NOW, ledger_root=str(root))
    record = rows(root, "missing")[0]
    assert record["realized_pnl"] == -1000. and record["settle_outcome"] == "expired_worthless"
    return {"instrument": "equity", "order_read": "timeout", "invented_pnl": -1000,
            "invented_outcome": record["settle_outcome"]}


def failed_cancel(root):
    class CancelFailure(FakeBroker):
        def get_option_quotes(self, symbols=None, **kwargs):
            return {"quotes": {symbols: {"bp": 1., "ap": 2.}}}
        def cancel_order(self, oid):
            raise TimeoutError("synthetic cancellation failure")
    broker = CancelFailure()
    execution.submit_option_entry_with_ladder(broker, symbol=OCC, qty=10, ask=2.,
                                             poll_fn=lambda resp: False, sleep_fn=lambda seconds: None)
    assert len(broker.orders) == 3 and sum(order["qty"] for order in broker.orders) == 30
    return {"intended_qty": 10, "submitted_potentially_live_qty": 30, "orders": 3}


def dead_pending_order(root):
    class Canceled(FakeBroker):
        reads = 0
        def get_order(self, oid):
            self.reads += 1
            return {"id": oid, "status": "canceled", "filled_qty": "0"}
    broker = Canceled()
    st = state()
    st["exit_pending"] = {"order_id": "dead"}
    res = risk.evaluate_risk_exits(broker, module="pending", managed={"ABC": st},
        pos_info={OCC: {"qty": 10, "avg_entry": 1., "current": .2}},
        policy=execution.ExecPolicy(), now_et=NOW,
        underlying_fn=lambda ticker: (90., 2.), ledger_root=str(root))
    assert res.plan == [] and broker.reads == 0
    assert res.skipped["ABC"] == "exit_order_already_resting"
    return {"broker_status": "canceled", "order_reads": broker.reads, "breached_stop_retried": False}


def unshared_lock(root):
    # Compile the actual runner's small state writer, avoiding module bootstraps.
    path = REPO / "strategies/momentum_expansion/live/runner.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_save_state")
    state_path = root / "momentum.json"
    ns = {"STATE_PATH": state_path, "json": json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    risk.save_state(state_path, {"managed": {"ABC": state()}})
    stale = risk.load_state(state_path)
    with patch.object(risk, "LOCK_ROOT", root / "locks"):
        with risk.module_state_lock("momentum_expansion") as acquired:
            assert acquired
            ns["_save_state"]({"managed": {"ABC": state(), "NEW": {"contracts": 1}}})
            assert "NEW" in risk.load_state(state_path)["managed"]
            risk.save_state(state_path, stale)
    assert "NEW" not in risk.load_state(state_path)["managed"]
    return {"runner_writes_while_risk_lock_held": True, "new_position_lost_by_stale_save": True}


def embargo_overlap(root):
    path = REPO / "scripts/label_bakeoff/run_bakeoff.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "split_with_embargo")
    ns = {"pd": pd, "np": np, "EMBARGO_BARS": 60}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    days = pd.bdate_range("2025-01-01", periods=300, tz="UTC")
    bars = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in (14, 18)])
    frame = pd.DataFrame({"x": 1}, index=pd.MultiIndex.from_product([bars, ["ABC"]]))
    train, val, _ = ns["split_with_embargo"](frame)
    end = train.index.get_level_values(0).max()
    first_val = val.index.get_level_values(0).min()
    distance = bars.get_loc(first_val) - bars.get_loc(end)
    assert distance < 25 < 60
    return {"promised_embargo_bars": 60, "actual_boundary_bar_distance": distance,
            "25_bar_training_target_overlaps_validation": True}


def checkpoint_future_volume(root):
    # Execute the real feature builder using a synthetic daily cache. Changing
    # only day 6 must not change features available at the close of day 5.
    path = REPO / "scripts/horizon_thesis/build_decision_panel.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ticker_frame")
    ns = {"pd": pd, "np": np, "BARS_1D": root, "MAX_H": 30,
          "HOLDS": [5, 10, 15, 20, 30], "CHECKPOINT": 5, "BETA_WINDOW": 60}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    daily = pd.DataFrame({"timestamp": pd.bdate_range("2025-01-01", periods=210, tz="UTC")
                                       + pd.Timedelta(hours=14),
                          "open": 100., "high": 101., "low": 99., "close": 100., "volume": 1000.})
    target = root / "ABC.parquet"
    daily.to_parquet(target)
    before = ns["ticker_frame"]("ABC")
    entry = 100
    daily.loc[entry + 5, "volume"] = 1_000_000.  # day 6, strictly after checkpoint
    daily.to_parquet(target)
    after = ns["ticker_frame"]("ABC")
    old, new = before.loc[entry, "cp_volume_ratio"], after.loc[entry, "cp_volume_ratio"]
    assert before.loc[entry, "cp_ret"] == after.loc[entry, "cp_ret"]
    assert old == 1. and new > 100.
    return {"day_5_volume_feature_before": old, "after_changing_only_day_6_volume": new,
            "feature_consumed_by_continuation_model": True}


def test_set_model_selection(root):
    path = REPO / "signals/meta_context/meta_ranker/colab_competition.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {"choose_best", "primary_metric_name"}]
    ns = {"pd": pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    results = pd.DataFrame({"family": ["validation_winner", "test_winner"],
                            "val_ndcg_at_10": [.9, .1], "test_ndcg_at_10": [.1, .9]})
    selected = ns["choose_best"](results)["family"]
    assert selected == "test_winner" and ns["primary_metric_name"](results) == "test_ndcg_at_10"
    return {"selected_family": selected, "selection_metric": "test_ndcg_at_10",
            "origin": "pre-existing, also relevant to current retraining"}


def main():
    output = {}
    with tempfile.TemporaryDirectory(prefix="cyno-system-review-") as tmp:
        for check in (stale_risk, partial_exit, deferred_acceptance, intraday_acceptance,
                      missing_order_loss, failed_cancel, dead_pending_order, unshared_lock,
                      embargo_overlap, checkpoint_future_volume, test_set_model_selection):
            root = Path(tmp) / check.__name__
            root.mkdir()
            output[check.__name__] = check(root)
    print("\nREVIEW_DEMONSTRATIONS_JSON")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
