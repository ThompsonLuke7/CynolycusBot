"""The broker-equity snapshot publishes on weekends; trading loops do not.

PORTFOLIO state expires after one day, and `policy.rule.broker` is one of only
two hard rules that still apply to risk-reducing orders. Every scheduler
defaults to `weekdays_only=True`, so the newest publication before a Monday
09:35 pre-open flush was Friday 20:05 ET -- about 61 hours old. The snapshot
requirement then rejects it, `portfolio` resolves to None, and every Meta order
is vetoed BROKER_PORTFOLIO_STATE_MISSING, exits included: exactly the queue that
flush exists to drain (2026-09-11: 14 exits stayed queued).

The capture itself is a read-only broker call, so running it on a Saturday and
Sunday costs nothing and keeps Monday's flush inside the freshness window.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import UI.combined_server as combined_server

_SNAPSHOT_LABEL = "Broker snapshot"


def _label_text(node: ast.AST) -> str:
    """The label argument, which is either a literal or an f-string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return ""


def _weekdays_only(keywords: dict[str, ast.AST]) -> bool:
    node = keywords.get("weekdays_only")
    if node is None:
        return True  # the scheduler's own default
    assert isinstance(node, ast.Constant), "weekdays_only must be a literal to be auditable"
    return bool(node.value)


def _schedule_loop_calls() -> list[tuple[str, bool]]:
    source = Path(inspect.getfile(combined_server)).read_text()
    calls = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "_schedule_loop":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        calls.append((_label_text(keywords.get("label", ast.Constant(""))), _weekdays_only(keywords)))
    return calls


def test_the_broker_snapshot_loop_is_scheduled_every_day():
    calls = _schedule_loop_calls()
    assert calls, "expected _schedule_loop call sites in the combined server"

    snapshots = [(label, weekdays) for label, weekdays in calls if _SNAPSHOT_LABEL in label]
    assert snapshots, f"expected a {_SNAPSHOT_LABEL!r} loop"
    assert all(weekdays is False for _label, weekdays in snapshots), (
        "the broker snapshot must run on weekends, or Monday's pre-open flush "
        "finds PORTFOLIO state ~61h old and vetoes every Meta order"
    )


def test_trading_loops_stay_on_weekdays():
    """Only the read-only capture gets the weekend; nothing that places orders."""
    trading = [(label, weekdays) for label, weekdays in _schedule_loop_calls()
               if _SNAPSHOT_LABEL not in label]

    assert len(trading) >= 3, "expected the trading loops to be scheduled here"
    offenders = [label for label, weekdays in trading if weekdays is not True]
    assert not offenders, f"these must not run on weekends: {offenders}"
