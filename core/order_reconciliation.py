"""Broker observations and durable exit evidence, independent of strategy/policy/DB."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from core.live_state import load_state, save_state

WORKING = frozenset({"new", "accepted", "pending_new", "accepted_for_bidding",
                     "partially_filled", "pending_cancel", "pending_replace", "held",
                     "stopped", "suspended", "calculated"})
TERMINAL = frozenset({"filled", "canceled", "cancelled", "expired", "rejected"})


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 else None
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class OrderObservation:
    order_id: str
    status: str = "unknown"
    filled_qty: float | None = None
    average_price: float | None = None

    @property
    def terminal(self):
        return self.status in TERMINAL

    @property
    def known(self):
        return self.status in WORKING | TERMINAL


def observe_order(client, order_id, response=None):
    """One bounded read. Missing/unknown evidence is never a terminal outcome.

    A replaced order needs its replacement's reservation reconciled; it is
    deliberately UNKNOWN here rather than permission to stack another sell.
    """
    order_id = str(order_id or "")
    payload = response if isinstance(response, dict) else {}
    if str(payload.get("status", "")).lower() not in TERMINAL | {"partially_filled"}:
        if order_id and order_id not in {"?", "None"} and hasattr(client, "get_order"):
            try:
                payload = client.get_order(order_id) or {}
            except Exception:
                # A partial fill in the submit response is still evidence.
                payload = response if isinstance(response, dict) else {}
    return OrderObservation(order_id, str(payload.get("status", "unknown")).lower(),
                            number(payload.get("filled_qty")), number(payload.get("filled_avg_price")))


def owned_quantity(state, route, broker_qty):
    """Account total is a ceiling, never proof of this strategy's ownership."""
    for key in ("remaining_qty", "contracts" if route == "option" else "shares", "qty", "entry_filled_qty"):
        qty = number(state.get(key))
        if qty is not None:
            return min(float(broker_qty), qty)
    return float(broker_qty)  # legacy states: current broker ceiling is all we know


def evidence_path(root, module, order_id):
    digest = hashlib.sha256(str(order_id).encode()).hexdigest()
    return Path(root) / str(module) / "exit_orders" / f"{digest}.json"


def save_evidence(root, module, order_id, evidence):
    save_state(evidence_path(root, module, order_id), evidence)


def read_evidence(root, module, order_id):
    path = evidence_path(root, module, order_id)
    return load_state(path) if path.exists() else None


def recover_pending_exits(root, module, managed):
    """Recover an accepted order persisted before the caller's state save.

    Never overwrite a different generation of a position. Completed evidence
    remains available by order ID to replay a stale pending record idempotently.
    """
    directory = Path(root) / str(module) / "exit_orders"
    for path in directory.glob("*.json"):
        record = load_state(path)
        key, saved = record["owner"], record["state"]
        current = managed.get(key)
        oid = saved["exit_pending"].get("order_id")
        if current is None and not record.get("complete"):
            managed[key] = saved
        elif (current is not None and current.get("entry_bar") == saved.get("entry_bar")
              and oid not in current.get("exit_reconciled_ids", []) and "exit_pending" not in current):
            current["exit_pending"] = saved["exit_pending"]


def settlement_activity(client, symbol, state, *, now):
    """Find executed option disposition evidence, never infer it from missingness.

    Reads are bounded; absence on a partial page or delayed paper activity stays
    unresolved and can be checked on a later pass.
    """
    if not hasattr(client, "get_account_activities"):
        return None
    import pandas as pd
    try:
        start = pd.Timestamp(state.get("entry_bar") or state["exit_pending"]["submitted_bar"])
        end = pd.Timestamp(now)
        token = None
        for _ in range(3):
            activities = client.get_account_activities(
                activity_types="OPEXP,OPEXC,OPASN", after=start.isoformat(),
                until=end.isoformat(), direction="desc", page_size=100, page_token=token)
            if not isinstance(activities, list):
                return None
            for row in activities:
                if (row.get("symbol") == symbol and row.get("status") == "executed"
                        and row.get("id") and row.get("activity_type") in {"OPEXP", "OPEXC", "OPASN"}):
                    return row
            if len(activities) < 100:
                break
            token = activities[-1].get("id")
            if not token:
                break
    except Exception:
        return None
    return None
