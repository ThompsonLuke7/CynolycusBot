"""Account-wide, broker-authoritative fill journal and reconciliation gate.

Strategy ledgers are useful projections, but they are not execution evidence.
This module keeps the broker's immutable ``FILL`` activities separately and
certifies that each has an owner and that every registered exit is represented
in its module ledger.  A failed certificate is deliberately a reporting stop,
not a best-effort warning.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.live_state import append_once, save_state

DEFAULT_ROOT = "Data/inference"


def _n(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _root(root: str | Path | None, account_label: str) -> Path:
    return Path(root or DEFAULT_ROOT) / "broker_reconciliation" / str(account_label)


def account_label_for_client(client: Any) -> str:
    """Derive the account partition from Alpaca's configured trading endpoint."""
    endpoint = str(getattr(client, "_trading_base", "")).lower()
    return "paper" if "paper-api" in endpoint else "live"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_order_ownership(*, order_id: str | None, module: str, symbol: str,
                             side: str, qty: float, route: str,
                             entry_avg_price: float | None = None,
                             reason: str | None = None, account_label: str = "paper",
                             root: str | Path | None = None) -> bool:
    """Durably register an accepted order before its module state can be lost.

    The order id is the global identity.  Conflicting duplicate registrations
    are retained as evidence and surfaced by the certificate rather than being
    silently reassigned.
    """
    oid = str(order_id or "").strip()
    if not oid or oid == "?":
        return False
    record = {
        "recorded_at": _now(), "order_id": oid, "module": str(module),
        "symbol": str(symbol), "side": str(side).lower(), "qty": float(qty),
        "route": str(route), "entry_avg_price": _n(entry_avg_price),
        "reason": reason,
    }
    return append_once(_root(root, account_label) / "order_registry.jsonl", record,
                       identity=f"order:{oid}")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def ingest_fills(client, *, account_label: str = "paper", root: str | Path | None = None,
                 after: str | None = None, until: str | None = None,
                 max_pages: int = 20) -> list[dict]:
    """Fetch and durably journal broker FILL activities, idempotently.

    Paging uses Alpaca's activity-id cursor.  Failure is raised: a partial
    activity read must never certify an account as reconciled.
    """
    if not hasattr(client, "get_account_activities"):
        raise RuntimeError("Broker client cannot retrieve account activities")
    end = until or _now()
    start = after or (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    journal = _root(root, account_label) / "fills.jsonl"
    seen: list[dict] = []
    token = None
    for _ in range(max_pages):
        page = client.get_account_activities(activity_types="FILL", after=start, until=end,
                                             direction="asc", page_size=100, page_token=token)
        if not isinstance(page, list):
            raise RuntimeError("Broker activities response is not a list")
        for raw in page:
            aid = str(raw.get("id") or "").strip()
            if not aid:
                raise RuntimeError("Broker fill missing activity id")
            record = {"ingested_at": _now(), "activity": raw}
            append_once(journal, record, identity=f"fill:{aid}")
            seen.append(raw)
        if len(page) < 100:
            return seen
        token = page[-1].get("id")
        if not token:
            raise RuntimeError("Broker activities page is missing its cursor")
    raise RuntimeError("Broker fill pagination exceeded max_pages")


def _module_exit_totals(root: Path, order_id: str) -> tuple[float, float]:
    qty = proceeds = 0.0
    for path in root.glob("*/exit_fills.jsonl"):
        for row in _read_jsonl(path):
            if str(row.get("order_id")) == order_id:
                q, p = _n(row.get("qty")), _n(row.get("proceeds"))
                if q is not None and p is not None:
                    qty += q
                    proceeds += p
    return qty, proceeds


def reconcile_account(client, *, account_label: str = "paper", root: str | Path | None = None,
                      after: str | None = None, until: str | None = None) -> dict:
    """Return a durable PASS/FAIL certificate; never repair a strategy ledger."""
    base = _root(root, account_label)
    fills = ingest_fills(client, account_label=account_label, root=root, after=after, until=until)
    registry_rows = _read_jsonl(base / "order_registry.jsonl")
    owners: dict[str, dict] = {}
    duplicate_owners: list[str] = []
    for row in registry_rows:
        oid = str(row.get("order_id") or "")
        if oid in owners and owners[oid] != row:
            duplicate_owners.append(oid)
        owners.setdefault(oid, row)

    unmatched, ledger_mismatches, checked = [], [], []
    by_order: dict[str, list[dict]] = {}
    for fill in fills:
        by_order.setdefault(str(fill.get("order_id") or ""), []).append(fill)
    for oid, rows in by_order.items():
        owner = owners.get(oid)
        if owner is None:
            unmatched.extend(rows)
            continue
        fill_qty = sum(_n(x.get("qty")) or 0.0 for x in rows)
        fill_proceeds = sum(
            (_n(x.get("qty")) or 0.0) * (_n(x.get("price")) or 0.0)
            for x in rows
        )
        side = str(owner.get("side", "")).lower()
        if side == "sell":
            ledger_qty, ledger_proceeds = _module_exit_totals(Path(root or DEFAULT_ROOT), oid)
            # Options exit_fills stores dollar proceeds (already x100); shares do not.
            multiplier = 100.0 if owner.get("route") == "option" else 1.0
            if not (math.isclose(fill_qty, ledger_qty, abs_tol=1e-8)
                    and math.isclose(fill_proceeds * multiplier, ledger_proceeds, abs_tol=0.01)):
                ledger_mismatches.append({"order_id": oid, "module": owner.get("module"),
                    "broker_qty": fill_qty, "ledger_qty": ledger_qty,
                    "broker_proceeds": round(fill_proceeds * multiplier, 2),
                    "ledger_proceeds": round(ledger_proceeds, 2)})
        checked.append(oid)

    quarantine = base / "quarantine.jsonl"
    for fill in unmatched:
        append_once(quarantine, {"detected_at": _now(), "reason": "unattributed_broker_fill",
                                 "activity": fill}, identity=f"quarantine:{fill['id']}")
    certificate = {
        "generated_at": _now(), "account_label": account_label, "after": after,
        "until": until, "fill_count": len(fills), "checked_order_ids": sorted(set(checked)),
        "unattributed_fills": unmatched, "ledger_mismatches": ledger_mismatches,
        "duplicate_order_owners": sorted(set(duplicate_owners)),
    }
    certificate["status"] = "PASS" if not (unmatched or ledger_mismatches or duplicate_owners) else "FAIL"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_state(base / "certificates" / f"reconciliation_{stamp}.json", certificate)
    save_state(base / "latest_certificate.json", certificate)
    return certificate


def require_clean_certificate(*, account_label: str = "paper", root: str | Path | None = None) -> dict:
    """Reporting guard: final realized-P&L claims require a current PASS."""
    path = _root(root, account_label) / "latest_certificate.json"
    if not path.exists():
        raise RuntimeError("Broker reconciliation certificate is missing; report is provisional")
    certificate = json.loads(path.read_text())
    if certificate.get("status") != "PASS":
        raise RuntimeError("Broker reconciliation failed; report is blocked pending fill review")
    return certificate
