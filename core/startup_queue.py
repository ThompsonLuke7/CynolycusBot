"""Actions queued to run the next time the live server starts.

The server is started by hand and is frequently down when a decision is made —
overnight, on a weekend, or after a crash. Anything that has to happen "next
time we're up" (close a position, force a data-readiness rebuild) otherwise
lives in someone's head until it is forgotten. This is that list, on disk.

Design notes
------------
* One JSON file, ``Data/runtime/startup_queue.json``. Entries are appended by
  the CLI and drained by the server at boot.
* Entries are **not** deleted when they run. They are marked ``done`` /
  ``failed`` and kept, so the queue doubles as a record of what was executed.
* Order actions default to the PAPER account and refuse to touch a live account
  unless the entry explicitly says ``account: "live"``. A queued order is still
  an order: it records who queued it and why.
* Nothing here places an order while the market is closed — entries that need an
  open market stay ``pending`` and are retried at the next startup.

CLI
---
    python -m core.startup_queue list
    python -m core.startup_queue add-close SNDK --qty all --note "close it out"
    python -m core.startup_queue add-readiness --note "stamp went stale"
    python -m core.startup_queue cancel <id>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_PATH = REPO / "Data/runtime/startup_queue.json"

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

KIND_CLOSE_POSITION = "close_position"
KIND_DATA_READINESS = "data_readiness"
KIND_NOTE = "note"
KINDS = (KIND_CLOSE_POSITION, KIND_DATA_READINESS, KIND_NOTE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path | str = DEFAULT_QUEUE_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.error("startup queue: %s is unreadable (%s) — treating as empty", path, exc)
        return []
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    return list(entries) if isinstance(entries, list) else []


def save(entries: list[dict[str, Any]], path: Path | str = DEFAULT_QUEUE_PATH) -> None:
    """Atomic write — a half-written queue must never strand a close order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated": _now(), "entries": entries}
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".startup_queue.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def enqueue(
    kind: str,
    *,
    params: dict[str, Any] | None = None,
    note: str = "",
    queued_by: str = "cli",
    account: str = "paper",
    path: Path | str = DEFAULT_QUEUE_PATH,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"unknown startup-queue kind {kind!r}; expected one of {KINDS}")
    account = str(account).strip().lower()
    if account not in {"paper", "live"}:
        raise ValueError(f"account must be 'paper' or 'live', got {account!r}")
    entry = {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "params": dict(params or {}),
        "note": note,
        "account": account,
        "queued_by": queued_by,
        "queued_at": _now(),
        "status": STATUS_PENDING,
        "attempts": 0,
        "result": None,
    }
    entries = load(path)
    entries.append(entry)
    save(entries, path)
    return entry


def pending(path: Path | str = DEFAULT_QUEUE_PATH) -> list[dict[str, Any]]:
    return [e for e in load(path) if e.get("status") == STATUS_PENDING]


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

META_STATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "signals/meta_context/meta_ranker/live_state.json"
)


def meta_owned_symbols(*, state_path: Path | None = None) -> set[str]:
    """Symbols the Meta Ranker currently believes it manages.

    Read defensively: a module that has never run owns nothing, and an
    unreadable state file must not break the operator's queue.
    """

    path = state_path if state_path is not None else META_STATE_PATH
    try:
        managed = json.loads(Path(path).read_text()).get("managed", {})
    except Exception:  # noqa: BLE001 - absent or corrupt state owns nothing
        return set()
    owned: set[str] = set()
    for ticker, record in (managed or {}).items():
        if not isinstance(record, dict):
            continue
        symbol = record.get("occ") if record.get("route") == "option" else record.get("symbol", ticker)
        if symbol:
            owned.add(str(symbol).strip().upper())
    return owned


def _close_position(
    client, entry: dict[str, Any], *, owned_symbols: set[str] | None = None
) -> dict[str, Any]:
    """Flatten one symbol. ``qty: "all"`` closes the whole position.

    This is a human escape hatch and it is deliberately never blocked, even
    when a strategy owns the symbol: it is risk-reducing, explicitly authored,
    and an operator reaching for it in an emergency must not be told no.

    What it must not do is act invisibly. A close on strategy-owned inventory
    is flagged in its record, so the strategy's own reconciliation and the
    audit trail can see where the position went instead of the strategy
    continuing to manage something that no longer exists.
    """
    params = entry.get("params") or {}
    symbol = str(params.get("symbol") or "").strip().upper()
    if not symbol:
        raise ValueError("close_position requires params.symbol")
    raw_qty = params.get("qty", "all")

    positions = client.get_positions() or []
    match = next((p for p in positions if str(p.get("symbol", "")).upper() == symbol), None)
    if match is None:
        # Already flat is success, not failure — the intent is satisfied.
        return {"closed": False, "reason": "not_held", "symbol": symbol}

    held = abs(float(match.get("qty") or 0))
    if held <= 0:
        return {"closed": False, "reason": "zero_qty", "symbol": symbol}
    qty = held if str(raw_qty).strip().lower() == "all" else min(float(raw_qty), held)
    if qty <= 0:
        return {"closed": False, "reason": "requested_qty_zero", "symbol": symbol}

    side = "sell" if str(match.get("side") or "long").lower() == "long" else "buy"
    order_qty = int(qty) if float(qty).is_integer() else qty
    resp = client.submit_order(
        symbol=symbol, qty=order_qty, side=side,
        order_type="market", time_in_force="day",
    )
    return {
        "closed": True, "symbol": symbol, "side": side, "qty": order_qty,
        "held_before": held, "order_id": (resp or {}).get("id"),
        "order_status": (resp or {}).get("status"),
        "strategy_owned": symbol in (owned_symbols or set()),
    }


def run_pending(
    *,
    client_factory: Callable[[str], Any] | None = None,
    readiness_runner: Callable[[], Any] | None = None,
    market_is_open: Callable[[], bool] | None = None,
    allow_live: bool = False,
    path: Path | str = DEFAULT_QUEUE_PATH,
) -> dict[str, Any]:
    """Execute every pending entry. Never raises: one bad entry cannot stop boot.

    Entries needing an open market are left ``pending`` when it is closed, so the
    next startup picks them up.
    """
    entries = load(path)
    summary = {"ran": 0, "done": 0, "failed": 0, "deferred": 0, "skipped": 0}
    if not entries:
        return summary

    if market_is_open is None:
        from core.calendar import is_market_open_now

        market_is_open = is_market_open_now
    clients: dict[str, Any] = {}

    def _client(account: str):
        if account not in clients:
            if client_factory is None:
                from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

                clients[account] = AlpacaOptionsClient(
                    env_file=".env" if account == "paper" else ".env.live")
            else:
                clients[account] = client_factory(account)
        return clients[account]

    changed = False
    for entry in entries:
        if entry.get("status") != STATUS_PENDING:
            continue
        kind = entry.get("kind")
        account = str(entry.get("account") or "paper").lower()

        if account == "live" and not allow_live:
            logger.warning("startup queue: %s targets the LIVE account — skipped (allow_live is off)",
                           entry.get("id"))
            summary["skipped"] += 1
            continue

        if kind == KIND_NOTE:
            logger.info("startup queue note [%s]: %s", entry.get("id"), entry.get("note"))
            entry.update(status=STATUS_DONE, ran_at=_now(), result={"noted": True})
            summary["ran"] += 1
            summary["done"] += 1
            changed = True
            continue

        if kind == KIND_CLOSE_POSITION and not market_is_open():
            logger.info("startup queue: %s (%s) needs an open market — staying pending",
                        entry.get("id"), (entry.get("params") or {}).get("symbol"))
            summary["deferred"] += 1
            continue

        entry["attempts"] = int(entry.get("attempts") or 0) + 1
        summary["ran"] += 1
        changed = True
        try:
            if kind == KIND_CLOSE_POSITION:
                owned = meta_owned_symbols()
                result = _close_position(
                    _client(account), entry, owned_symbols=owned
                )
                if result.get("strategy_owned"):
                    logger.warning(
                        "startup queue: closed %s, which the Meta Ranker still "
                        "manages — its state will need reconciling",
                        result.get("symbol"),
                    )
            elif kind == KIND_DATA_READINESS:
                if readiness_runner is None:
                    raise RuntimeError("no readiness runner supplied")
                result = {"readiness": readiness_runner()}
            else:
                raise ValueError(f"unknown kind {kind!r}")
        except Exception as exc:  # noqa: BLE001
            logger.error("startup queue: %s (%s) FAILED: %s", entry.get("id"), kind, exc)
            entry.update(status=STATUS_FAILED, ran_at=_now(), result={"error": str(exc)})
            summary["failed"] += 1
        else:
            logger.info("startup queue: %s (%s) done: %s", entry.get("id"), kind, result)
            entry.update(status=STATUS_DONE, ran_at=_now(), result=result)
            summary["done"] += 1

    if changed:
        save(entries, path)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_list(args) -> int:
    entries = load(args.path)
    if not entries:
        print("startup queue is empty")
        return 0
    for e in entries:
        params = json.dumps(e.get("params") or {}, sort_keys=True)
        print(f"{e.get('id')}  {e.get('status'):<9} {e.get('kind'):<15} "
              f"{e.get('account'):<5} {params}  {e.get('note') or ''}")
    return 0


def _cmd_add_close(args) -> int:
    entry = enqueue(
        KIND_CLOSE_POSITION,
        params={"symbol": args.symbol, "qty": args.qty},
        note=args.note, account=args.account, path=args.path,
    )
    print(f"queued {entry['id']}: close {args.symbol} qty={args.qty} on {args.account}")
    return 0


def _cmd_add_readiness(args) -> int:
    entry = enqueue(KIND_DATA_READINESS, note=args.note, path=args.path)
    print(f"queued {entry['id']}: data readiness")
    return 0


def _cmd_add_note(args) -> int:
    entry = enqueue(KIND_NOTE, note=args.note, path=args.path)
    print(f"queued {entry['id']}: note")
    return 0


def _cmd_cancel(args) -> int:
    entries = load(args.path)
    for e in entries:
        if e.get("id") == args.id and e.get("status") == STATUS_PENDING:
            e.update(status=STATUS_CANCELLED, ran_at=_now())
            save(entries, args.path)
            print(f"cancelled {args.id}")
            return 0
    print(f"no pending entry with id {args.id}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Actions to run at the next live-server startup.")
    parser.add_argument("--path", type=Path, default=DEFAULT_QUEUE_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show every queued entry").set_defaults(func=_cmd_list)

    close = sub.add_parser("add-close", help="Close a position at next startup")
    close.add_argument("symbol")
    close.add_argument("--qty", default="all", help="'all' (default) or a share/contract count")
    close.add_argument("--account", default="paper", choices=("paper", "live"))
    close.add_argument("--note", default="")
    close.set_defaults(func=_cmd_add_close)

    readiness = sub.add_parser("add-readiness", help="Force a data-readiness rebuild at next startup")
    readiness.add_argument("--note", default="")
    readiness.set_defaults(func=_cmd_add_readiness)

    note = sub.add_parser("add-note", help="Log a reminder at next startup")
    note.add_argument("note")
    note.set_defaults(func=_cmd_add_note)

    cancel = sub.add_parser("cancel", help="Cancel a pending entry")
    cancel.add_argument("id")
    cancel.set_defaults(func=_cmd_cancel)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
