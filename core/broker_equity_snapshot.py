"""Durable broker equity/position snapshots for daily P&L attribution.

The broker's account-level day P&L is authoritative, while individual position
``unrealized_intraday_pl`` fields can use a different mark basis.  Capturing
both at the regular and extended-session close makes future long/call/put/share
attribution reproducible from the exact broker marks observed at those times.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "Data/inference/account_snapshots"
_ET = ZoneInfo("America/New_York")
_ACCOUNT_FIELDS = ("equity", "last_equity", "portfolio_value", "cash", "buying_power")
_POSITION_FIELDS = (
    "symbol", "asset_class", "qty", "side", "avg_entry_price", "current_price",
    "market_value", "cost_basis", "unrealized_pl", "unrealized_plpc",
    "unrealized_intraday_pl", "unrealized_intraday_plpc",
)


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def snapshot_path(*, account_label: str, root: Path, now: datetime) -> Path:
    session = now.astimezone(_ET).strftime("%Y%m%d")
    safe_label = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in account_label)
    return root / f"broker_equity_{session}_{safe_label}.jsonl"


def capture_snapshot(
    *,
    client: Any,
    account_label: str = "paper",
    root: Path = DEFAULT_ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Read broker account/positions and append one JSONL close-mark snapshot."""
    captured = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    account = client.get_account() or {}
    positions = client.get_positions() or []
    record = {
        "schema_version": 1,
        "captured_at_utc": captured.isoformat(),
        "session_date_et": captured.astimezone(_ET).date().isoformat(),
        "account_label": account_label,
        "account": {field: _json_scalar(account.get(field)) for field in _ACCOUNT_FIELDS},
        "positions": [
            {field: _json_scalar(position.get(field)) for field in _POSITION_FIELDS}
            for position in positions
        ],
    }
    out = snapshot_path(account_label=account_label, root=root, now=captured)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return {"path": str(out), "position_count": len(record["positions"]), **record}


def capture_from_env(*, env_file: str, account_label: str = "paper", root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    return capture_snapshot(
        client=AlpacaOptionsClient(env_file=env_file),
        account_label=account_label,
        root=root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a read-only broker equity/position snapshot.")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--account-label", default="paper")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = capture_from_env(env_file=args.env, account_label=args.account_label, root=args.root)
    print(json.dumps({"path": result["path"], "position_count": result["position_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
