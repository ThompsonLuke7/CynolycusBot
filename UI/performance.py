"""Read-only, module-scoped performance summaries for dashboard surfaces.

The summary intentionally uses only the durable realized-P/L ledgers written by
``core.live_4h_exec.record_exit_realized_pnl``.  It never substitutes a
backtest, an account-wide broker total, or a current mark for a closed result.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
LEDGER_ROOT = REPO / "Data" / "inference"


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def module_performance(module: str, *, open_upl: float | None = None) -> dict[str, Any]:
    """Return realized ledger metrics plus the module's current marked P/L.

    ``tracked_pnl`` is deliberately labelled as a partial track record until a
    module has complete ledger coverage.  This prevents historical account P/L
    or research results being silently presented as live module performance.
    """
    path = LEDGER_ROOT / str(module) / "closed_trades.jsonl"
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pnl = _number(row.get("realized_pnl"))
            if pnl is not None:
                rows.append({**row, "realized_pnl": pnl})

    realized = round(sum(r["realized_pnl"] for r in rows), 2)
    wins = sum(r["realized_pnl"] > 0 for r in rows)
    today = datetime.now(timezone.utc).date().isoformat()
    today_pnl = round(sum(r["realized_pnl"] for r in rows if str(r.get("ts", "")).startswith(today)), 2)
    marked = _number(open_upl) or 0.0
    return {
        "module": module,
        "ledger_available": path.exists(),
        "ledger_path": str(path),
        "closed_trades": len(rows),
        "win_rate": round(wins / len(rows) * 100, 1) if rows else None,
        "realized_pnl": realized,
        "today_realized_pnl": today_pnl,
        "open_unrealized_pnl": round(marked, 2),
        "tracked_pnl": round(realized + marked, 2),
        "status": "tracked" if rows else "no_closed_ledger",
    }
