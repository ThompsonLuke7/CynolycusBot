"""Read the closed-setup ledger and the abstention log; say what happened.

This is the half of "measurable" that is not collection.  Everything here is
descriptive: it aggregates recorded outcomes and never fits, tunes, or selects.

Two rules it enforces on the caller's behalf:

* **Cost assumptions are grouped, not averaged over.**  Every row carries the
  spread/slippage it was priced under.  Silently blending rows priced under
  different assumptions would produce a number that describes no policy at all,
  so a mixed ledger is reported as mixed.
* **Small samples are labelled.**  A bucket under ``MIN_REPORTABLE_N`` is still
  shown, because hiding it is worse, but it is flagged so nobody quotes it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


#: Below this, a bucket's mean is noise. Not a significance test — a guard
#: against quoting a 3-trade "edge".
MIN_REPORTABLE_N = 30


@dataclass(frozen=True)
class Bucket:
    """One slice of the ledger."""

    key: str
    n: int
    win_rate: float | None
    mean_net_return: float | None
    median_net_return: float | None
    total_net_return: float
    mean_r: float | None
    mean_mfe_over_mae: float | None
    median_bars_held: float | None

    @property
    def underpowered(self) -> bool:
        return self.n < MIN_REPORTABLE_N

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "n": self.n, "win_rate": self.win_rate,
            "mean_net_return": self.mean_net_return,
            "median_net_return": self.median_net_return,
            "total_net_return": self.total_net_return,
            "mean_r": self.mean_r, "mean_mfe_over_mae": self.mean_mfe_over_mae,
            "median_bars_held": self.median_bars_held,
            "underpowered": self.underpowered,
        }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read an append-only log, skipping any row a crash left half-written."""
    file = Path(path)
    if not file.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize(rows: Sequence[dict[str, Any]], *, key: str = "all") -> Bucket:
    returns = [_f(r.get("net_return")) for r in rows]
    returns = [x for x in returns if x is not None]
    if not returns:
        return Bucket(key, len(rows), None, None, None, 0.0, None, None, None)
    rs = [_f(r.get("realized_r_after_costs")) for r in rows]
    rs = [x for x in rs if x is not None]
    ratios = [
        _f(r.get("mfe_points")) / _f(r.get("mae_points"))
        for r in rows
        if _f(r.get("mae_points")) not in (None, 0.0) and _f(r.get("mfe_points")) is not None
    ]
    bars = [_f(r.get("bars_held")) for r in rows]
    bars = [x for x in bars if x is not None]
    return Bucket(
        key=key,
        n=len(rows),
        win_rate=sum(1 for x in returns if x > 0) / len(returns),
        mean_net_return=_mean(returns),
        median_net_return=_median(returns),
        total_net_return=sum(returns),
        mean_r=_mean(rs),
        mean_mfe_over_mae=_mean(ratios),
        median_bars_held=_median(bars),
    )


def group_by(rows: Sequence[dict[str, Any]], field: str) -> list[Bucket]:
    """Summarize by one field, largest bucket first."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(field)
        if isinstance(value, list):
            value = "+".join(str(x) for x in value) or "none"
        groups.setdefault(str(value), []).append(row)
    return sorted(
        (summarize(items, key=name) for name, items in groups.items()),
        key=lambda bucket: bucket.n, reverse=True,
    )


def cost_assumption_groups(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """How many rows sit under each cost assumption.

    More than one entry means the ledger spans a config change and its headline
    number describes no single policy.
    """
    counts: dict[str, int] = {}
    for row in rows:
        key = (
            f"spread={row.get('cost_spread_bps')}bps,"
            f"slippage={row.get('cost_slippage_bps')}bps,"
            f"commission={row.get('cost_commission_per_share')}"
        )
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def abstention_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What the engine declined, and on what grounds."""
    by_reason: dict[str, int] = {}
    by_regime: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("no_trade_reason") or "unknown")
        regime = str(row.get("context_regime") or "unknown")
        by_reason[reason] = by_reason.get(reason, 0) + 1
        by_regime[regime] = by_regime.get(regime, 0) + 1
    return {
        "total": len(rows),
        "by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
        "by_regime": dict(sorted(by_regime.items(), key=lambda kv: -kv[1])),
    }


def build_report(
    ledger_rows: Sequence[dict[str, Any]],
    abstention_rows: Sequence[dict[str, Any]] = (),
    *,
    group_fields: Iterable[str] = ("setup_type", "direction", "terminal_state", "context_regime", "target_level_type"),
) -> dict[str, Any]:
    taken = len(ledger_rows)
    declined = len(abstention_rows)
    return {
        "headline": summarize(ledger_rows).to_dict(),
        "cost_assumptions": cost_assumption_groups(ledger_rows),
        "selectivity": {
            "taken": taken,
            "declined": declined,
            "take_rate": taken / (taken + declined) if (taken + declined) else None,
        },
        "groups": {
            field: [bucket.to_dict() for bucket in group_by(ledger_rows, field)]
            for field in group_fields
        },
        "abstentions": abstention_summary(abstention_rows),
    }


def render_report(report: dict[str, Any]) -> str:
    """Plain-text rendering, so a report can be read from a terminal or a log."""
    out: list[str] = []
    head = report["headline"]
    out.append("=" * 78)
    out.append("INTRADAY STRUCTURE — CLOSED SETUP LEDGER")
    out.append("=" * 78)
    if not head["n"]:
        out.append("No closed setups recorded yet. The ledger fills forward from")
        out.append("the first setup that reaches a terminal state.")
        return "\n".join(out)

    out.append(_bucket_line("ALL", head))
    costs = report["cost_assumptions"]
    if len(costs) > 1:
        out.append("")
        out.append("WARNING: rows span more than one cost assumption, so the headline")
        out.append("number above describes no single policy. Breakdown:")
        for key, count in costs.items():
            out.append(f"    {count:6d}  {key}")
    elif costs:
        out.append(f"  costs: {next(iter(costs))}")

    sel = report["selectivity"]
    if sel["declined"]:
        out.append("")
        out.append(
            f"  selectivity: took {sel['taken']}, declined {sel['declined']}"
            f" ({sel['take_rate']:.1%} take rate)"
        )

    for field, buckets in report["groups"].items():
        rows = [b for b in buckets if b["n"]]
        if not rows:
            continue
        out.append("")
        out.append(f"--- by {field} " + "-" * max(0, 60 - len(field)))
        for bucket in rows:
            out.append(_bucket_line(bucket["key"], bucket))

    abstentions = report["abstentions"]
    if abstentions["total"]:
        out.append("")
        out.append("--- declined " + "-" * 60)
        for reason, count in abstentions["by_reason"].items():
            out.append(f"  {count:6d}  {reason}")
        out.append("  regime at the moment of declining:")
        for regime, count in abstentions["by_regime"].items():
            out.append(f"  {count:6d}  {regime}")

    if any(b["underpowered"] for buckets in report["groups"].values() for b in buckets if b["n"]):
        out.append("")
        out.append(f"(*) fewer than {MIN_REPORTABLE_N} setups — not a result, do not quote.")
    return "\n".join(out)


def _bucket_line(label: str, bucket: dict[str, Any]) -> str:
    flag = " *" if bucket.get("underpowered") else "  "
    mean = bucket["mean_net_return"]
    median = bucket["median_net_return"]
    win = bucket["win_rate"]
    r = bucket["mean_r"]
    bars = bucket["median_bars_held"]
    return (
        f"  {label[:38]:<38}{flag} n={bucket['n']:<6d}"
        f" mean={_pct(mean):>9}  median={_pct(median):>9}"
        f"  win={_pct(win, 1):>7}  R={_num(r):>7}  bars={_num(bars, 0):>6}"
    )


def _pct(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None else f"{100 * value:+.{places}f}%"


def _num(value: float | None, places: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
