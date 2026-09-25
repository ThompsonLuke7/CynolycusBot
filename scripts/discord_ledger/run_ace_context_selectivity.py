"""Audit pre-candidate Discord context, news context, and universe availability.

This is a research-only association study.  ACE's source candidates and the
previously frozen same-symbol/session controls are inputs; neither outcomes nor
post-candidate data are used as features.  It deliberately keeps three data
qualities separate:

* Discord watchlists have conservative exported-message availability times.
* News uses article timestamps but the local library has no historical
  observation/ingestion timestamp, so it is retrospective timestamp-bounded
  context rather than certified point-in-time evidence.
* Universe membership is returned only where an immutable snapshot existed at
  the candidate time.  Missing historical snapshots remain missing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.discord_ledger.build import available_at


ROOT = Path(__file__).resolve().parents[2]
ACE_ID = "1079391263083733072"
WATCHLIST_WINDOWS = (timedelta(hours=24), timedelta(days=7))
NEWS_WINDOW = timedelta(hours=24)
NEWS_SAFE_COLUMNS = (
    "ticker", "timestamp", "record_id", "record_catalyst_score", "p_bullish",
    "is_direct_catalyst", "source", "catalyst_family", "catalyst_subtype",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"timezone-aware timestamp required: {value!r}")
    return timestamp.tz_convert("UTC")


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.isoformat().replace("+00:00", "Z")


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_watchlists(context_path: Path, messages_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load same-caller, same-symbol watchlists with conservative availability."""

    messages = {str(row["message_id"]): row for row in read_jsonl(messages_path)}
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(context_path):
        if row.get("author_id") != ACE_ID or row.get("event_type") != "watchlist":
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        message = messages.get(str(row.get("message_id")))
        if not symbol or message is None:
            continue
        available, basis = available_at(message)
        if available is None:
            continue
        by_symbol[symbol].append({
            "message_id": str(row["message_id"]),
            "available_at_utc": _utc(available),
            "availability_basis": basis,
            "speech_act": row.get("speech_act"),
            "has_level": bool(row.get("levels")),
            "source_quote": row.get("source_quote"),
        })
    for values in by_symbol.values():
        values.sort(key=lambda row: (row["available_at_utc"], row["message_id"]))
    return dict(by_symbol)


def watchlist_features(events: list[dict[str, Any]], decision: pd.Timestamp) -> dict[str, Any]:
    """Strictly-pre-decision watchlist features and exact supporting IDs."""

    prior = [event for event in events if event["available_at_utc"] < decision]
    result: dict[str, Any] = {
        "watchlist_24h": False, "watchlist_7d": False,
        "watchlist_level_7d": False, "watchlist_intention_or_instruction_7d": False,
        "watchlist_message_ids_24h": [], "watchlist_message_ids_7d": [],
        "watchlist_latest_available_at_utc": None,
        "watchlist_availability_basis": [],
    }
    for window in WATCHLIST_WINDOWS:
        selected = [event for event in prior if event["available_at_utc"] >= decision - window]
        label = "24h" if window == WATCHLIST_WINDOWS[0] else "7d"
        result[f"watchlist_{label}"] = bool(selected)
        result[f"watchlist_message_ids_{label}"] = [event["message_id"] for event in selected]
        if label == "7d":
            result["watchlist_level_7d"] = any(event["has_level"] for event in selected)
            result["watchlist_intention_or_instruction_7d"] = any(
                event["speech_act"] in {"intention", "instruction", "conditional"} for event in selected
            )
            result["watchlist_availability_basis"] = sorted({event["availability_basis"] for event in selected})
    if prior:
        result["watchlist_latest_available_at_utc"] = _iso(prior[-1]["available_at_utc"])
    return result


def load_news(news_path: Path, symbols: set[str]) -> dict[str, pd.DataFrame]:
    """Load only article-level, non-hindsight fields from the local news index."""

    frame = pd.read_parquet(news_path, columns=list(NEWS_SAFE_COLUMNS))
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame = frame[frame["ticker"].isin(symbols)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[frame["timestamp"].notna()].copy()
    return {symbol: group.sort_values("timestamp").reset_index(drop=True)
            for symbol, group in frame.groupby("ticker", sort=False)}


def news_features(news_by_symbol: dict[str, pd.DataFrame], symbol: str, decision: pd.Timestamp) -> dict[str, Any]:
    """Article-time-bounded context.  No same-day aggregate or realized labels."""

    frame = news_by_symbol.get(symbol)
    if frame is None:
        selected = pd.DataFrame(columns=NEWS_SAFE_COLUMNS)
    else:
        selected = frame[(frame["timestamp"] < decision) & (frame["timestamp"] >= decision - NEWS_WINDOW)]
    scores = pd.to_numeric(selected.get("record_catalyst_score"), errors="coerce")
    bullish = pd.to_numeric(selected.get("p_bullish"), errors="coerce")
    direct = pd.to_numeric(selected.get("is_direct_catalyst"), errors="coerce")
    return {
        "news_24h_article_count": int(len(selected)),
        "news_24h_has_article": bool(len(selected)),
        "news_24h_direct_catalyst_count": int((direct.fillna(0) > 0).sum()),
        "news_24h_max_record_catalyst_score": None if scores.dropna().empty else float(scores.max()),
        "news_24h_mean_p_bullish": None if bullish.dropna().empty else float(bullish.mean()),
        "news_latest_article_at_utc": None if selected.empty else _iso(selected["timestamp"].max()),
        "news_record_ids_24h": selected["record_id"].astype(str).tolist(),
        "news_context_status": "timestamp_bounded_not_certified_pit",
    }


@lru_cache(maxsize=8)
def _universe_snapshots(snapshot_dir: str) -> tuple[tuple[pd.Timestamp, str], ...]:
    """Index immutable snapshot names once; never substitute a current universe."""

    found = []
    for path in Path(snapshot_dir).glob("shared_universe_*.csv.gz"):
        stamp = path.name.removeprefix("shared_universe_").removesuffix(".csv.gz")
        try:
            taken = pd.Timestamp(datetime.strptime(stamp, "%Y%m%dT%H%M%SZ"), tz=timezone.utc)
        except ValueError:
            continue
        found.append((taken, str(path)))
    return tuple(sorted(found))


@lru_cache(maxsize=64)
def _read_universe_snapshot(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def universe_features(symbol: str, decision: pd.Timestamp, snapshot_dir: Path) -> dict[str, Any]:
    candidates = [item for item in _universe_snapshots(str(snapshot_dir)) if item[0] <= decision]
    if not candidates:
        return {"universe_status": "missing_pit_snapshot",
                "universe_error": f"no universe snapshot at or before {decision.isoformat()}",
                "universe_snapshot_utc": None, "in_eligible_universe": None}
    taken, path = candidates[-1]
    universe = _read_universe_snapshot(path)
    snapshot_values = universe.get("snapshot_utc")
    snapshot = str(snapshot_values.iloc[0]) if snapshot_values is not None and not universe.empty else _iso(taken)
    match = universe[universe["ticker"].astype(str).str.upper() == symbol]
    return {"universe_status": "resolved", "universe_error": None,
            "universe_snapshot_utc": snapshot,
            "in_eligible_universe": bool(match["is_eligible"].iloc[0]) if not match.empty else False}


def _group_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(map(str, row.get("source_message_ids") or [])))


def build_rows(
    alerts: list[dict[str, Any]], controls: list[dict[str, Any]], *,
    watchlists: dict[str, list[dict[str, Any]]], news_by_symbol: dict[str, pd.DataFrame],
    snapshot_dir: Path,
) -> list[dict[str, Any]]:
    control_counts = Counter(_group_key(row) for row in controls)
    rows: list[dict[str, Any]] = []
    for kind, source_rows in (("alert", alerts), ("control", controls)):
        for source in source_rows:
            symbol = str(source["symbol"]).upper()
            decision = _utc(source["decision_at_utc"])
            group = _group_key(source)
            weight = 1.0 if kind == "alert" else 1.0 / control_counts[group]
            row = {
                "row_kind": kind,
                "sample_weight": weight,
                "symbol": symbol,
                "decision_at_utc": _iso(decision),
                "direction_sign": int(source["direction_sign"]),
                "source_message_ids": list(group),
                "exclusive_bucket": source.get("exclusive_bucket"),
                "source_tags": source.get("source_tags", []),
            }
            row.update(watchlist_features(watchlists.get(symbol, []), decision))
            row.update(news_features(news_by_symbol, symbol, decision))
            row.update(universe_features(symbol, decision, snapshot_dir))
            rows.append(row)
    return rows


def _weighted_rate(rows: list[dict[str, Any]], column: str) -> float | None:
    usable = [row for row in rows if row.get(column) is not None]
    if not usable:
        return None
    weights = np.array([row["sample_weight"] for row in usable], dtype=float)
    values = np.array([bool(row[column]) for row in usable], dtype=float)
    return float(np.average(values, weights=weights))


def _weighted_mean(rows: list[dict[str, Any]], column: str) -> float | None:
    usable = [row for row in rows if row.get(column) is not None]
    if not usable:
        return None
    weights = np.array([row["sample_weight"] for row in usable], dtype=float)
    values = np.array([float(row[column]) for row in usable], dtype=float)
    return float(np.average(values, weights=weights))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns = (
        "watchlist_24h", "watchlist_7d", "watchlist_level_7d",
        "watchlist_intention_or_instruction_7d", "news_24h_has_article",
        "in_eligible_universe",
    )
    means = ("news_24h_article_count", "news_24h_direct_catalyst_count",
             "news_24h_max_record_catalyst_score", "news_24h_mean_p_bullish")
    result: dict[str, Any] = {}
    for kind in ("alert", "control"):
        group = [row for row in rows if row["row_kind"] == kind]
        result[kind] = {
            "rows": len(group), "effective_weight": sum(row["sample_weight"] for row in group),
            "rates": {column: _weighted_rate(group, column) for column in columns},
            "means": {column: _weighted_mean(group, column) for column in means},
            "universe_resolved_rows": sum(row["universe_status"] == "resolved" for row in group),
        }
    result["alert_minus_control"] = {
        "rates": {column: None if result["alert"]["rates"][column] is None or result["control"]["rates"][column] is None
                  else result["alert"]["rates"][column] - result["control"]["rates"][column] for column in columns},
        "means": {column: None if result["alert"]["means"][column] is None or result["control"]["means"][column] is None
                  else result["alert"]["means"][column] - result["control"]["means"][column] for column in means},
    }
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def write_report(path: Path, summary: dict[str, Any], *, inputs: dict[str, str]) -> None:
    alert, control, difference = summary["alert"], summary["control"], summary["alert_minus_control"]
    lines = [
        "# ACE point-in-time context selectivity", "",
        "Research-only association study: ACE candidates are compared with frozen, same-symbol/session "
        "non-alert controls. Controls are weighted so every source alert contributes one total control weight.",
        "No forward returns, source outcome claims, same-day news aggregates, option prices, or execution assumptions are inputs.", "",
        "## Coverage", "",
        f"- Alerts: {alert['rows']} (effective weight {alert['effective_weight']:.1f}).",
        f"- Controls: {control['rows']} (effective weight {control['effective_weight']:.1f}).",
        f"- Universe snapshots resolved: alerts {alert['universe_resolved_rows']}, controls {control['universe_resolved_rows']}. "
        "Missing snapshots are left unresolved rather than substituted with today's universe.",
        "- Discord features use conservative creation/edit availability. News uses article timestamps only and is not certified "
        "point-in-time because the local index has no historical observation/ingestion time.", "",
        "## Selectivity", "",
        "| Feature | ACE alerts | Controls | Difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for column in alert["rates"]:
        lines.append(f"| {column} | {_pct(alert['rates'][column])} | {_pct(control['rates'][column])} | {_pct(difference['rates'][column])} |")
    for column in alert["means"]:
        left, right, delta = alert["means"][column], control["means"][column], difference["means"][column]
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        lines.append(f"| {column} (mean) | {fmt(left)} | {fmt(right)} | {fmt(delta)} |")
    lines.extend(["", "## Inputs", ""] + [f"- {name}: `{value}`." for name, value in inputs.items()] + ["", "## Interpretation", "",
                  "This evaluates association with the observed ACE alert timing, not whether an independent system discovers "
                  "the candidate universe, predicts profitability, or can execute options. Small and incomplete universe coverage "
                  "is a data limitation, not a negative value."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alerts", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--news", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "Data/shared/universe/snapshots")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite research output: {args.out}")
    alerts, controls = read_jsonl(args.alerts), read_jsonl(args.controls)
    symbols = {str(row["symbol"]).upper() for row in alerts + controls}
    watchlists = load_watchlists(args.context, args.messages)
    news_by_symbol = load_news(args.news, symbols)
    rows = build_rows(alerts, controls, watchlists=watchlists, news_by_symbol=news_by_symbol,
                      snapshot_dir=args.snapshot_dir)
    summary = summarize(rows)
    args.out.mkdir(parents=True)
    write_jsonl(args.out / "candidate_context.jsonl", rows)
    (args.out / "results.json").write_text(json.dumps({
        "schema": "ace_context_selectivity_v1", "research_only": True,
        "inputs": {str(path): _digest(path) for path in (args.alerts, args.controls, args.context, args.messages, args.news)},
        "summary": summary,
        "limitations": [
            "News is bounded by article timestamp but has no historical observation/ingestion timestamp in this local index.",
            "The source candidate symbol and direction are injected; this is not a full-universe discovery test.",
            "No post-candidate returns or source-reported outcomes are used as features.",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    write_report(args.out / "report.md", summary, inputs={
        "alerts": str(args.alerts), "controls": str(args.controls), "context": str(args.context),
        "messages": str(args.messages), "news": str(args.news), "universe_snapshots": str(args.snapshot_dir),
    })
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
