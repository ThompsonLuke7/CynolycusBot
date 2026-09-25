"""Append-only, IEX-underlying candidate tape for prospective ACE research.

This collector deliberately captures a bounded universe, not a fabricated all-US
universe.  Every capture stores the point-in-time universe snapshot used, raw
IEX one-minute bars, and both long/short feature rows.  It neither reads option
prices nor submits orders.  Discord context and ACE labels remain explicitly
unknown until a later exported-message reconciliation supplies them.

Run every five minutes during regular market hours, for example:

    python -m scripts.discord_ledger.capture_ace_candidate_tape --once

Use a supervisor/scheduler for the repeating process; this program writes one
immutable directory per capture and refuses to overwrite any prior evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import pandas as pd

from scripts.discord_ledger.audit_ace_candidate_tape_coverage import snapshot_as_of, snapshot_index
from scripts.discord_ledger.build import read_jsonl, write_jsonl
from scripts.discord_ledger.spy_trigger_pilot import snapshot


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALERTS = ROOT / "research/discord_ledger_2026-09-21/ace_setup_outcomes_20260923_02/source_alert_outcomes.jsonl"
DEFAULT_SNAPSHOT_DIR = ROOT / "Data/shared/universe/snapshots"
DEFAULT_OUT = ROOT / "Data/inference/ace_candidate_tape"
MAX_IEX_BASIC_SYMBOLS = 30
ET = "America/New_York"


def _utc(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError(f"timezone-aware timestamp required: {value!r}")
    return result.tz_convert("UTC")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_regular_session(now: pd.Timestamp) -> bool:
    local = _utc(now).tz_convert(ET)
    if local.weekday() >= 5:
        return False
    moment = local.hour * 60 + local.minute
    return 9 * 60 + 30 <= moment < 16 * 60


def session_start(now: pd.Timestamp) -> pd.Timestamp:
    local = _utc(now).tz_convert(ET)
    return local.normalize() + pd.Timedelta(hours=9, minutes=30)


def ace_symbols(alerts: Iterable[dict[str, Any]], *, max_symbols: int) -> list[tuple[str, int]]:
    """Deterministic bounded universe from prior source-cited ACE symbols."""
    if max_symbols < 1 or max_symbols > MAX_IEX_BASIC_SYMBOLS:
        raise ValueError(f"max_symbols must be 1..{MAX_IEX_BASIC_SYMBOLS} for the free IEX collection profile")
    counts = Counter(str(row.get("symbol") or "").upper().strip() for row in alerts)
    counts.pop("", None)
    if not counts:
        raise ValueError("no valid ACE source symbols")
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_symbols]


def fetch_iex_session_bars(symbols: list[str], *, start: pd.Timestamp, end: pd.Timestamp,
                           env_file: str | None = ".env") -> pd.DataFrame:
    """Read only IEX one-minute bars; no cached/live execution state is changed."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from core.API.Alpaca_API.core.config import AlpacaConfig

    config = AlpacaConfig.from_env(env_file)
    client = StockHistoricalDataClient(api_key=config.key_id, secret_key=config.secret_key)
    response = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=_utc(start).to_pydatetime(),
        end=_utc(end).to_pydatetime(),
        adjustment=Adjustment.RAW,
        feed=DataFeed.IEX,
        limit=100_000,
    ))
    frame = response.df
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap"])
    frame = frame.reset_index()
    required = {"symbol", "timestamp", "high", "low", "close", "volume", "vwap"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"IEX response missing required bar fields: {sorted(missing)}")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"], kind="stable")
    return frame.drop_duplicates(["symbol", "timestamp"], keep="last").reset_index(drop=True)


def build_candidates(*, bars: pd.DataFrame, selected_symbols: list[tuple[str, int]], decision: pd.Timestamp,
                     capture_at: pd.Timestamp, run_id: str, snapshot_taken: pd.Timestamp,
                     snapshot_path: Path, snapshot_hash: str, eligible_symbols: set[str]) -> list[dict[str, Any]]:
    """Emit both directional candidate rows, retaining unavailable features as unavailable."""
    normalized = bars.copy()
    if not normalized.empty:
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
        normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    rows: list[dict[str, Any]] = []
    for symbol, historical_alert_count in selected_symbols:
        frame = normalized[normalized["symbol"] == symbol].copy() if not normalized.empty else pd.DataFrame()
        base = snapshot(frame, decision) if not frame.empty else {
            "status": "missing_symbol_bars", "latest_completed_bar_start_utc": None, "close": None,
            "return_5m": None, "return_20m": None, "volume_last5_vs_prior20": None,
            "distance_above_opening_range_high_pct": None, "distance_above_session_vwap_pct": None,
        }
        for sign in (1, -1):
            row = {
                "schema": "ace_candidate_tape_v1",
                "run_id": run_id,
                "capture_at_utc": _utc(capture_at).isoformat().replace("+00:00", "Z"),
                "decision_at_utc": _utc(decision).isoformat().replace("+00:00", "Z"),
                "session_date_et": _utc(decision).tz_convert(ET).date().isoformat(),
                "symbol": symbol,
                "direction_sign": sign,
                "direction": "long" if sign == 1 else "short",
                "historical_ace_alert_count": historical_alert_count,
                "universe_snapshot_taken_at_utc": _utc(snapshot_taken).isoformat().replace("+00:00", "Z"),
                "universe_snapshot_path": str(snapshot_path),
                "universe_snapshot_sha256": snapshot_hash,
                "in_snapshot_eligible_universe": symbol in eligible_symbols,
                "market_data_feed": "iex",
                "feature_status": base["status"],
                "latest_completed_bar_start_utc": base["latest_completed_bar_start_utc"],
                "entry_underlying_close": base["close"],
                "directional_return_5m": None if base["return_5m"] is None else sign * float(base["return_5m"]),
                "directional_return_20m": None if base["return_20m"] is None else sign * float(base["return_20m"]),
                "directional_vwap_distance_pct": None if base["distance_above_session_vwap_pct"] is None else sign * float(base["distance_above_session_vwap_pct"]),
                "volume_last5_vs_prior20": base["volume_last5_vs_prior20"],
                "directional_opening_range_high_distance_pct": None if base["distance_above_opening_range_high_pct"] is None else sign * float(base["distance_above_opening_range_high_pct"]),
                "discord_context_status": "not_captured_in_collector",
                "observed_ace_alert": None,
                "observed_ace_message_ids": [],
                "model_score": None,
                "model_score_status": "not_scored_without_certified_live_discord_context",
            }
            rows.append(row)
    return rows


def persist_capture(*, out_root: Path, run_id: str, candidates: list[dict[str, Any]], bars: pd.DataFrame,
                    metadata: dict[str, Any]) -> Path:
    run_dir = out_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(run_dir / "candidates.jsonl", candidates)
    bars.to_parquet(run_dir / "iex_bars.parquet", index=False)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return run_dir


def capture_once(*, now: pd.Timestamp, alerts_path: Path, snapshot_dir: Path, out_root: Path,
                 max_symbols: int, env_file: str | None, fetcher: Callable[..., pd.DataFrame] = fetch_iex_session_bars) -> dict[str, Any]:
    capture_at = _utc(now)
    if not is_regular_session(capture_at):
        return {"status": "outside_regular_session", "capture_at_utc": capture_at.isoformat().replace("+00:00", "Z")}
    alerts = list(read_jsonl(alerts_path))
    selected = ace_symbols(alerts, max_symbols=max_symbols)
    index = snapshot_index(snapshot_dir)
    universe_evidence = snapshot_as_of(index, capture_at)
    if universe_evidence is None:
        raise LookupError("no immutable universe snapshot at or before capture time")
    snapshot_taken, snapshot_path = universe_evidence
    universe = pd.read_csv(snapshot_path)
    if "ticker" not in universe.columns or "is_eligible" not in universe.columns:
        raise ValueError(f"{snapshot_path}: missing ticker or is_eligible")
    eligible = set(universe.loc[universe["is_eligible"].astype(bool), "ticker"].astype(str).str.upper())
    decision = capture_at
    bars = fetcher([symbol for symbol, _ in selected], start=session_start(capture_at).tz_convert("UTC"), end=decision, env_file=env_file)
    run_id = capture_at.strftime("%Y%m%dT%H%M%S.%fZ")
    snapshot_hash = _sha256(snapshot_path)
    candidates = build_candidates(
        bars=bars, selected_symbols=selected, decision=decision, capture_at=capture_at, run_id=run_id,
        snapshot_taken=snapshot_taken, snapshot_path=snapshot_path, snapshot_hash=snapshot_hash,
        eligible_symbols=eligible,
    )
    status_counts = Counter(row["feature_status"] for row in candidates)
    metadata = {
        "schema": "ace_candidate_tape_capture_v1", "research_only": True,
        "run_id": run_id, "capture_at_utc": capture_at.isoformat().replace("+00:00", "Z"),
        "decision_at_utc": decision.isoformat().replace("+00:00", "Z"), "market_data_feed": "iex",
        "alerts_input_path": str(alerts_path), "alerts_input_sha256": _sha256(alerts_path),
        "universe_snapshot_path": str(snapshot_path), "universe_snapshot_sha256": snapshot_hash,
        "universe_snapshot_taken_at_utc": snapshot_taken.isoformat().replace("+00:00", "Z"),
        "selected_symbols": [{"symbol": symbol, "historical_ace_alert_count": count} for symbol, count in selected],
        "candidate_count": len(candidates), "bar_rows": len(bars), "feature_status_counts": dict(status_counts),
        "limitations": [
            "IEX is a single-exchange underlying feed; it is not SIP/consolidated market data.",
            "The bounded universe is derived from prior ACE symbols and is not full-universe discovery.",
            "Discord context and observed ACE labels are intentionally unknown until a later source-export reconciliation.",
            "No option price, quote, fill, contract selection, or order action is captured or inferred.",
        ],
    }
    run_dir = persist_capture(out_root=out_root, run_id=run_id, candidates=candidates, bars=bars, metadata=metadata)
    return {"status": "captured", "run_dir": str(run_dir), "candidate_count": len(candidates),
            "bar_rows": len(bars), "feature_status_counts": dict(status_counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alerts", type=Path, default=DEFAULT_ALERTS)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-symbols", type=int, default=MAX_IEX_BASIC_SYMBOLS)
    parser.add_argument("--env-file", default=".env", help="Optional Alpaca credential file/profile; never stored in output.")
    parser.add_argument("--once", action="store_true", help="Capture once; default is a 5-minute daemon.")
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.interval_seconds < 300:
        raise ValueError("interval-seconds must be >=300 for the free IEX collection profile")
    for path in (args.alerts,):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.snapshot_dir.is_dir():
        raise FileNotFoundError(args.snapshot_dir)
    while True:
        try:
            result = capture_once(now=pd.Timestamp.now(tz="UTC"), alerts_path=args.alerts,
                                  snapshot_dir=args.snapshot_dir, out_root=args.out_root,
                                  max_symbols=args.max_symbols, env_file=args.env_file)
            # A long-lived collector is intentionally quiet overnight. The
            # one-shot command still reports outside-session status for smoke checks.
            if args.once or result["status"] != "outside_regular_session":
                print(json.dumps(result, sort_keys=True))
        except Exception as exc:  # A failure must not create a partial run directory.
            print(json.dumps({"status": "capture_error", "error_type": type(exc).__name__, "message": str(exc)}))
            if args.once:
                raise
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
