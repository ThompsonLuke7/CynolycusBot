"""Frozen, point-in-time caller-alert and underlying-return experiment.

This research deliberately evaluates the *underlying*, not option marks or
claimed P&L.  A positive row is a source-cited call/put entry in the managed
ledger.  A negative row is a directional candidate at a five-minute grid time
in the same caller/symbol/session, at least 30 minutes from every exported
entry alert.  It is a no-nearby-alert control, not a verified losing trade.

The experiment has two independent questions:
* Can a fixed, interpretable classifier distinguish the caller's alert timing
  from matched non-alert candidate times?
* Did the caller's direction have better subsequent underlying movement than
  those controls at fixed 20- and 60-minute horizons?

All features use completed, contiguous one-minute IEX bars strictly before a
decision timestamp.  The bars were fetched retrospectively, so their original
publication time is not certified; the calculation itself does not use future
bars in a feature.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.discord_ledger.build import read_jsonl, write_jsonl
from scripts.discord_ledger.spy_trigger_pilot import snapshot


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACE = ROOT / "research/discord_ledger_2026-09-21/underlying_proxy_20260922_07/underlying_proxy_ledger.jsonl"
DEFAULT_FT = ROOT / "research/discord_ledger_2026-09-21/underlying_proxy_ft_20260922_03/underlying_proxy_ledger.jsonl"
FEATURE_COLUMNS = (
    "directional_return_5m",
    "directional_return_20m",
    "directional_vwap_distance_pct",
    "volume_last5_vs_prior20",
    "directional_opening_range_high_distance_pct",
)
HORIZONS_MINUTES = (20, 60)
CONTROL_GRID_START_ET = (9, 56)
CONTROL_GRID_END_ET = (15, 0)
EXCLUSION_MINUTES = 30
MAX_CONTROL_TIMES_PER_CALLER_SYMBOL_SESSION = 12
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)


@dataclass(frozen=True)
class Alert:
    caller: str
    symbol: str
    decision: pd.Timestamp
    sign: int
    trade_id: str
    message_ids: tuple[str, ...]
    source_file: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _date_et(value: pd.Timestamp) -> str:
    return value.tz_convert("America/New_York").date().isoformat()


def _direction(option_type: object) -> int | None:
    return {"call": 1, "put": -1}.get(str(option_type).lower())


def _load_alerts(path: Path, caller: str) -> tuple[list[Alert], Counter]:
    """Read only entries with a source-mapped, available underlying session."""
    status = Counter()
    alerts: dict[tuple[str, str, pd.Timestamp, int], Alert] = {}
    for row in read_jsonl(path):
        sign = _direction(row.get("option_type"))
        entry = row.get("entry_underlying_snapshot") or {}
        if sign is None:
            status["excluded_missing_call_put_direction"] += 1
            continue
        if entry.get("status") != "ok":
            status[f"excluded_entry_snapshot_{entry.get('status', 'missing')}"] += 1
            continue
        source_file = Path(str(entry.get("source_session_file", "")))
        if not source_file.is_file():
            status["excluded_missing_source_session_file"] += 1
            continue
        try:
            decision = pd.Timestamp(row["entry_available_at_utc"]).tz_convert("UTC")
        except (KeyError, TypeError, ValueError):
            status["excluded_invalid_entry_timestamp"] += 1
            continue
        symbol = str(entry.get("proxy_symbol") or "").upper()
        if not symbol:
            status["excluded_missing_proxy_symbol"] += 1
            continue
        key = (caller, symbol, decision, sign)
        candidate = Alert(caller=caller, symbol=symbol, decision=decision, sign=sign,
                          trade_id=str(row["trade_id"]),
                          message_ids=tuple(sorted(str(item) for item in row.get("source_message_ids", []))),
                          source_file=source_file)
        previous = alerts.get(key)
        if previous is None:
            alerts[key] = candidate
        else:
            # A message can produce multiple ledger life cycles.  For timing
            # imitation it is one directional alert, with all source evidence retained.
            alerts[key] = Alert(caller=caller, symbol=symbol, decision=decision, sign=sign,
                                trade_id="|".join(sorted({*previous.trade_id.split("|"), candidate.trade_id})),
                                message_ids=tuple(sorted({*previous.message_ids, *candidate.message_ids})),
                                source_file=source_file)
            status["deduplicated_same_caller_symbol_time_direction"] += 1
    status["admissible_directional_alerts"] = len(alerts)
    return sorted(alerts.values(), key=lambda value: (value.caller, value.decision, value.symbol, value.sign)), status


def _read_symbol_frame(path: Path, symbol: str, cache: dict[tuple[Path, str], pd.DataFrame]) -> pd.DataFrame:
    key = (path, symbol)
    if key not in cache:
        frame = pd.read_parquet(path)
        required = {"symbol", "timestamp", "high", "low", "close", "volume", "vwap"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        frame = frame[frame["symbol"].astype(str).str.upper() == symbol].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if frame["timestamp"].duplicated().any():
            raise ValueError(f"{path}: duplicate {symbol} minute timestamps")
        if (frame["close"] <= 0).any() or (frame["volume"] < 0).any():
            raise ValueError(f"{path}: invalid {symbol} price or volume")
        # Keeps forward-label lookup O(log n) without changing the feature
        # snapshot implementation or its completed-bar rule.
        frame["_timestamp_ns"] = frame["timestamp"].astype("int64")
        cache[key] = frame
    return cache[key]


def _forward_return(frame: pd.DataFrame, base: dict, sign: int, horizon_minutes: int) -> float | None:
    """Directional close-to-close return, only with a fully contiguous future path."""
    if base.get("status") != "ok" or base.get("close") is None:
        return None
    start = pd.Timestamp(base["latest_completed_bar_start_utc"]).tz_convert("UTC")
    start_ns = start.value
    end_ns = (start + pd.Timedelta(minutes=horizon_minutes)).value
    timestamps_ns = frame.get("_timestamp_ns")
    if timestamps_ns is None:
        timestamps_ns = frame["timestamp"].astype("int64")
    values = timestamps_ns.to_numpy()
    start_position = int(np.searchsorted(values, start_ns))
    end_position = int(np.searchsorted(values, end_ns))
    # Ordered, unique timestamps plus precisely horizon-many positions between
    # endpoints proves each intervening one-minute timestamp is present.
    if (start_position >= len(frame) or end_position >= len(frame)
            or values[start_position] != start_ns or values[end_position] != end_ns
            or end_position - start_position != horizon_minutes):
        return None
    return float(sign * (float(frame.iloc[end_position]["close"]) / float(base["close"]) - 1.0))


def _feature_row(*, kind: str, alert: Alert | None, caller: str, symbol: str,
                 decision: pd.Timestamp, sign: int, source_file: Path,
                 frame: pd.DataFrame) -> dict | None:
    base = snapshot(frame, decision)
    if base["status"] != "ok" or any(base.get(column) is None for column in (
        "return_5m", "return_20m", "distance_above_session_vwap_pct",
        "volume_last5_vs_prior20", "distance_above_opening_range_high_pct",
    )):
        return None
    row = {
        "row_kind": kind,
        "is_alert": int(kind == "alert"),
        "caller": caller,
        "proxy_symbol": symbol,
        "direction": "long" if sign == 1 else "short",
        "direction_sign": sign,
        "decision_at_utc": decision.isoformat(),
        "session_date_et": _date_et(decision),
        "trade_ids": [] if alert is None else alert.trade_id.split("|"),
        "source_message_ids": [] if alert is None else list(alert.message_ids),
        "source_session_file": str(source_file),
        "completed_bar_at_utc": base["latest_completed_bar_start_utc"],
        "entry_underlying_close": base["close"],
        "feature_status": "ok",
        "directional_return_5m": sign * float(base["return_5m"]),
        "directional_return_20m": sign * float(base["return_20m"]),
        "directional_vwap_distance_pct": sign * float(base["distance_above_session_vwap_pct"]),
        "volume_last5_vs_prior20": float(base["volume_last5_vs_prior20"]),
        "directional_opening_range_high_distance_pct": sign * float(base["distance_above_opening_range_high_pct"]),
    }
    for horizon in HORIZONS_MINUTES:
        row[f"directional_forward_return_{horizon}m"] = _forward_return(frame, base, sign, horizon)
    return row


def _control_times(alerts: Iterable[Alert]) -> Iterable[tuple[str, str, Path, pd.Timestamp, int]]:
    """Directional candidates from relevant caller/symbol/session groups."""
    grouped: dict[tuple[str, str, str, Path], list[pd.Timestamp]] = defaultdict(list)
    for alert in alerts:
        grouped[(alert.caller, alert.symbol, _date_et(alert.decision), alert.source_file)].append(alert.decision)
    for (caller, symbol, session_date, source_file), alert_times in grouped.items():
        day = pd.Timestamp(session_date, tz="America/New_York")
        start = day + pd.Timedelta(hours=CONTROL_GRID_START_ET[0], minutes=CONTROL_GRID_START_ET[1])
        end = day + pd.Timedelta(hours=CONTROL_GRID_END_ET[0], minutes=CONTROL_GRID_END_ET[1])
        eligible_times = []
        for decision_et in pd.date_range(start, end, freq="5min"):
            decision = decision_et.tz_convert("UTC")
            if any(abs((decision - alert_time).total_seconds()) <= EXCLUSION_MINUTES * 60 for alert_time in alert_times):
                continue
            eligible_times.append(decision)
        # Fixed, evenly-spaced subsampling limits a few heavily-alerted days
        # from overwhelming the caller's regular universe.  It is selected
        # before features/outcomes are calculated and is not performance-tuned.
        if len(eligible_times) > MAX_CONTROL_TIMES_PER_CALLER_SYMBOL_SESSION:
            positions = np.linspace(0, len(eligible_times) - 1,
                                    MAX_CONTROL_TIMES_PER_CALLER_SYMBOL_SESSION, dtype=int)
            eligible_times = [eligible_times[position] for position in positions]
        for decision in eligible_times:
            yield caller, symbol, source_file, decision, 1
            yield caller, symbol, source_file, decision, -1


def build_samples(alerts: list[Alert]) -> tuple[list[dict], dict]:
    """Build source-cited alerts and matched directional candidate controls."""
    cache: dict[tuple[Path, str], pd.DataFrame] = {}
    samples: list[dict] = []
    usable_alerts: list[Alert] = []
    exclusions = Counter()
    for alert in alerts:
        frame = _read_symbol_frame(alert.source_file, alert.symbol, cache)
        row = _feature_row(kind="alert", alert=alert, caller=alert.caller, symbol=alert.symbol,
                           decision=alert.decision, sign=alert.sign, source_file=alert.source_file, frame=frame)
        if row is None:
            exclusions["alert_feature_snapshot_unavailable"] += 1
        else:
            samples.append(row)
            usable_alerts.append(alert)
    # Outcome controls match the observed caller directional mix within a
    # caller/symbol/session.  The alert-imitation classifier still receives
    # both directions at every control timestamp, by design.
    outcome_weights = Counter((alert.caller, alert.symbol, _date_et(alert.decision), alert.sign)
                              for alert in usable_alerts)
    outcome_sources: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
    for alert in usable_alerts:
        outcome_sources[(alert.caller, alert.symbol, _date_et(alert.decision), alert.sign)].update(alert.message_ids)
    seen_controls: set[tuple[str, str, pd.Timestamp, int]] = set()
    for caller, symbol, source_file, decision, sign in _control_times(usable_alerts):
        key = (caller, symbol, decision, sign)
        if key in seen_controls:
            continue
        seen_controls.add(key)
        frame = _read_symbol_frame(source_file, symbol, cache)
        row = _feature_row(kind="matched_no_nearby_alert_control", alert=None, caller=caller, symbol=symbol,
                           decision=decision, sign=sign, source_file=source_file, frame=frame)
        if row is None:
            exclusions["control_feature_snapshot_unavailable"] += 1
        else:
            outcome_key = (caller, symbol, _date_et(decision), sign)
            row["outcome_control_weight"] = int(outcome_weights[outcome_key])
            row["matched_alert_direction_source_message_ids"] = sorted(outcome_sources[outcome_key])
            samples.append(row)
    samples.sort(key=lambda row: (row["caller"], row["decision_at_utc"], row["proxy_symbol"], row["direction_sign"], row["is_alert"]))
    return samples, {
        "admissible_alerts_with_features": sum(row["is_alert"] for row in samples),
        "matched_no_nearby_alert_directional_controls_with_features": sum(not row["is_alert"] for row in samples),
        "excluded": dict(exclusions),
        "source_session_files_read": len({row["source_session_file"] for row in samples}),
    }


def _date_splits(rows: list[dict]) -> dict[str, set[str]]:
    dates = sorted({row["session_date_et"] for row in rows})
    if len(dates) < 5:
        raise ValueError(f"Need >=5 independent session dates for chronological evaluation; got {len(dates)}")
    n_train = max(1, int(len(dates) * SPLIT_FRACTIONS[0]))
    n_validation = max(1, int(len(dates) * SPLIT_FRACTIONS[1]))
    if n_train + n_validation >= len(dates):
        n_validation = 1
        n_train = len(dates) - 2
    return {"train": set(dates[:n_train]), "validation": set(dates[n_train:n_train + n_validation]),
            "test": set(dates[n_train + n_validation:])}


def _outcome_summary(rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for horizon in HORIZONS_MINUTES:
        column = f"directional_forward_return_{horizon}m"
        data = {}
        for kind, kind_rows in (("alerts", [row for row in rows if row["is_alert"]]),
                                ("matched_controls", [row for row in rows if not row["is_alert"]
                                                      and row.get("outcome_control_weight", 0) > 0])):
            values = np.array([row[column] for row in kind_rows if row[column] is not None], dtype=float)
            weights = np.array([row.get("outcome_control_weight", 1) for row in kind_rows if row[column] is not None], dtype=float)
            # Alert rows have unit weight.  Controls use source-cited alert
            # counts to reproduce each caller/symbol/session directional mix.
            data[kind] = {"n": int(len(values)), "weighted_n": float(weights.sum()),
                          "mean_pct": None if not len(values) else float(np.average(values, weights=weights) * 100),
                          "median_pct": None if not len(values) else float(np.median(values) * 100),
                          "directional_hit_rate": None if not len(values) else float(np.average(values > 0, weights=weights))}
        if data["alerts"]["mean_pct"] is not None and data["matched_controls"]["mean_pct"] is not None:
            data["alert_minus_control_mean_pct"] = data["alerts"]["mean_pct"] - data["matched_controls"]["mean_pct"]
            data["alert_minus_control_hit_rate"] = data["alerts"]["directional_hit_rate"] - data["matched_controls"]["directional_hit_rate"]
        else:
            data["alert_minus_control_mean_pct"] = None
            data["alert_minus_control_hit_rate"] = None
        result[f"{horizon}m"] = data
    return result


def _evaluate_caller(rows: list[dict], caller: str) -> dict:
    caller_rows = [row for row in rows if row["caller"] == caller]
    splits = _date_splits(caller_rows)
    for row in caller_rows:
        row["split"] = next(name for name, dates in splits.items() if row["session_date_et"] in dates)
    partitions = {name: [row for row in caller_rows if row["split"] == name] for name in splits}
    train = partitions["train"]
    x_train = np.array([[row[column] for column in FEATURE_COLUMNS] for row in train], dtype=float)
    y_train = np.array([row["is_alert"] for row in train], dtype=int)
    model_report: dict[str, object] = {"status": "not_fit"}
    if len(set(y_train)) == 2:
        model = Pipeline([("scale", StandardScaler()),
                          ("logistic", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=7))])
        model.fit(x_train, y_train)
        model_report = {"status": "fit_fixed_logistic_regression",
                        "hyperparameters": {"C": 1.0, "class_weight": "balanced", "random_state": 7},
                        "standardized_coefficients": {column: float(value) for column, value in zip(
                            FEATURE_COLUMNS, model.named_steps["logistic"].coef_[0])}}
        for partition_name, partition_rows in partitions.items():
            if not partition_rows:
                model_report[partition_name] = {"n": 0}
                continue
            x = np.array([[row[column] for column in FEATURE_COLUMNS] for row in partition_rows], dtype=float)
            y = np.array([row["is_alert"] for row in partition_rows], dtype=int)
            probability = model.predict_proba(x)[:, 1]
            metrics = {"n": int(len(y)), "alert_count": int(y.sum()), "alert_rate": float(y.mean()),
                       "average_precision": None, "roc_auc": None,
                       "precision_at_threshold_0_5": None, "recall_at_threshold_0_5": None}
            if len(set(y)) == 2:
                metrics["average_precision"] = float(average_precision_score(y, probability))
                metrics["roc_auc"] = float(roc_auc_score(y, probability))
                predicted = probability >= 0.5
                metrics["precision_at_threshold_0_5"] = float(y[predicted].mean()) if predicted.any() else 0.0
                metrics["recall_at_threshold_0_5"] = float(predicted[y == 1].mean()) if (y == 1).any() else None
            model_report[partition_name] = metrics
    return {
        "caller": caller,
        "fixed_design": {"features": list(FEATURE_COLUMNS), "forward_horizons_minutes": list(HORIZONS_MINUTES),
                         "control_grid_et": f"{CONTROL_GRID_START_ET[0]:02d}:{CONTROL_GRID_START_ET[1]:02d}-"
                                            f"{CONTROL_GRID_END_ET[0]:02d}:{CONTROL_GRID_END_ET[1]:02d} every 5 minutes",
                         "max_evenly_spaced_control_times_per_caller_symbol_session": MAX_CONTROL_TIMES_PER_CALLER_SYMBOL_SESSION,
                         "same_symbol_alert_exclusion_minutes": EXCLUSION_MINUTES,
                         "chronological_split_fractions": list(SPLIT_FRACTIONS)},
        "split_session_dates": {name: sorted(dates) for name, dates in splits.items()},
        "counts": {name: {"rows": len(partition), "alerts": sum(row["is_alert"] for row in partition),
                           "controls": sum(not row["is_alert"] for row in partition)}
                   for name, partition in partitions.items()},
        "alert_imitation_model": model_report,
        "independent_underlying_outcome_comparison": {name: _outcome_summary(partition)
                                                        for name, partition in partitions.items()},
    }


def _report(results: dict) -> str:
    lines = ["# Caller underlying-direction experiment", "",
             "This is an underlying-only, research-only result. It does not estimate option fills, option P&L, or prove a trade was executable.", "",
             "Controls are same-caller/symbol/session five-minute directional candidates at least 30 minutes from an exported entry alert. Outcome comparisons weight their directions to the source-cited alert mix within each caller/symbol/session. They are not verified losing trades.", ""]
    for caller, result in results["callers"].items():
        lines.extend([f"## {caller}", "", "| Test metric | 20m | 60m |", "| --- | ---: | ---: |"])
        outcome = result["independent_underlying_outcome_comparison"]["test"]
        for label, field in (("Alert n", "n"), ("Alert mean directional return (%)", "mean_pct"),
                             ("Alert directional hit rate", "directional_hit_rate"),
                             ("Control mean directional return (%)", "mean_pct"),
                             ("Alert − control mean (%)", "alert_minus_control_mean_pct")):
            values = []
            for horizon in ("20m", "60m"):
                source = outcome[horizon]
                if label.startswith("Control"):
                    value = source["matched_controls"][field]
                elif label.startswith("Alert −"):
                    value = source[field]
                else:
                    value = source["alerts"][field]
                values.append("n/a" if value is None else (str(value) if field == "n" else f"{value:.4f}"))
            lines.append(f"| {label} | {values[0]} | {values[1]} |")
        model = result["alert_imitation_model"]
        test = model.get("test", {}) if isinstance(model, dict) else {}
        lines.extend(["", f"Fixed alert-imitation model status: `{model.get('status')}`. "
                      f"Test ROC AUC: `{test.get('roc_auc')}`; test average precision: `{test.get('average_precision')}`.", ""])
    lines.extend(["## Interpretation boundary", "",
                  "A positive alert-minus-control result is evidence that the exported alerts had more favorable underlying movement than matched quiet windows in this sample. It is not evidence that a learned rule will remain profitable or that historical option pricing would have matched the caller. Any strategy recreation must be validated on a later, untouched time period with realistic execution assumptions.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ace-ledger", type=Path, default=DEFAULT_ACE)
    parser.add_argument("--ft-ledger", type=Path, default=DEFAULT_FT)
    parser.add_argument("--callers", nargs="+", choices=("ACE", "FT"), default=("ACE", "FT"),
                        help="Caller-specific runs are independent and use identical frozen rules.")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; must not already exist.")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {args.out}")
    alerts_by_caller, load_status = {}, {}
    ledger_paths = {"ACE": args.ace_ledger, "FT": args.ft_ledger}
    for caller in args.callers:
        path = ledger_paths[caller]
        if not path.is_file():
            raise FileNotFoundError(path)
        alerts_by_caller[caller], load_status[caller] = _load_alerts(path, caller)
    all_alerts = [alert for caller_alerts in alerts_by_caller.values() for alert in caller_alerts]
    samples, sample_summary = build_samples(all_alerts)
    results = {"schema": "caller_underlying_experiment_v1", "research_only": True,
               "ledger_inputs": {caller: {"path": str(ledger_paths[caller]), "sha256": _sha256(ledger_paths[caller])}
                                 for caller in args.callers},
               "market_data_limitations": ["IEX underlying trade bars were fetched retrospectively; original publication timing, quote state, and adjustments are not certified.",
                                           "This analysis does not use option prices, option fills, quantities, or P&L."],
               "control_label_warning": "No nearby exported entry alert is not a verified non-trade, losing trade, or full representation of the caller's watchlist.",
               "alert_load_status": {caller: dict(counter) for caller, counter in load_status.items()},
               "sample_build_summary": sample_summary,
               "callers": {caller: _evaluate_caller(samples, caller) for caller in args.callers}}
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "samples.jsonl", samples)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(_report(results), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "sample_build_summary": sample_summary,
                      "test_outcomes": {caller: results["callers"][caller]["independent_underlying_outcome_comparison"]["test"]
                                        for caller in args.callers}}, indent=2))


if __name__ == "__main__":
    main()
