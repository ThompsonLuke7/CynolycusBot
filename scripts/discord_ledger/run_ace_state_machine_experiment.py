"""Evaluate a fixed, screenshot-derived ACE-style state-machine hypothesis.

This is deliberately a research-only, *selection-conditioned* experiment.
Every candidate already has ACE's observed symbol and direction injected.  The
experiment asks only whether a small set of point-in-time daily level and
intraday-confirmation states occur more often at observed ACE alert times than
at frozen same-symbol, same-session quiet-window candidates.  It does not
discover a universe, recreate ACE's proprietary cycle indicator, validate
options P&L, or submit orders.

The rules are fixed from the supplied screenshot interpretation before this
run.  They must not be retuned on the held-out test partition.  Because the
hypothesis itself was formulated after inspecting Discord material, even its
test result is exploratory; a prospective immutable candidate tape is the
required independent confirmation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.discord_ledger.build import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALERTS = ROOT / "research/discord_ledger_2026-09-21/ace_setup_outcomes_20260923_02/source_alert_outcomes.jsonl"
DEFAULT_CONTROLS = ROOT / "research/discord_ledger_2026-09-21/ace_setup_outcomes_20260923_02/matched_controls.jsonl"
DEFAULT_DAILY_BARS = ROOT / "Data/shared/bars/1d"

# Fixed from the screenshot-derived hypothesis, not selected on outcomes.
MIN_DAILY_HISTORY = 55
ATR_WINDOW = 14
EXTREME_WINDOW = 20
PULLBACK_MIN_ATR = 0.25
PULLBACK_MAX_ATR = 5.0
APPROACHING_LEVEL_ATR = 0.75
AT_LEVEL_ATR = 0.35
BOOTSTRAP_DRAWS = 5_000
RNG_SEED = 2007
OUTCOME_HORIZONS = (15, 30, 60)
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"timezone-aware timestamp required: {value!r}")
    return timestamp.tz_convert("UTC")


def _source_group(row: dict[str, Any]) -> tuple[str, ...]:
    ids = tuple(sorted(map(str, row.get("source_message_ids") or [])))
    if not ids:
        raise ValueError("candidate missing source_message_ids; cannot make an auditable control group")
    return ids


def _date_splits(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    dates = sorted({str(row["session_date_et"]) for row in rows})
    if len(dates) < 5:
        raise ValueError(f"need at least five session dates, got {len(dates)}")
    train_n = max(1, int(len(dates) * SPLIT_FRACTIONS[0]))
    validation_n = max(1, int(len(dates) * SPLIT_FRACTIONS[1]))
    if train_n + validation_n >= len(dates):
        validation_n, train_n = 1, len(dates) - 2
    return {
        "train": set(dates[:train_n]),
        "validation": set(dates[train_n:train_n + validation_n]),
        "test": set(dates[train_n + validation_n:]),
    }


def prepare_daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate daily OHLCV and add the ET session date without altering rows."""
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"daily bars missing columns: {sorted(missing)}")
    daily = frame.copy()
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True, errors="raise")
    if daily["timestamp"].duplicated().any() or not daily["timestamp"].is_monotonic_increasing:
        raise ValueError("daily timestamps must be unique and increasing")
    for column in ("open", "high", "low", "close"):
        daily[column] = pd.to_numeric(daily[column], errors="raise")
        if (~np.isfinite(daily[column])).any() or (daily[column] <= 0).any():
            raise ValueError(f"invalid daily {column}")
    if (daily["high"] < daily["low"]).any():
        raise ValueError("daily high is below low")
    daily["session_date_et"] = daily["timestamp"].dt.tz_convert("America/New_York").dt.date
    return daily


def daily_structure(
    daily: pd.DataFrame, *, decision_at_utc: object, entry_price: float, direction_sign: int,
) -> dict[str, Any]:
    """Compute daily levels using only sessions completed strictly before decision ET date."""
    decision = _utc(decision_at_utc)
    decision_date = decision.tz_convert("America/New_York").date()
    history = daily[daily["session_date_et"] < decision_date].copy()
    base: dict[str, Any] = {
        "daily_feature_status": "insufficient_completed_daily_history",
        "completed_daily_sessions_used": int(len(history)),
        "latest_completed_daily_session_et": None if history.empty else history.iloc[-1]["session_date_et"].isoformat(),
        "sma_5": None, "ema_21": None, "sma_34": None, "sma_55": None,
        "atr_14": None, "nearest_level": None, "nearest_level_abs_atr": None,
        "directional_pullback_from_20d_extreme_atr": None,
        "trend_supportive": None, "pullback_due": None,
    }
    if len(history) < MIN_DAILY_HISTORY:
        return base
    close = history["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat([
        history["high"] - history["low"],
        (history["high"] - previous_close).abs(),
        (history["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean().iloc[-1]
    if not np.isfinite(atr) or float(atr) <= 0:
        base["daily_feature_status"] = "invalid_atr"
        return base
    levels = {
        "sma_5": float(close.rolling(5, min_periods=5).mean().iloc[-1]),
        "ema_21": float(close.ewm(span=21, adjust=False, min_periods=21).mean().iloc[-1]),
        "sma_34": float(close.rolling(34, min_periods=34).mean().iloc[-1]),
        "sma_55": float(close.rolling(55, min_periods=55).mean().iloc[-1]),
    }
    price = float(entry_price)
    if not np.isfinite(price) or price <= 0 or direction_sign not in {-1, 1}:
        raise ValueError("invalid entry price or direction")
    nearest_name, nearest_value = min(levels.items(), key=lambda item: abs(price - item[1]))
    recent_high = float(history["high"].tail(EXTREME_WINDOW).max())
    recent_low = float(history["low"].tail(EXTREME_WINDOW).min())
    # Positive means a retracement from the directionally relevant 20-day extreme.
    pullback = (recent_high - price) / float(atr) if direction_sign == 1 else (price - recent_low) / float(atr)
    # This is a diagnostic trend proxy, not ACE's displayed private BULL/BEAR field.
    trend = (
        levels["ema_21"] >= levels["sma_55"] and levels["sma_5"] >= levels["ema_21"]
        if direction_sign == 1
        else levels["ema_21"] <= levels["sma_55"] and levels["sma_5"] <= levels["ema_21"]
    )
    base.update(levels)
    base.update({
        "daily_feature_status": "ok", "atr_14": float(atr), "nearest_level": nearest_name,
        "nearest_level_abs_atr": float(abs(price - nearest_value) / atr),
        "directional_pullback_from_20d_extreme_atr": float(pullback),
        "trend_supportive": bool(trend),
        "pullback_due": bool(PULLBACK_MIN_ATR <= pullback <= PULLBACK_MAX_ATR),
    })
    return base


def unavailable_daily_structure(status: str) -> dict[str, Any]:
    """A complete fail-closed schema for symbols with no usable daily source."""
    return {
        "daily_feature_status": status, "completed_daily_sessions_used": 0,
        "latest_completed_daily_session_et": None, "sma_5": None, "ema_21": None,
        "sma_34": None, "sma_55": None, "atr_14": None, "nearest_level": None,
        "nearest_level_abs_atr": None, "directional_pullback_from_20d_extreme_atr": None,
        "trend_supportive": None, "pullback_due": None,
    }


def state_machine(daily: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Apply fixed sequential states; missing inputs fail closed and are recorded."""
    base = {
        "candidate": daily["daily_feature_status"] == "ok",
        "trending": False, "pullback_due_state": False, "approaching_level": False,
        "at_level": False, "confirmed": False, "trade_hypothesis": False,
        "intraday_confirmation": None,
    }
    if not base["candidate"]:
        return base
    base["trending"] = bool(daily["trend_supportive"])
    base["pullback_due_state"] = bool(base["trending"] and daily["pullback_due"])
    base["approaching_level"] = bool(
        base["pullback_due_state"] and daily["nearest_level_abs_atr"] <= APPROACHING_LEVEL_ATR
    )
    base["at_level"] = bool(
        base["pullback_due_state"] and daily["nearest_level_abs_atr"] <= AT_LEVEL_ATR
    )
    required = ("directional_return_5m", "directional_vwap_distance_pct", "volume_last5_vs_prior20")
    if any(row.get(column) is None or not np.isfinite(float(row[column])) for column in required):
        return base
    intraday = bool(
        float(row["directional_return_5m"]) > 0
        and float(row["directional_vwap_distance_pct"]) > 0
    )
    base["intraday_confirmation"] = intraday
    base["confirmed"] = bool(base["at_level"] and intraday)
    base["trade_hypothesis"] = bool(base["confirmed"] and float(row["volume_last5_vs_prior20"]) >= 1.0)
    return base


def enrich_candidates(
    alerts: Iterable[dict[str, Any]], controls: Iterable[dict[str, Any]], daily_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Add audited daily structure/state fields; never fill absent data."""
    rows: list[dict[str, Any]] = []
    cache: dict[str, pd.DataFrame | None] = {}
    hashes: dict[str, str] = {}
    for kind, source_rows in (("alert", alerts), ("control", controls)):
        for source in source_rows:
            row = dict(source)
            symbol = str(row["symbol"]).upper()
            path = daily_dir / f"{symbol}.parquet"
            if symbol not in cache:
                if not path.is_file():
                    cache[symbol] = None
                else:
                    cache[symbol] = prepare_daily_bars(pd.read_parquet(path))
                    hashes[str(path)] = _sha256(path)
            if cache[symbol] is None:
                structure = unavailable_daily_structure("missing_daily_bars")
            else:
                structure = daily_structure(
                    cache[symbol], decision_at_utc=row["decision_at_utc"],
                    entry_price=float(row["entry_underlying_close"]), direction_sign=int(row["direction_sign"]),
                )
            row.update(structure)
            row.update(state_machine(structure, row))
            row["is_observed_alert"] = int(kind == "alert")
            row["candidate_group_message_ids"] = list(_source_group(row))
            rows.append(row)
    rows.sort(key=lambda r: (r["session_date_et"], r["decision_at_utc"], r["is_observed_alert"]))
    validate_candidates(rows)
    return rows, hashes


def validate_candidates(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["is_observed_alert"]) not in {0, 1}:
            raise ValueError("is_observed_alert must be binary")
        grouped[tuple(row["candidate_group_message_ids"])].append(row)
    for group, members in grouped.items():
        if sum(int(row["is_observed_alert"]) for row in members) != 1:
            raise ValueError(f"group {group} does not contain exactly one observed alert")
        if len({row["session_date_et"] for row in members}) != 1:
            raise ValueError(f"group {group} spans multiple session dates")


def _bootstrap_ci(values: np.ndarray) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(RNG_SEED)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(BOOTSTRAP_DRAWS)])
    return [float(np.quantile(means, 0.025) * 100), float(np.quantile(means, 0.975) * 100)]


def _mde_pct(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    return float(z * np.std(values, ddof=1) / np.sqrt(len(values)) * 100)


def state_metrics(rows: list[dict[str, Any]], state: str) -> dict[str, Any]:
    """Group-balanced observed-alert reproduction and post-state outcomes."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row["candidate_group_message_ids"])].append(row)
    alert_state: list[float] = []
    control_state: list[float] = []
    state_delta: list[float] = []
    outcome: dict[str, dict[str, list[float]]] = {
        f"{horizon}m": {"alert": [], "control": [], "delta": []} for horizon in OUTCOME_HORIZONS
    }
    for members in groups.values():
        alerts = [row for row in members if row["is_observed_alert"]]
        controls = [row for row in members if not row["is_observed_alert"]]
        if len(alerts) != 1 or not controls:
            continue
        alert = alerts[0]
        a_pass = float(bool(alert[state]))
        c_pass = float(np.mean([bool(row[state]) for row in controls]))
        alert_state.append(a_pass)
        control_state.append(c_pass)
        state_delta.append(a_pass - c_pass)
        # Outcome comparison is deliberately conditional on this rule firing
        # on both sides of a source group; it is descriptive, not a backtest.
        for horizon in OUTCOME_HORIZONS:
            column = f"directional_forward_return_{horizon}m"
            selected_controls = [float(row[column]) for row in controls if row[state] and row.get(column) is not None]
            alert_value = alert.get(column)
            if a_pass and alert_value is not None and selected_controls:
                outcome[f"{horizon}m"]["alert"].append(float(alert_value))
                outcome[f"{horizon}m"]["control"].append(float(np.mean(selected_controls)))
                outcome[f"{horizon}m"]["delta"].append(float(alert_value) - float(np.mean(selected_controls)))
    result: dict[str, Any] = {
        "paired_source_groups": len(state_delta),
        "observed_alert_pass_rate": float(np.mean(alert_state)) if alert_state else None,
        "matched_control_group_weighted_pass_rate": float(np.mean(control_state)) if control_state else None,
        "alert_minus_control_pass_rate_pct": float(np.mean(state_delta) * 100) if state_delta else None,
        "pass_rate_difference_bootstrap_95pct_ci_pct": _bootstrap_ci(np.array(state_delta, dtype=float)),
        "pass_rate_difference_mde_pct": _mde_pct(np.array(state_delta, dtype=float)),
        "outcome_conditioning_warning": "Outcome rows require both an alert and at least one control to pass this state; they are descriptive, selection-conditioned underlying returns, not option P&L or a tradable backtest.",
        "outcomes": {},
    }
    for horizon, values in outcome.items():
        delta = np.array(values["delta"], dtype=float)
        result["outcomes"][horizon] = {
            "paired_groups_passing_state": int(len(delta)),
            "alert_mean_pct": None if not values["alert"] else float(np.mean(values["alert"]) * 100),
            "matched_control_mean_pct": None if not values["control"] else float(np.mean(values["control"]) * 100),
            "alert_minus_control_mean_pct": None if not len(delta) else float(np.mean(delta) * 100),
            "bootstrap_95pct_ci_pct": _bootstrap_ci(delta),
            "minimum_detectable_effect_pct": _mde_pct(delta),
        }
    return result


STATE_ORDER = ("candidate", "trending", "pullback_due_state", "approaching_level", "at_level", "confirmed", "trade_hypothesis")


def run_experiment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    splits = _date_splits(rows)
    partitions = {name: [row for row in rows if row["session_date_et"] in dates] for name, dates in splits.items()}
    return {
        "schema": "ace_fixed_state_machine_experiment_v1",
        "research_only": True,
        "selection_target": "observed ACE alert timing within already-selected same-symbol/direction candidate groups",
        "hypothesis_status": "exploratory screenshot-derived fixed rule set; no outcome fitting or grid search",
        "chronological_split_dates": {name: sorted(days) for name, days in splits.items()},
        "partition_counts": {
            name: {"candidates": len(partition), "observed_alerts": sum(r["is_observed_alert"] for r in partition),
                   "unlabeled_controls": sum(1 - r["is_observed_alert"] for r in partition)}
            for name, partition in partitions.items()
        },
        "daily_feature_coverage": {
            name: {
                status: sum(row["daily_feature_status"] == status for row in partition)
                for status in sorted({str(row["daily_feature_status"]) for row in partition})
            }
            for name, partition in partitions.items()
        },
        "fixed_rule_parameters": {
            "daily_history_sessions": MIN_DAILY_HISTORY, "atr_window_sessions": ATR_WINDOW,
            "pullback_extreme_window_sessions": EXTREME_WINDOW, "pullback_atr_range": [PULLBACK_MIN_ATR, PULLBACK_MAX_ATR],
            "approaching_level_abs_atr_max": APPROACHING_LEVEL_ATR, "at_level_abs_atr_max": AT_LEVEL_ATR,
            "intraday_confirmation": "directional 5-minute return > 0 AND directional session-VWAP distance > 0",
            "trade_hypothesis": "confirmed AND last-5-minute volume / prior-20-minute volume >= 1",
        },
        "state_definitions": {
            "candidate": "at least 55 completed prior ET daily sessions and valid ATR",
            "trending": "direction-aware SMA5/EMA21/SMA55 ordering diagnostic",
            "pullback_due_state": "trending and 0.25 to 5 ATR retracement from directionally relevant prior 20-day extreme",
            "approaching_level": "pullback state and nearest of SMA5/EMA21/SMA34/SMA55 within 0.75 ATR",
            "at_level": "pullback state and nearest listed daily average within 0.35 ATR",
            "confirmed": "at-level plus favorable completed 5-minute return and VWAP position",
            "trade_hypothesis": "confirmed plus short-volume expansion; it is not an order instruction",
        },
        "states": {name: {state: state_metrics(partition, state) for state in STATE_ORDER} for name, partition in partitions.items()},
        "limitations": [
            "The candidate panel injects ACE's historical symbol and direction; this cannot test theme selection, universe discovery, or direction choice.",
            "Controls are no-nearby-alert candidates, not known losing or invalid setups.",
            "The visible Cycle/Due-low/D EM/divergence values are not reconstructed because the export does not provide their source calculation; this uses a separately named ordinary pullback proxy.",
            "Daily inputs are restricted to ET sessions strictly before the alert date; intraday features are precomputed completed-bar fields from the frozen source artifact.",
            "Forward outcomes are underlying directional returns only; no options fills, contract prices, or P&L are inferred.",
        ],
    }


def _fmt(value: object, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ACE fixed state-machine experiment", "",
        "Research-only, selection-conditioned test of the screenshot-derived daily-level/pullback/confirmation hypothesis. It contains **no fitted model, grid search, option repricing, or trade execution**. Its candidate rows already contain ACE's historical symbol and direction, so it does not recreate candidate selection.", "",
        "## Point-in-time design", "",
        "- Daily SMA5, EMA21, SMA34, SMA55 and ATR are calculated only from ET sessions strictly before each decision date. The decision day's daily bar is excluded.",
        "- The hypothesis uses an ordinary 20-day-extreme pullback proxy. It does not claim to reproduce ACE's private Cycle/Due low/D EM or divergence indicators.",
        "- Observed alerts are compared with frozen, same-symbol/session/direction no-nearby-alert candidates. Those controls are unlabeled—not proven bad trades.",
        "- Rules were fixed from the pasted screenshot interpretation before this run and were not outcome-fitted. The hypothesis itself is still exploratory because it was formed after viewing Discord material.", "",
        "## Daily-data coverage", "",
        "Rows with missing daily bars or insufficient prior daily history are retained in the candidate file and fail all downstream states; they are not replaced or silently discarded.", "",
        "| Partition | Daily feature status | Rows |", "| --- | --- | ---: |",
    ]
    for partition, statuses in report["daily_feature_coverage"].items():
        for status, count in statuses.items():
            lines.append(f"| {partition} | {status} | {count} |")
    lines.extend(["", "## Fixed state rules", ""])
    for state, definition in report["state_definitions"].items():
        lines.append(f"- `{state}`: {definition}.")
    lines.extend(["", "## Held-out state reproduction", "", "The test partition is chronological. Positive pass-rate lift means this state occurred more often at an observed alert than in its matched quiet-window controls; it does not establish profitability.", "", "| State | Groups | Alert pass | Control pass | Difference (pp) | 95% CI (pp) | MDE (pp) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for state in STATE_ORDER:
        value = report["states"]["test"][state]
        ci = value["pass_rate_difference_bootstrap_95pct_ci_pct"]
        ci_text = "—" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(f"| {state} | {value['paired_source_groups']} | {_fmt(value['observed_alert_pass_rate'])} | {_fmt(value['matched_control_group_weighted_pass_rate'])} | {_fmt(value['alert_minus_control_pass_rate_pct'])} | {ci_text} | {_fmt(value['pass_rate_difference_mde_pct'])} |")
    lines.extend(["", "## Held-out underlying returns when both sides passed", "", "These are a sparse diagnostic only: a group contributes only when the alert and at least one matching control both pass the named state. Returns are directional underlying returns, not option P&L. An MDE larger than the observed difference means the sample cannot measure a plausible effect.", ""])
    for state in ("at_level", "confirmed", "trade_hypothesis"):
        lines.extend([f"### `{state}`", "", "| Horizon | Paired groups | Alert mean (%) | Control mean (%) | Difference (%) | 95% CI (%) | MDE (%) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for horizon, value in report["states"]["test"][state]["outcomes"].items():
            ci = value["bootstrap_95pct_ci_pct"]
            ci_text = "—" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            lines.append(f"| {horizon} | {value['paired_groups_passing_state']} | {_fmt(value['alert_mean_pct'])} | {_fmt(value['matched_control_mean_pct'])} | {_fmt(value['alert_minus_control_mean_pct'])} | {ci_text} | {_fmt(value['minimum_detectable_effect_pct'])} |")
        lines.append("")
    lines.extend(["## Interpretation boundary", "", "A positive timing association would only support this small state proxy inside the already-known candidate panel. A null with a large MDE is ‘not measurable here,’ not proof that daily levels do not matter. The next testable variant is to collect actual ACE level/cycle screenshots or structured values, then prospectively score an immutable, independently generated candidate tape across both directions before any integration into a live strategy."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alerts", type=Path, default=DEFAULT_ALERTS)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--daily-bars", type=Path, default=DEFAULT_DAILY_BARS)
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwritten.")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out}")
    for path in (args.alerts, args.controls):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.daily_bars.is_dir():
        raise NotADirectoryError(args.daily_bars)
    rows, daily_hashes = enrich_candidates(read_jsonl(args.alerts), read_jsonl(args.controls), args.daily_bars)
    report = run_experiment(rows)
    report["inputs"] = {str(args.alerts): _sha256(args.alerts), str(args.controls): _sha256(args.controls), **daily_hashes}
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "candidates_with_states.jsonl", rows)
    (args.out / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    write_report(args.out / "report.md", report)
    print(json.dumps({"out": str(args.out), "partitions": report["partition_counts"], "held_out_states": report["states"]["test"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
