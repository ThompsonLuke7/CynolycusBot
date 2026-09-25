"""Chronological ACE alert-timing experiment with positive-unlabeled controls.

This is deliberately a *selection-proxy* experiment, not a profitability or
trade-execution backtest.  Source-cited ACE entry alerts are observed
positives.  The same-symbol/session no-nearby-alert candidates are unlabeled:
they may include unexported setups or real opportunities.  Consequently,
``P(observed alert | features)`` is useful only to rank this frozen candidate
panel; it is not the probability that a candidate is a profitable trade.

All model inputs are pre-decision underlying structure plus Discord watchlist
state with conservative message availability.  Underlying forward returns are
kept entirely separate for held-out descriptive evaluation.  No option marks,
fills, quantities, source P&L, article-time news, or universe-membership
fields enter either model.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from scripts.discord_ledger.build import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALERTS = ROOT / "research/discord_ledger_2026-09-21/ace_setup_outcomes_20260923_02/source_alert_outcomes.jsonl"
DEFAULT_CONTROLS = ROOT / "research/discord_ledger_2026-09-21/ace_setup_outcomes_20260923_02/matched_controls.jsonl"
DEFAULT_CONTEXT = ROOT / "research/discord_ledger_2026-09-21/ace_context_selectivity_20260923_01/candidate_context.jsonl"

PRICE_FEATURES = (
    "directional_return_5m",
    "directional_return_20m",
    "directional_vwap_distance_pct",
    "volume_last5_vs_prior20",
    "directional_opening_range_high_distance_pct",
)
WATCHLIST_FEATURES = (
    "watchlist_24h",
    "watchlist_7d",
    "watchlist_level_7d",
    "watchlist_intention_or_instruction_7d",
)
TIME_FEATURES = ("session_time_sin", "session_time_cos")
MODEL_SPECS = {
    "price_structure_logistic": {"kind": "logistic", "features": PRICE_FEATURES},
    "price_time_watchlist_logistic": {"kind": "logistic", "features": PRICE_FEATURES + TIME_FEATURES + WATCHLIST_FEATURES},
    "price_time_watchlist_hgb": {"kind": "hgb", "features": PRICE_FEATURES + TIME_FEATURES + WATCHLIST_FEATURES},
    "price_time_watchlist_xgb": {"kind": "xgb", "features": PRICE_FEATURES + TIME_FEATURES + WATCHLIST_FEATURES},
}
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
BOOTSTRAP_DRAWS = 5_000
RNG_SEED = 731
OUTCOME_HORIZONS = (15, 60)


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


def _source_key(row: dict[str, Any], *, context_kind: str) -> tuple[str, str, int, str, tuple[str, ...]]:
    """A candidate's immutable identity across outcome and context artifacts."""
    return (
        context_kind,
        str(row["symbol"]).upper(),
        int(row["direction_sign"]),
        _utc(row["decision_at_utc"]).isoformat(),
        tuple(sorted(map(str, row.get("source_message_ids") or []))),
    )


def _source_group(row: dict[str, Any]) -> tuple[str, ...]:
    ids = tuple(sorted(map(str, row.get("source_message_ids") or [])))
    if not ids:
        raise ValueError("candidate missing source_message_ids; cannot construct an auditable candidate group")
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


def _time_features(decision: pd.Timestamp) -> dict[str, float]:
    """Minutes since 09:30 ET as a circular intraday feature, fixed pre-run."""
    local = decision.tz_convert("America/New_York")
    minutes = (local.hour * 60 + local.minute) - (9 * 60 + 30)
    angle = 2.0 * np.pi * minutes / 390.0
    return {"session_time_sin": float(np.sin(angle)), "session_time_cos": float(np.cos(angle))}


def join_candidates(
    alerts: Iterable[dict[str, Any]], controls: Iterable[dict[str, Any]], context: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join precomputed candidates to context without filling missing fields."""
    context_index: dict[tuple[str, str, int, str, tuple[str, ...]], dict[str, Any]] = {}
    for row in context:
        kind = str(row.get("row_kind"))
        if kind not in {"alert", "control"}:
            raise ValueError(f"unrecognized context row_kind {kind!r}")
        key = _source_key(row, context_kind=kind)
        if key in context_index:
            raise ValueError(f"duplicate context candidate key: {key}")
        context_index[key] = row

    joined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str, tuple[str, ...]]] = set()
    for kind, source_rows in (("alert", alerts), ("control", controls)):
        for source in source_rows:
            key = _source_key(source, context_kind=kind)
            if key in seen:
                raise ValueError(f"duplicate source candidate key: {key}")
            seen.add(key)
            context_row = context_index.get(key)
            if context_row is None:
                raise ValueError(f"missing context for source candidate: {key}")
            candidate = dict(source)
            candidate["is_observed_alert"] = int(kind == "alert")
            candidate["candidate_group_message_ids"] = list(_source_group(source))
            candidate.update(_time_features(_utc(source["decision_at_utc"])))
            for column in WATCHLIST_FEATURES:
                value = context_row.get(column)
                if not isinstance(value, bool):
                    raise ValueError(f"{key}: expected Boolean {column}, got {value!r}")
                candidate[column] = int(value)
            joined.append(candidate)
    extra = set(context_index) - seen
    if extra:
        raise ValueError(f"context contains {len(extra)} unjoined candidates; input artifacts do not match")
    joined.sort(key=lambda row: (row["session_date_et"], row["decision_at_utc"], row["is_observed_alert"]))
    return joined


def validate_candidates(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no candidates")
    for row in rows:
        for column in PRICE_FEATURES + TIME_FEATURES + WATCHLIST_FEATURES:
            value = row.get(column)
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"missing or non-finite model input {column} for {row['decision_at_utc']}")
        if int(row["is_observed_alert"]) not in (0, 1):
            raise ValueError("is_observed_alert must be 0 or 1")
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row["candidate_group_message_ids"])].append(row)
    for group, members in grouped.items():
        positives = sum(int(row["is_observed_alert"]) for row in members)
        dates = {row["session_date_et"] for row in members}
        if positives != 1 or len(dates) != 1:
            raise ValueError(f"invalid candidate group {group}: positives={positives}, session_dates={sorted(dates)}")


def _matrix(rows: list[dict[str, Any]], features: tuple[str, ...]) -> np.ndarray:
    return np.array([[float(row[column]) for column in features] for row in rows], dtype=float)


def _labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(row["is_observed_alert"]) for row in rows], dtype=int)


def _build_model(spec: dict[str, Any]) -> Any:
    if spec["kind"] == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000, random_state=RNG_SEED)),
        ])
    if spec["kind"] == "hgb":
        # Fixed conservative capacity; this is an ablation, not a test-set-tuned model.
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=100, max_leaf_nodes=5, l2_regularization=10.0,
            random_state=RNG_SEED,
        )
    if spec["kind"] == "xgb":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cpu",
            n_estimators=80, max_depth=2, learning_rate=0.04, min_child_weight=8,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=20.0,
            n_jobs=1, random_state=RNG_SEED,
        )
    raise ValueError(f"unknown model kind {spec['kind']!r}")


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("both observed classes are required in train partition")
    return np.where(y == 1, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))


def observed_label_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "n": int(len(y)), "observed_alert_count": int(y.sum()), "observed_alert_rate": float(y.mean()) if len(y) else None,
        "average_precision": None, "roc_auc": None,
    }
    if len(y) and len(np.unique(y)) == 2:
        result["average_precision"] = float(average_precision_score(y, score))
        result["roc_auc"] = float(roc_auc_score(y, score))
    return result


def group_ranking_metrics(rows: list[dict[str, Any]], score: np.ndarray) -> dict[str, float | int | None]:
    groups: dict[tuple[str, ...], list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, value in zip(rows, score, strict=True):
        groups[tuple(row["candidate_group_message_ids"])].append((row, float(value)))
    ranks: list[float] = []
    for members in groups.values():
        alerts = [(row, value) for row, value in members if row["is_observed_alert"]]
        if len(alerts) != 1:
            raise ValueError("group partition lost or duplicated its observed alert")
        _, alert_score = alerts[0]
        controls = np.array([value for row, value in members if not row["is_observed_alert"]], dtype=float)
        if not len(controls):
            continue
        rank = 1.0 + float(np.sum(controls > alert_score)) + 0.5 * float(np.sum(controls == alert_score))
        ranks.append(rank)
    if not ranks:
        return {"eligible_groups": 0, "top_1_rate": None, "mean_reciprocal_rank": None, "mean_rank": None}
    values = np.array(ranks)
    return {
        "eligible_groups": int(len(values)), "top_1_rate": float(np.mean(values <= 1.0)),
        "mean_reciprocal_rank": float(np.mean(1.0 / values)), "mean_rank": float(np.mean(values)),
    }


def bootstrap_mean_ci(values: np.ndarray, *, seed: int = RNG_SEED) -> list[float] | None:
    if not len(values):
        return None
    generator = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for index in range(BOOTSTRAP_DRAWS):
        means[index] = generator.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025) * 100), float(np.quantile(means, 0.975) * 100)]


def mde_pct(values: np.ndarray, *, alpha: float = 0.05, power: float = 0.80) -> float | None:
    """Two-sided normal-approximation MDE for independent group-level deltas."""
    if len(values) < 2:
        return None
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return float(z * np.std(values, ddof=1) / np.sqrt(len(values)) * 100)


def held_out_underlying_outcomes(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | list[float] | None]]:
    """Source alert minus its matched candidates; outcomes are never model inputs."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row["candidate_group_message_ids"])].append(row)
    result: dict[str, dict[str, float | int | list[float] | None]] = {}
    for horizon in OUTCOME_HORIZONS:
        column = f"directional_forward_return_{horizon}m"
        deltas: list[float] = []
        alerts: list[float] = []
        controls: list[float] = []
        for members in groups.values():
            alert_rows = [row for row in members if row["is_observed_alert"]]
            control_rows = [row for row in members if not row["is_observed_alert"]]
            if len(alert_rows) != 1:
                raise ValueError("candidate group must retain one observed alert")
            alert_value = alert_rows[0].get(column)
            control_values = [row.get(column) for row in control_rows]
            if alert_value is None or any(value is None for value in control_values) or not control_values:
                continue
            alerts.append(float(alert_value))
            controls.append(float(np.mean(control_values)))
            deltas.append(float(alert_value) - float(np.mean(control_values)))
        values = np.array(deltas, dtype=float)
        result[f"{horizon}m"] = {
            "paired_groups": int(len(values)),
            "alert_mean_pct": None if not alerts else float(np.mean(alerts) * 100),
            "matched_control_group_mean_pct": None if not controls else float(np.mean(controls) * 100),
            "alert_minus_control_mean_pct": None if not len(values) else float(values.mean() * 100),
            "bootstrap_95pct_ci_pct": bootstrap_mean_ci(values),
            "minimum_detectable_effect_pct": mde_pct(values),
            "mde_assumptions": "two-sided alpha=0.05, 80% power, normal approximation over independent alert/control groups",
        }
    return result


def _fit_and_score(model: Any, spec: dict[str, Any], train_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], *, permute: bool) -> tuple[np.ndarray, dict[str, float]]:
    features = spec["features"]
    x_train, y_train = _matrix(train_rows, features), _labels(train_rows)
    if permute:
        y_train = np.random.default_rng(RNG_SEED).permutation(y_train)
    if spec["kind"] in {"hgb", "xgb"}:
        model.fit(x_train, y_train, sample_weight=_balanced_weights(y_train))
    else:
        model.fit(x_train, y_train)
    score = model.predict_proba(_matrix(target_rows, features))[:, 1]
    coefficients: dict[str, float] = {}
    if spec["kind"] == "logistic":
        coefficients = {feature: float(value) for feature, value in zip(features, model.named_steps["model"].coef_[0], strict=True)}
    return score, coefficients


def _fixed_hyperparameters(kind: str) -> dict[str, object]:
    if kind == "logistic":
        return {"C": 1.0, "class_weight": "balanced", "random_state": RNG_SEED}
    if kind == "hgb":
        return {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 5, "l2_regularization": 10.0, "class_balance": "inverse_frequency", "random_state": RNG_SEED}
    if kind == "xgb":
        return {"tree_method": "hist", "device": "cpu", "n_estimators": 80, "max_depth": 2, "learning_rate": 0.04, "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 20.0, "class_balance": "inverse_frequency", "n_jobs": 1, "random_state": RNG_SEED}
    raise ValueError(f"unknown model kind {kind!r}")

def run_experiment(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    validate_candidates(rows)
    split_dates = _date_splits(rows)
    partitions = {name: [row for row in rows if row["session_date_et"] in dates] for name, dates in split_dates.items()}
    for name, partition in partitions.items():
        labels = _labels(partition)
        if len(partition) == 0 or len(np.unique(labels)) != 2:
            raise ValueError(f"{name} partition does not contain both observed labels")
    report: dict[str, Any] = {
        "schema": "ace_pu_timing_experiment_v1", "research_only": True,
        "selection_target": "observed ACE alert versus matched unlabeled quiet-window candidate",
        "probability_warning": "Scores estimate observed-alert propensity in this sampled panel; they are not trade probability, win probability, or expected P&L.",
        "chronological_split_dates": {name: sorted(dates) for name, dates in split_dates.items()},
        "partition_counts": {name: {"candidates": len(partition), "observed_alerts": int(_labels(partition).sum()),
                                  "unlabeled_controls": int(len(partition) - _labels(partition).sum())}
                             for name, partition in partitions.items()},
        "models": {},
        "out_of_sample_underlying_evaluation": held_out_underlying_outcomes(partitions["test"]),
    }
    scored_test_rows: dict[str, list[dict[str, Any]]] = {}
    for name, spec in MODEL_SPECS.items():
        model = _build_model(spec)
        model_report: dict[str, Any] = {
            "model_kind": spec["kind"], "features": list(spec["features"]),
            "fixed_hyperparameters": _fixed_hyperparameters(spec["kind"]),
            "metrics_label_warning": "All metrics distinguish observed alerts from unlabeled candidates; controls are not verified negatives.",
        }
        for partition_name in ("train", "validation", "test"):
            score, coefficients = _fit_and_score(_build_model(spec), spec, partitions["train"], partitions[partition_name], permute=False)
            y = _labels(partitions[partition_name])
            model_report[partition_name] = {
                "observed_label_metrics": observed_label_metrics(y, score),
                "within_source_group_retrieval": group_ranking_metrics(partitions[partition_name], score),
            }
            if partition_name == "train" and coefficients:
                model_report["standardized_coefficients"] = coefficients
            if partition_name == "test":
                scored = []
                for row, value in zip(partitions["test"], score, strict=True):
                    item = dict(row)
                    item["observed_alert_propensity_score"] = float(value)
                    scored.append(item)
                scored_test_rows[name] = scored
        report["models"][name] = model_report

    # A falsification arm for the highest-capacity fixed model. It uses the
    # exact same train/test partitions and feature set; only train labels are
    # permuted.  It is diagnostic, not a calibrated significance test.
    primary = "price_time_watchlist_xgb"
    null_score, _ = _fit_and_score(_build_model(MODEL_SPECS[primary]), MODEL_SPECS[primary], partitions["train"], partitions["test"], permute=True)
    report["permuted_train_label_null"] = {
        "model": primary,
        "seed": RNG_SEED,
        "test_observed_label_metrics": observed_label_metrics(_labels(partitions["test"]), null_score),
        "test_within_source_group_retrieval": group_ranking_metrics(partitions["test"], null_score),
        "interpretation": "If the real model does not materially exceed this diagnostic null, the observed-label timing association is not established in this sample.",
    }
    return report, scored_test_rows


def _format(value: object, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def write_report(path: Path, report: dict[str, Any]) -> None:
    test_outcome = report["out_of_sample_underlying_evaluation"]
    lines = [
        "# ACE positive-unlabeled timing experiment", "",
        "Research-only. This measures whether a compact, fixed selector reproduces the timing of *observed* ACE alerts among frozen same-symbol/session quiet-window candidates. It is not a profitability backtest, option model, full-universe discovery test, or live-trading recommendation.", "",
        "## Design", "",
        "- Positives: source-cited ACE entry alerts. Unlabeled candidates: frozen same-symbol/session, same-direction no-nearby-alert windows. An unlabeled candidate is not a verified bad trade.",
        "- Inputs: completed-bar directional price/volume structure, circular time-of-session, and strictly pre-decision exported Discord watchlist flags. No outcomes, options data, claimed P&L, news, or universe membership are model inputs.",
        "- Split: chronological by ET session date, fixed before fitting. All variants are fixed ablations; none was selected on the test partition.",
        "- Scores are observed-alert propensities for this sampled panel—not probabilities of profitability or execution.", "",
        "## Held-out observed-label retrieval", "",
        "| Model | Test AP | Test ROC AUC | Group top-1 | Mean reciprocal rank |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, model in report["models"].items():
        observed = model["test"]["observed_label_metrics"]
        retrieval = model["test"]["within_source_group_retrieval"]
        lines.append(f"| {name} | {_format(observed['average_precision'])} | {_format(observed['roc_auc'])} | {_format(retrieval['top_1_rate'])} | {_format(retrieval['mean_reciprocal_rank'])} |")
    null_metrics = report["permuted_train_label_null"]["test_observed_label_metrics"]
    null_retrieval = report["permuted_train_label_null"]["test_within_source_group_retrieval"]
    lines.extend([
        f"| permuted-train-label null ({report['permuted_train_label_null']['model']}) | {_format(null_metrics['average_precision'])} | {_format(null_metrics['roc_auc'])} | {_format(null_retrieval['top_1_rate'])} | {_format(null_retrieval['mean_reciprocal_rank'])} |", "",
        "## Separate held-out underlying outcome description", "",
        "These are not trained targets and are not option P&L. Each value is the alert's directional underlying return less the mean of its frozen matched candidates. The MDE is the approximate absolute mean difference this sample could detect with 80% power; if it is large, a null is ‘not measurable here,’ not evidence of no edge.", "",
        "| Horizon | Paired groups | Alert mean (%) | Control-group mean (%) | Difference (%) | 95% bootstrap CI (%) | MDE (%) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for horizon, result in test_outcome.items():
        ci = result["bootstrap_95pct_ci_pct"]
        ci_text = "—" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(f"| {horizon} | {result['paired_groups']} | {_format(result['alert_mean_pct'])} | {_format(result['matched_control_group_mean_pct'])} | {_format(result['alert_minus_control_mean_pct'])} | {ci_text} | {_format(result['minimum_detectable_effect_pct'])} |")
    lines.extend([
        "", "## Interpretation boundary", "",
        "A model can be useful only if it exceeds the permuted-label diagnostic in a later untouched candidate tape and its selected candidates then have a practically meaningful underlying outcome after realistic delay/cost assumptions. This dataset injects ACE's symbol and direction, so it cannot answer universe discovery, direction selection, contract selection, or management. The next testable variant is a new immutable candidate tape that independently enumerates ACE's traded universe and both directions at each decision time; do not add reinforcement learning before that prerequisite exists.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alerts", type=Path, default=DEFAULT_ALERTS)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwritten.")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out}")
    for path in (args.alerts, args.controls, args.context):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = join_candidates(read_jsonl(args.alerts), read_jsonl(args.controls), read_jsonl(args.context))
    report, scored = run_experiment(rows)
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "candidates.jsonl", rows)
    for name, candidate_rows in scored.items():
        write_jsonl(args.out / f"test_scored_{name}.jsonl", candidate_rows)
    report["inputs"] = {str(path): _sha256(path) for path in (args.alerts, args.controls, args.context)}
    report["limitations"] = [
        "The candidate panel injects ACE's symbol and direction; it is not full-universe discovery or direction selection.",
        "No-nearby-alert controls are unlabeled, not verified negative examples or losing trades.",
        "Underlying IEX bars were fetched retrospectively; original availability, quotes, corporate-action handling, and option execution are not certified.",
        "Forward underlying returns are descriptive held-out outcomes, not model inputs or option P&L estimates.",
    ]
    (args.out / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    write_report(args.out / "report.md", report)
    print(json.dumps({"out": str(args.out), "partition_counts": report["partition_counts"],
                      "test_outcomes": report["out_of_sample_underlying_evaluation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
