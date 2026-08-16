from __future__ import annotations

import json
import math
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    ndcg_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class CompetitionConfig:
    task_name: str
    target_column: str
    feature_columns: list[str]
    train_frac: float
    val_frac: float
    seeds: list[int]
    families: list[str]
    output_dir: Path
    sample_weight_column: str | None = None
    neutral_weight_factor: float = 1.0
    positive_label: int | float | None = None
    rank_group: str = "timestamp"
    top_k: int = 10
    top_feature_n: int = 50
    xgb_config: dict[str, Any] | None = None
    lgbm_config: dict[str, Any] | None = None
    device: str = "cpu"
    timestamp_column: str = "timestamp"
    id_columns: tuple[str, ...] = ("timestamp", "ticker")


def parse_seeds(value: str | None, default_count: int = 5) -> list[int]:
    if value:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
        if seeds:
            return seeds
    return list(range(42, 42 + int(default_count)))


def parse_families(value: str | None) -> list[str]:
    if not value:
        return ["xgb_classifier", "xgb_ranker", "lgbm_classifier", "lgbm_ranker"]
    aliases = {
        "xgb": "xgb_classifier",
        "xgboost": "xgb_classifier",
        "xgb_classifier": "xgb_classifier",
        "xgb_ranker": "xgb_ranker",
        "lgbm": "lgbm_classifier",
        "lightgbm": "lgbm_classifier",
        "lgbm_classifier": "lgbm_classifier",
        "lgbm_ranker": "lgbm_ranker",
    }
    out = []
    for part in value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown model family: {part}")
        out.append(aliases[key])
    return list(dict.fromkeys(out))


def load_bundle(bundle: Path, work_dir: Path, manifest_name: str) -> dict[str, Any]:
    work_dir.mkdir(exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        try:
            tar.extractall(work_dir, filter="data")
        except TypeError:
            tar.extractall(work_dir)
    return json.loads((work_dir / manifest_name).read_text())


def time_split(frame: pd.DataFrame, train_frac: float, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return frame.iloc[:t1].copy(), frame.iloc[t1:t2].copy(), frame.iloc[t2:].copy()


def normalize_features(frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    missing = [col for col in feature_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    for col in feature_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return feature_columns


def resolve_positive_label(y: np.ndarray, positive_label: int | float | None) -> int | float:
    labels = sorted(pd.Series(y).dropna().unique().tolist())
    if not labels:
        raise ValueError("Target has no labels.")
    if positive_label is not None:
        return positive_label
    return labels[-1]


def class_count(y: np.ndarray) -> int:
    return int(pd.Series(y).dropna().nunique())


def sample_weights(frame: pd.DataFrame, cfg: CompetitionConfig) -> np.ndarray | None:
    if cfg.sample_weight_column and cfg.sample_weight_column in frame.columns:
        w = pd.to_numeric(frame[cfg.sample_weight_column], errors="coerce").fillna(1.0).to_numpy(float)
    else:
        w = np.ones(len(frame), dtype=float)
    y = frame[cfg.target_column].to_numpy()
    if cfg.neutral_weight_factor != 1.0 and pd.Series(y).nunique() >= 3:
        labels = sorted(pd.Series(y).dropna().unique().tolist())
        neutral = labels[len(labels) // 2]
        w = w * np.where(y == neutral, cfg.neutral_weight_factor, 1.0)
    return w.astype(np.float32)


def make_group_key(frame: pd.DataFrame, cfg: CompetitionConfig) -> pd.Series:
    if cfg.rank_group == "date":
        return pd.to_datetime(frame[cfg.timestamp_column], utc=True).dt.tz_convert(None).dt.normalize().astype(str)
    if cfg.rank_group == "ticker_date" and "ticker" in frame.columns:
        day = pd.to_datetime(frame[cfg.timestamp_column], utc=True).dt.tz_convert(None).dt.normalize().astype(str)
        return frame["ticker"].astype(str) + "|" + day
    if cfg.rank_group == "timestamp" and cfg.timestamp_column in frame.columns:
        return pd.to_datetime(frame[cfg.timestamp_column], utc=True).astype(str)
    return pd.Series("all", index=frame.index)


def sorted_for_ranker(frame: pd.DataFrame, cfg: CompetitionConfig) -> tuple[pd.DataFrame, list[int]]:
    out = frame.copy()
    out["_rank_group_key"] = make_group_key(out, cfg)
    out = out.sort_values(["_rank_group_key", cfg.timestamp_column] if cfg.timestamp_column in out.columns else ["_rank_group_key"])
    sizes = out.groupby("_rank_group_key", sort=False).size().astype(int).tolist()
    return out.drop(columns=["_rank_group_key"]), sizes


def relevance(frame: pd.DataFrame, cfg: CompetitionConfig) -> np.ndarray:
    y = frame[cfg.target_column].to_numpy()
    positive = resolve_positive_label(y, cfg.positive_label)
    return (y == positive).astype(np.float32)


def xgb_classifier_params(seed: int, n_classes: int, cfg: CompetitionConfig) -> dict[str, Any]:
    params = dict(cfg.xgb_config or {})
    params.pop("early_stopping_rounds", None)
    params.update(
        {
            "random_state": seed,
            "tree_method": "hist",
            "device": cfg.device,
            "n_jobs": -1,
            "verbosity": 1,
        }
    )
    params.setdefault("n_estimators", 800)
    params.setdefault("learning_rate", 0.04)
    params.setdefault("max_depth", 5)
    params.setdefault("subsample", 0.85)
    params.setdefault("colsample_bytree", 0.85)
    params.setdefault("eval_metric", "mlogloss" if n_classes > 2 else "logloss")
    if n_classes > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
    else:
        params["objective"] = "binary:logistic"
    return params


def xgb_ranker_params(seed: int, cfg: CompetitionConfig) -> dict[str, Any]:
    params = dict(cfg.xgb_config or {})
    params.pop("early_stopping_rounds", None)
    params.pop("objective", None)
    params.pop("num_class", None)
    params.update(
        {
            "random_state": seed,
            "tree_method": "hist",
            "device": cfg.device,
            "n_jobs": -1,
            "verbosity": 1,
            "objective": "rank:ndcg",
            "eval_metric": f"ndcg@{cfg.top_k}",
        }
    )
    params.setdefault("n_estimators", 800)
    params.setdefault("learning_rate", 0.04)
    params.setdefault("max_depth", 5)
    params.setdefault("subsample", 0.85)
    params.setdefault("colsample_bytree", 0.85)
    return params


def lgbm_classifier_params(seed: int, n_classes: int, cfg: CompetitionConfig) -> dict[str, Any]:
    params = dict(cfg.lgbm_config or {})
    params.update({"random_state": seed, "n_jobs": -1, "verbose": -1})
    params.setdefault("n_estimators", 900)
    params.setdefault("learning_rate", 0.04)
    params.setdefault("num_leaves", 63)
    params.setdefault("subsample", 0.85)
    params.setdefault("colsample_bytree", 0.85)
    params.setdefault("objective", "multiclass" if n_classes > 2 else "binary")
    return params


def lgbm_ranker_params(seed: int, cfg: CompetitionConfig) -> dict[str, Any]:
    params = dict(cfg.lgbm_config or {})
    params.update({"random_state": seed, "n_jobs": -1, "verbose": -1, "objective": "lambdarank", "metric": "ndcg"})
    params.setdefault("n_estimators", 900)
    params.setdefault("learning_rate", 0.04)
    params.setdefault("num_leaves", 63)
    params.setdefault("subsample", 0.85)
    params.setdefault("colsample_bytree", 0.85)
    return params


def classifier_score(model: Any, x_data: np.ndarray, positive_index: int) -> np.ndarray:
    proba = model.predict_proba(x_data)
    if proba.ndim == 1:
        return proba
    return proba[:, positive_index]


def classification_metrics(
    model: Any,
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    labels: list[Any],
    positive_label: Any,
    split: str,
) -> dict[str, float]:
    x_data = frame[features].to_numpy(np.float32)
    y_true = frame[target].to_numpy()
    pred = model.predict(x_data)
    out: dict[str, float] = {
        f"{split}_accuracy": float(accuracy_score(y_true, pred)),
        f"{split}_balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        f"{split}_f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        f"{split}_positive_precision": float(precision_score(y_true, pred, labels=labels, pos_label=positive_label, average="binary" if len(labels) == 2 else None, zero_division=0) if len(labels) == 2 else precision_score(y_true == positive_label, pred == positive_label, zero_division=0)),
        f"{split}_positive_recall": float(recall_score(y_true == positive_label, pred == positive_label, zero_division=0)),
        f"{split}_positive_f1": float(f1_score(y_true == positive_label, pred == positive_label, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_data)
        try:
            out[f"{split}_log_loss"] = float(log_loss(y_true, proba, labels=labels))
        except ValueError:
            out[f"{split}_log_loss"] = float("nan")
    return out


def rank_metrics(scores: np.ndarray, frame: pd.DataFrame, cfg: CompetitionConfig, split: str) -> dict[str, float]:
    work = pd.DataFrame(
        {
            "relevance": relevance(frame, cfg),
            "score": np.asarray(scores, dtype=float),
            "group": make_group_key(frame, cfg).to_numpy(),
        }
    )
    ndcgs = []
    precision = []
    mean_rel = []
    for _, group_df in work.groupby("group", sort=False):
        if len(group_df) < 2:
            continue
        y_group = group_df["relevance"].to_numpy(float)
        s_group = group_df["score"].to_numpy(float)
        k = min(cfg.top_k, len(group_df))
        if y_group.sum() > 0:
            try:
                ndcgs.append(float(ndcg_score([y_group], [s_group], k=k)))
            except ValueError:
                pass
        top = np.argsort(-s_group)[:k]
        precision.append(float(y_group[top].mean()))
        mean_rel.append(float(y_group[top].sum()))
    return {
        f"{split}_ndcg_at_{cfg.top_k}": float(np.nanmean(ndcgs)) if ndcgs else float("nan"),
        f"{split}_precision_at_{cfg.top_k}": float(np.nanmean(precision)) if precision else float("nan"),
        f"{split}_positives_at_{cfg.top_k}": float(np.nanmean(mean_rel)) if mean_rel else float("nan"),
    }


def feature_importance(model: Any, features: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "booster_"):
        values = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    else:
        values = np.zeros(len(features), dtype=float)
    return (
        pd.DataFrame({"feature": features, "importance": values})
        .sort_values("importance", ascending=False)
        .assign(rank=lambda d: np.arange(1, len(d) + 1))
    )


def top_pick_ids(frame: pd.DataFrame, scores: np.ndarray, cfg: CompetitionConfig) -> set[str]:
    work = frame[[c for c in cfg.id_columns if c in frame.columns]].copy()
    work["_score"] = np.asarray(scores, dtype=float)
    work["_group"] = make_group_key(frame, cfg).to_numpy()
    ids: set[str] = set()
    for group_key, group in work.groupby("_group", sort=False):
        top = group.nlargest(min(cfg.top_k, len(group)), "_score")
        for _, row in top.iterrows():
            parts = [str(group_key)]
            for col in cfg.id_columns:
                if col in row.index:
                    parts.append(str(row[col]))
            ids.add("|".join(parts))
    return ids


def mean_pairwise_jaccard(sets: list[set[str]]) -> float:
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            vals.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return float(np.mean(vals)) if vals else float("nan")


def train_one_family(
    family: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: CompetitionConfig,
) -> tuple[Any, dict[str, Any], pd.DataFrame, set[str]]:
    labels = sorted(train_df[cfg.target_column].dropna().unique().tolist())
    positive_label = resolve_positive_label(train_df[cfg.target_column].to_numpy(), cfg.positive_label)
    positive_index = labels.index(positive_label) if positive_label in labels else len(labels) - 1
    n_classes = len(labels)
    early_rounds = int((cfg.xgb_config or {}).get("early_stopping_rounds", 60))

    if family == "xgb_classifier":
        params = xgb_classifier_params(seed, n_classes, cfg)
        model = xgb.XGBClassifier(**params, early_stopping_rounds=early_rounds)
        model.fit(
            train_df[cfg.feature_columns].to_numpy(np.float32),
            train_df[cfg.target_column].to_numpy(),
            sample_weight=sample_weights(train_df, cfg),
            eval_set=[(val_df[cfg.feature_columns].to_numpy(np.float32), val_df[cfg.target_column].to_numpy())],
            verbose=50,
        )
        test_scores = classifier_score(model, test_df[cfg.feature_columns].to_numpy(np.float32), positive_index)
    elif family == "xgb_ranker":
        rank_train, train_group = sorted_for_ranker(train_df, cfg)
        rank_val, val_group = sorted_for_ranker(val_df, cfg)
        model = xgb.XGBRanker(**xgb_ranker_params(seed, cfg), early_stopping_rounds=early_rounds)
        model.fit(
            rank_train[cfg.feature_columns].to_numpy(np.float32),
            relevance(rank_train, cfg),
            group=train_group,
            eval_set=[(rank_val[cfg.feature_columns].to_numpy(np.float32), relevance(rank_val, cfg))],
            eval_group=[val_group],
            verbose=50,
        )
        test_scores = model.predict(test_df[cfg.feature_columns].to_numpy(np.float32))
    elif family in {"lgbm_classifier", "lgbm_ranker"}:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("Install LightGBM in Colab first: !pip install -q lightgbm") from exc
        if family == "lgbm_classifier":
            model = lgb.LGBMClassifier(**lgbm_classifier_params(seed, n_classes, cfg))
            model.fit(
                train_df[cfg.feature_columns],
                train_df[cfg.target_column],
                sample_weight=sample_weights(train_df, cfg),
                eval_set=[(val_df[cfg.feature_columns], val_df[cfg.target_column])],
                callbacks=[lgb.early_stopping(early_rounds, verbose=True)],
            )
            test_scores = classifier_score(model, test_df[cfg.feature_columns].to_numpy(np.float32), positive_index)
        else:
            rank_train, train_group = sorted_for_ranker(train_df, cfg)
            rank_val, val_group = sorted_for_ranker(val_df, cfg)
            model = lgb.LGBMRanker(**lgbm_ranker_params(seed, cfg))
            model.fit(
                rank_train[cfg.feature_columns],
                relevance(rank_train, cfg),
                group=train_group,
                eval_set=[(rank_val[cfg.feature_columns], relevance(rank_val, cfg))],
                eval_group=[val_group],
                callbacks=[lgb.early_stopping(early_rounds, verbose=True)],
            )
            test_scores = model.predict(test_df[cfg.feature_columns])
    else:
        raise ValueError(f"Unsupported family: {family}")

    row: dict[str, Any] = {
        "family": family,
        "seed": seed,
        "positive_label": positive_label,
        "n_features": len(cfg.feature_columns),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "best_iteration": int(getattr(model, "best_iteration", getattr(model, "best_iteration_", -1)) or -1),
    }
    if "classifier" in family:
        row.update(classification_metrics(model, val_df, cfg.feature_columns, cfg.target_column, labels, positive_label, "val"))
        row.update(classification_metrics(model, test_df, cfg.feature_columns, cfg.target_column, labels, positive_label, "test"))
    row.update(rank_metrics(test_scores, test_df, cfg, "test"))
    picks = top_pick_ids(test_df, test_scores, cfg)
    return model, row, feature_importance(model, cfg.feature_columns), picks


def summarize_runs(rows: list[dict[str, Any]]) -> pd.DataFrame:
    results = pd.DataFrame(rows)
    metric_cols = [c for c in results.columns if c.startswith(("val_", "test_"))]
    out = []
    for family, group in results.groupby("family", sort=True):
        row: dict[str, Any] = {"family": family, "runs": int(len(group))}
        for col in metric_cols:
            row[f"{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())
            row[f"{col}_std"] = float(pd.to_numeric(group[col], errors="coerce").std())
        out.append(row)
    return pd.DataFrame(out)


def stability_frame(importances: dict[tuple[str, int], pd.DataFrame], cfg: CompetitionConfig) -> pd.DataFrame:
    rows = []
    for family in cfg.families:
        fam_items = [(seed, df) for (fam, seed), df in importances.items() if fam == family]
        top_sets = [set(df.head(cfg.top_feature_n)["feature"]) for _, df in fam_items]
        family_jaccard = mean_pairwise_jaccard(top_sets)
        counts: dict[str, list[float]] = {}
        for _, df in fam_items:
            for _, row in df.head(cfg.top_feature_n).iterrows():
                counts.setdefault(str(row["feature"]), []).append(float(row["rank"]))
        for feature, ranks in counts.items():
            rows.append(
                {
                    "family": family,
                    "feature": feature,
                    "top_feature_frequency": len(ranks) / max(len(fam_items), 1),
                    "avg_top_rank": float(np.mean(ranks)),
                    "pairwise_top_feature_jaccard": family_jaccard,
                }
            )
    return pd.DataFrame(rows).sort_values(["family", "top_feature_frequency", "avg_top_rank"], ascending=[True, False, True])


def pick_overlap_frame(picks: dict[tuple[str, int], set[str]], cfg: CompetitionConfig) -> pd.DataFrame:
    rows = []
    for family in cfg.families:
        fam_sets = [v for (fam, _seed), v in picks.items() if fam == family]
        rows.append(
            {
                "family": family,
                "top_pick_pairwise_jaccard": mean_pairwise_jaccard(fam_sets),
                "top_pick_sets": len(fam_sets),
                "avg_top_picks_per_seed": float(np.mean([len(s) for s in fam_sets])) if fam_sets else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def choose_best(results: pd.DataFrame) -> pd.Series:
    ranker_metric = "test_ndcg_at_10" if "test_ndcg_at_10" in results.columns else None
    if "val_log_loss" in results.columns and results["val_log_loss"].notna().any():
        sortable = results.sort_values(["val_log_loss", "test_positive_f1"], ascending=[True, False], na_position="last")
        return sortable.iloc[0]
    if ranker_metric:
        return results.sort_values(ranker_metric, ascending=False, na_position="last").iloc[0]
    metric = [c for c in results.columns if c.startswith("test_ndcg_at_")]
    if metric:
        return results.sort_values(metric[0], ascending=False, na_position="last").iloc[0]
    return results.iloc[0]


def run_competition(frame: pd.DataFrame, cfg: CompetitionConfig) -> dict[str, Any]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(cfg.timestamp_column).reset_index(drop=True)
    frame = frame.dropna(subset=[cfg.target_column]).copy()
    normalize_features(frame, cfg.feature_columns)
    train_df, val_df, test_df = time_split(frame, cfg.train_frac, cfg.val_frac)

    rows: list[dict[str, Any]] = []
    importances: dict[tuple[str, int], pd.DataFrame] = {}
    picks: dict[tuple[str, int], set[str]] = {}
    models: dict[tuple[str, int], Any] = {}
    errors: list[dict[str, str]] = []

    for family in cfg.families:
        for seed in cfg.seeds:
            print(f"training {family} seed={seed}")
            try:
                model, row, fi, pick_ids = train_one_family(family, seed, train_df, val_df, test_df, cfg)
            except Exception as exc:
                errors.append({"family": family, "seed": str(seed), "error": repr(exc)})
                print(f"ERROR {family} seed={seed}: {exc!r}")
                continue
            rows.append(row)
            importances[(family, seed)] = fi
            picks[(family, seed)] = pick_ids
            models[(family, seed)] = model
            fi.to_csv(cfg.output_dir / f"feature_importance_{family}_seed{seed}.csv", index=False)
            joblib.dump(model, cfg.output_dir / f"model_{family}_seed{seed}.joblib")
            if hasattr(model, "save_model"):
                model.save_model(cfg.output_dir / f"model_{family}_seed{seed}.native")

    if not rows:
        raise RuntimeError(f"No model runs completed. Errors: {errors}")

    results = pd.DataFrame(rows)
    summary = summarize_runs(rows)
    stability = stability_frame(importances, cfg)
    pick_overlap = pick_overlap_frame(picks, cfg)
    best = choose_best(results)
    best_key = (str(best["family"]), int(best["seed"]))
    best_model = models[best_key]

    results.to_csv(cfg.output_dir / "seed_results.csv", index=False)
    summary.to_csv(cfg.output_dir / "model_family_summary.csv", index=False)
    stability.to_csv(cfg.output_dir / "feature_stability.csv", index=False)
    pick_overlap.to_csv(cfg.output_dir / "top_pick_overlap.csv", index=False)
    if errors:
        (cfg.output_dir / "run_errors.json").write_text(json.dumps(errors, indent=2))

    joblib.dump(best_model, cfg.output_dir / "best_model.joblib")
    if hasattr(best_model, "save_model"):
        best_model.save_model(cfg.output_dir / "best_model_native.txt")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_name": cfg.task_name,
        "target_column": cfg.target_column,
        "families": cfg.families,
        "seeds": cfg.seeds,
        "positive_label": resolve_positive_label(frame[cfg.target_column].to_numpy(), cfg.positive_label),
        "rank_group": cfg.rank_group,
        "top_k": cfg.top_k,
        "best": best.to_dict(),
        "errors": errors,
    }
    (cfg.output_dir / "competition_meta.json").write_text(json.dumps(metadata, indent=2, default=str))
    return {
        "results": results,
        "summary": summary,
        "stability": stability,
        "pick_overlap": pick_overlap,
        "best": best,
        "best_model": best_model,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def write_artifact_bundle(output_dir: Path, bundle_path: Path) -> None:
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                tar.add(path, arcname=path.name)
