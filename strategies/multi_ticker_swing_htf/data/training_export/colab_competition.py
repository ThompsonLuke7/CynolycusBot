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


# Heuristics for the per-family "one lucky seed" noise flag (see summarize_runs).
NOISE_CV_THRESHOLD = 0.5          # std / |mean| of the primary metric across seeds
NOISE_BEST_GAP_THRESHOLD = 0.5    # (best - median) / (|median| + eps) across seeds


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
    # Continuous target for regression families + Spearman (defaults to target_column).
    regression_target_column: str | None = None
    # Binary 0/1 relevance for ranking + NDCG/precision@k across ALL families.
    # Falls back to (target_column == positive_label) when unset.
    relevance_column: str | None = None
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

    def reg_target(self) -> str:
        return self.regression_target_column or self.target_column


def parse_seeds(value: str | None, default_count: int = 5) -> list[int]:
    if value:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
        if seeds:
            return seeds
    return list(range(42, 42 + int(default_count)))


# Full family set incl. regression — the regression-target trainers (momentum / HTF /
# meta) request this explicitly; the default stays classifier+ranker so existing callers
# (e.g. multi_ticker_swing's 3-class trainer) are unaffected.
ALL_FAMILIES = [
    "xgb_regressor", "xgb_classifier", "xgb_ranker",
    "lgbm_regressor", "lgbm_classifier", "lgbm_ranker",
]


def parse_families(value: str | None) -> list[str]:
    if not value:
        return ["xgb_classifier", "xgb_ranker", "lgbm_classifier", "lgbm_ranker"]
    aliases = {
        "xgb": "xgb_classifier",
        "xgboost": "xgb_classifier",
        "xgb_classifier": "xgb_classifier",
        "xgb_ranker": "xgb_ranker",
        "xgb_regressor": "xgb_regressor",
        "xgb_reg": "xgb_regressor",
        "lgbm": "lgbm_classifier",
        "lightgbm": "lgbm_classifier",
        "lgbm_classifier": "lgbm_classifier",
        "lgbm_ranker": "lgbm_ranker",
        "lgbm_regressor": "lgbm_regressor",
        "lgbm_reg": "lgbm_regressor",
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
    """Binary 0/1 relevance for ranking + NDCG/precision@k, shared by all families."""
    if cfg.relevance_column and cfg.relevance_column in frame.columns:
        rel = pd.to_numeric(frame[cfg.relevance_column], errors="coerce").fillna(0.0)
        return (rel > 0).astype(np.float32).to_numpy()
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


def xgb_regressor_params(seed: int, cfg: CompetitionConfig) -> dict[str, Any]:
    params = dict(cfg.xgb_config or {})
    params.pop("early_stopping_rounds", None)
    params.pop("num_class", None)
    params.update(
        {
            "random_state": seed,
            "tree_method": "hist",
            "device": cfg.device,
            "n_jobs": -1,
            "verbosity": 1,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }
    )
    params.setdefault("n_estimators", 800)
    params.setdefault("learning_rate", 0.04)
    params.setdefault("max_depth", 5)
    params.setdefault("subsample", 0.85)
    params.setdefault("colsample_bytree", 0.85)
    return params


def lgbm_regressor_params(seed: int, cfg: CompetitionConfig) -> dict[str, Any]:
    params = dict(cfg.lgbm_config or {})
    params.update({"random_state": seed, "n_jobs": -1, "verbose": -1, "objective": "regression", "metric": "rmse"})
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


def spearman_metric(scores: np.ndarray, frame: pd.DataFrame, cfg: CompetitionConfig, split: str) -> dict[str, float]:
    """Rank correlation of the model score against the continuous quality target.

    Universal across families (regressor/classifier/ranker all emit a per-row score),
    so it lets families be compared on the same axis as the regression objective.
    """
    y = pd.to_numeric(frame[cfg.reg_target()], errors="coerce")
    s = pd.Series(np.asarray(scores, dtype=float), index=frame.index)
    try:
        rho = float(s.corr(y, method="spearman"))
    except Exception:
        rho = float("nan")
    return {f"{split}_spearman": rho}


def feature_importance(model: Any, features: list[str]) -> pd.DataFrame:
    # LightGBM's sklearn wrapper defaults feature_importances_ to importance_type="split"
    # (raw count of times a feature was used), which inflates high-cardinality features
    # (week_of_year got 8.9% of split share vs 3.9% of gain). Ask the booster for gain
    # explicitly so every family reports the SAME quantity. XGBoost's
    # feature_importances_ is already gain-based, so it keeps that path.
    if hasattr(model, "booster_"):
        values = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
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


def _train_labels(train_df: pd.DataFrame, cfg: CompetitionConfig) -> tuple[list[Any], Any, int, int]:
    labels = sorted(train_df[cfg.target_column].dropna().unique().tolist())
    positive_label = resolve_positive_label(train_df[cfg.target_column].to_numpy(), cfg.positive_label)
    positive_index = labels.index(positive_label) if positive_label in labels else len(labels) - 1
    return labels, positive_label, positive_index, len(labels)


def fit_family(
    family: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: CompetitionConfig,
) -> Any:
    """Fit a single model of `family` on (train, val). Shared by the competition and OOF."""
    feats = cfg.feature_columns
    early_rounds = int((cfg.xgb_config or {}).get("early_stopping_rounds", 60))
    _, _, _, n_classes = _train_labels(train_df, cfg)

    if family == "xgb_classifier":
        model = xgb.XGBClassifier(**xgb_classifier_params(seed, n_classes, cfg), early_stopping_rounds=early_rounds)
        model.fit(
            train_df[feats].to_numpy(np.float32), train_df[cfg.target_column].to_numpy(),
            sample_weight=sample_weights(train_df, cfg),
            eval_set=[(val_df[feats].to_numpy(np.float32), val_df[cfg.target_column].to_numpy())], verbose=50,
        )
        return model
    if family == "xgb_regressor":
        model = xgb.XGBRegressor(**xgb_regressor_params(seed, cfg), early_stopping_rounds=early_rounds)
        model.fit(
            train_df[feats].to_numpy(np.float32), pd.to_numeric(train_df[cfg.reg_target()], errors="coerce").to_numpy(float),
            eval_set=[(val_df[feats].to_numpy(np.float32), pd.to_numeric(val_df[cfg.reg_target()], errors="coerce").to_numpy(float))],
            verbose=50,
        )
        return model
    if family == "xgb_ranker":
        rank_train, train_group = sorted_for_ranker(train_df, cfg)
        rank_val, val_group = sorted_for_ranker(val_df, cfg)
        model = xgb.XGBRanker(**xgb_ranker_params(seed, cfg), early_stopping_rounds=early_rounds)
        model.fit(
            rank_train[feats].to_numpy(np.float32), relevance(rank_train, cfg), group=train_group,
            eval_set=[(rank_val[feats].to_numpy(np.float32), relevance(rank_val, cfg))], eval_group=[val_group], verbose=50,
        )
        return model
    if family in {"lgbm_classifier", "lgbm_ranker", "lgbm_regressor"}:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("Install LightGBM in Colab first: !pip install -q lightgbm") from exc
        if family == "lgbm_classifier":
            model = lgb.LGBMClassifier(**lgbm_classifier_params(seed, n_classes, cfg))
            model.fit(
                train_df[feats], train_df[cfg.target_column], sample_weight=sample_weights(train_df, cfg),
                eval_set=[(val_df[feats], val_df[cfg.target_column])],
                callbacks=[lgb.early_stopping(early_rounds, verbose=True)],
            )
            return model
        if family == "lgbm_regressor":
            model = lgb.LGBMRegressor(**lgbm_regressor_params(seed, cfg))
            model.fit(
                train_df[feats], pd.to_numeric(train_df[cfg.reg_target()], errors="coerce"),
                eval_set=[(val_df[feats], pd.to_numeric(val_df[cfg.reg_target()], errors="coerce"))],
                callbacks=[lgb.early_stopping(early_rounds, verbose=True)],
            )
            return model
        rank_train, train_group = sorted_for_ranker(train_df, cfg)
        rank_val, val_group = sorted_for_ranker(val_df, cfg)
        model = lgb.LGBMRanker(**lgbm_ranker_params(seed, cfg))
        model.fit(
            rank_train[feats], relevance(rank_train, cfg), group=train_group,
            eval_set=[(rank_val[feats], relevance(rank_val, cfg))], eval_group=[val_group],
            callbacks=[lgb.early_stopping(early_rounds, verbose=True)],
        )
        return model
    raise ValueError(f"Unsupported family: {family}")


def score_family(model: Any, family: str, frame: pd.DataFrame, cfg: CompetitionConfig, positive_index: int) -> np.ndarray:
    """Per-row score for any family: P(positive) for classifiers, raw prediction otherwise."""
    x_data = frame[cfg.feature_columns].to_numpy(np.float32)
    if "classifier" in family:
        return classifier_score(model, x_data, positive_index)
    if family.startswith("lgbm"):
        return np.asarray(model.predict(frame[cfg.feature_columns]), dtype=float)
    return np.asarray(model.predict(x_data), dtype=float)


def train_one_family(
    family: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: CompetitionConfig,
) -> tuple[Any, dict[str, Any], pd.DataFrame, set[str]]:
    labels, positive_label, positive_index, _ = _train_labels(train_df, cfg)
    model = fit_family(family, seed, train_df, val_df, cfg)
    test_scores = score_family(model, family, test_df, cfg, positive_index)

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
    row.update(spearman_metric(test_scores, test_df, cfg, "test"))
    picks = top_pick_ids(test_df, test_scores, cfg)
    return model, row, feature_importance(model, cfg.feature_columns), picks


def primary_metric_name(results: pd.DataFrame) -> str | None:
    """The universal model-selection metric, preferred order: NDCG@k > precision@k > Spearman."""
    for prefix in ("test_ndcg_at_", "test_precision_at_"):
        cols = [c for c in results.columns if c.startswith(prefix)]
        if cols and pd.to_numeric(results[cols[0]], errors="coerce").notna().any():
            return cols[0]
    if "test_spearman" in results.columns and pd.to_numeric(results["test_spearman"], errors="coerce").notna().any():
        return "test_spearman"
    return None


def summarize_runs(rows: list[dict[str, Any]]) -> pd.DataFrame:
    results = pd.DataFrame(rows)
    metric_cols = [c for c in results.columns if c.startswith(("val_", "test_"))]
    primary = primary_metric_name(results)
    out = []
    for family, group in results.groupby("family", sort=True):
        row: dict[str, Any] = {"family": family, "runs": int(len(group))}
        for col in metric_cols:
            row[f"{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())
            row[f"{col}_std"] = float(pd.to_numeric(group[col], errors="coerce").std())
        # Noise flag: did one lucky seed carry the family? Compare best vs median and CV.
        if primary:
            vals = pd.to_numeric(group[primary], errors="coerce").dropna()
            if len(vals) >= 2:
                med = float(vals.median())
                mean = float(vals.mean())
                std = float(vals.std())
                cv = std / (abs(mean) + 1e-9)
                best_gap = (float(vals.max()) - med) / (abs(med) + 1e-9)
                row["primary_metric"] = primary
                row["primary_best"] = float(vals.max())
                row["primary_median"] = med
                row["primary_cv"] = cv
                row["noise_flag"] = bool(cv > NOISE_CV_THRESHOLD or best_gap > NOISE_BEST_GAP_THRESHOLD)
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
    """Best single run by the universal metric: NDCG@k, tiebreak precision@k then Spearman."""
    ndcg = [c for c in results.columns if c.startswith("test_ndcg_at_")]
    prec = [c for c in results.columns if c.startswith("test_precision_at_")]
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if ndcg:
        sort_cols.append(ndcg[0]); ascending.append(False)
    if prec:
        sort_cols.append(prec[0]); ascending.append(False)
    if "test_spearman" in results.columns:
        sort_cols.append("test_spearman"); ascending.append(False)
    if not sort_cols:
        return results.iloc[0]
    return results.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0]


def run_competition(frame: pd.DataFrame, cfg: CompetitionConfig) -> dict[str, Any]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(cfg.timestamp_column).reset_index(drop=True)
    # Require a usable label: the classifier/relevance target, and the regression
    # target when any regression family is in play.
    required = [cfg.target_column]
    if any("regressor" in fam for fam in cfg.families):
        required.append(cfg.reg_target())
    required = list(dict.fromkeys(c for c in required if c in frame.columns))
    frame = frame.dropna(subset=required).copy()
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
    # Pick the best FAMILY by its seed-averaged primary metric (robust to one lucky
    # seed), then the best SEED within that family.
    primary = primary_metric_name(results)
    if primary:
        fam_mean = results.groupby("family")[primary].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        best_family = str(fam_mean.idxmax())
        best = choose_best(results[results["family"] == best_family])
    else:
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


def date_folds(
    ts: pd.Series,
    *,
    train_months: int,
    embargo_days: int,
    test_months: int,
) -> list[dict[str, pd.Timestamp]]:
    """Non-overlapping walk-forward folds (train window / embargo gap / test window).

    Same shape the base momentum/HTF trainers used; built off the data's own date range.
    """
    ts = pd.to_datetime(ts, utc=True)
    date_min, date_max = ts.min(), ts.max()
    folds: list[dict[str, pd.Timestamp]] = []
    test_end = date_max
    while True:
        test_start = test_end - pd.DateOffset(months=test_months)
        train_end = test_start - pd.Timedelta(days=embargo_days)
        train_start = train_end - pd.DateOffset(months=train_months)
        if train_start < date_min:
            break
        folds.append(dict(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end))
        test_end = test_start
    return list(reversed(folds))


def walk_forward_oof(
    frame: pd.DataFrame,
    cfg: CompetitionConfig,
    family: str,
    seed: int,
    *,
    train_months: int = 18,
    embargo_days: int = 21,
    test_months: int = 4,
    min_train_rows: int = 20000,
    min_test_rows: int = 1000,
    val_frac: float = 0.2,
    diagnostic_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Refit `family` per walk-forward fold and emit out-of-fold per-row scores.

    The competition picks the winning family/seed on a single split; this regenerates
    full-history OOF predictions with that winner so downstream backtests / the meta
    spine keep their contract. Returns a frame indexed by id_columns with `score`, `y`
    (continuous quality), and any `diagnostic_columns` present.
    """
    diagnostic_columns = diagnostic_columns or []
    work = frame.sort_values(cfg.timestamp_column).reset_index(drop=True).copy()
    required = list(dict.fromkeys(c for c in (cfg.target_column, cfg.reg_target()) if c in work.columns))
    work = work.dropna(subset=required)
    normalize_features(work, cfg.feature_columns)
    ts = pd.to_datetime(work[cfg.timestamp_column], utc=True)
    folds = date_folds(ts, train_months=train_months, embargo_days=embargo_days, test_months=test_months)
    print(f"walk_forward_oof: {family} seed={seed} folds={len(folds)}")

    blocks: list[pd.DataFrame] = []
    for fi, f in enumerate(folds):
        tr = work[(ts >= f["train_start"]) & (ts <= f["train_end"])]
        te = work[(ts >= f["test_start"]) & (ts <= f["test_end"])]
        if len(tr) < min_train_rows or len(te) < min_test_rows:
            continue
        days = np.sort(pd.to_datetime(tr[cfg.timestamp_column], utc=True).dt.normalize().unique())
        split_at = min(max(1, int(len(days) * (1.0 - val_frac))), len(days) - 1)
        cut = pd.Timestamp(days[split_at])
        tr_ts = pd.to_datetime(tr[cfg.timestamp_column], utc=True)
        inner_tr, inner_val = tr[tr_ts < cut], tr[tr_ts >= cut]
        if len(inner_tr) < min_train_rows // 2 or len(inner_val) < 200:
            continue
        _, _, positive_index, _ = _train_labels(inner_tr, cfg)
        model = fit_family(family, seed, inner_tr, inner_val, cfg)
        scores = score_family(model, family, te, cfg, positive_index)
        block = te[[c for c in cfg.id_columns if c in te.columns]].copy()
        block["score"] = np.asarray(scores, dtype=float)
        block["y"] = pd.to_numeric(te[cfg.reg_target()], errors="coerce").to_numpy(float)
        for c in diagnostic_columns:
            if c in te.columns:
                block[c] = te[c].to_numpy()
        blocks.append(block)
        print(f"  fold {fi}: train={len(inner_tr)} val={len(inner_val)} test={len(te)}")

    if not blocks:
        return pd.DataFrame(columns=list(cfg.id_columns) + ["score", "y"])
    oof = pd.concat(blocks, ignore_index=True)
    idx = [c for c in cfg.id_columns if c in oof.columns]
    return oof.set_index(idx) if idx else oof


def write_artifact_bundle(output_dir: Path, bundle_path: Path) -> None:
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                tar.add(path, arcname=path.name)
