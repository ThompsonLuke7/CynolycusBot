from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data.load_data import get_ticker_processed_base_dir, get_ticker_processed_split_dir
from Data.retrieve_data import normalize_ticker
from Features.feature_matrix_regime import AgentFeatureConfig, build_agent_feature_matrix
from Features.label_generations import build_meta_entry_labels, build_meta_exit_labels
from Models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector
from Policy.training_logging import log_training_run


DART_DEFAULTS: dict[str, object] = {
    "rate_drop": 0.1,
    "skip_drop": 0.5,
    "one_drop": 1,
    "sample_type": "uniform",
    "normalize_type": "tree",
}

ALL_TARGETS = ["y_enter_long", "y_enter_short", "y_exit_long", "y_exit_short"]


@dataclass(frozen=True)
class TrainConfig:
    ticker: str = "SPY"
    dataset_name: str = "10min"
    x_filename: str = "X_10min_tree.parquet"
    processed_root: str | None = None
    ga_model_root: str | None = None
    model_root: str | None = None
    targets: str = "all"
    pivot_label_dir: str = "swing"
    tb_label_dir: str = "tb"
    include_vix_features: bool = True
    session_tz: str = "America/New_York"
    atr_col: str = "atr"
    a_tp: float = 1.6
    b_sl: float = 0.8
    cost_bps: float = 2.0
    use_next_open: bool = True
    hazard_k: int = 2
    xgb_booster: str | None = None
    xgb_rate_drop: float | None = None
    xgb_skip_drop: float | None = None
    xgb_one_drop: bool | None = None
    xgb_sample_type: str | None = None
    xgb_normalize_type: str | None = None
    n_estimators: int | None = None
    early_stopping_rounds: int = 50
    random_state: int = 42


def _normalize_target_token(token: str) -> list[str]:
    key = token.strip().lower()
    if key == "all":
        return list(ALL_TARGETS)
    if key == "enter":
        return ["y_enter_long", "y_enter_short"]
    if key == "exit":
        return ["y_exit_long", "y_exit_short"]
    aliases = {
        "enter_long": "y_enter_long",
        "long_enter": "y_enter_long",
        "enter_short": "y_enter_short",
        "short_enter": "y_enter_short",
        "exit_long": "y_exit_long",
        "long_exit": "y_exit_long",
        "exit_short": "y_exit_short",
        "short_exit": "y_exit_short",
        "y_enter_long": "y_enter_long",
        "y_enter_short": "y_enter_short",
        "y_exit_long": "y_exit_long",
        "y_exit_short": "y_exit_short",
    }
    if key in aliases:
        return [aliases[key]]
    raise ValueError(f"Unknown target token: {token}")


def _resolve_targets(spec: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in [p.strip() for p in str(spec).split(",") if p.strip()]:
        for col in _normalize_target_token(token):
            if col not in seen:
                seen.add(col)
                out.append(col)
    if not out:
        return list(ALL_TARGETS)
    return out


def _sanitize_feature_matrix_for_xgboost(X: np.ndarray) -> np.ndarray:
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    if np.isfinite(X).all():
        return X
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    return X


def _load_split_indices(
    ticker: str,
    dataset_name: str,
    x_filename: str,
    *,
    processed_root: Path | None = None,
) -> dict[str, np.ndarray]:
    clean = normalize_ticker(ticker)
    if processed_root is None:
        split_root = get_ticker_processed_split_dir(clean)
    else:
        split_root = processed_root / "splits"
    x_stem = Path(x_filename).stem
    candidates = [
        split_root / dataset_name / x_stem,
        split_root / dataset_name,
    ]
    for split_dir in candidates:
        train_path = split_dir / "train_idx.npy"
        val_path = split_dir / "val_idx.npy"
        test_path = split_dir / "test_idx.npy"
        if train_path.exists() and val_path.exists() and test_path.exists():
            return {
                "train": np.sort(np.load(train_path)),
                "val": np.sort(np.load(val_path)),
                "test": np.sort(np.load(test_path)),
            }
    raise FileNotFoundError(
        f"Missing split files under {split_root / dataset_name} (x_stem={x_stem})."
    )


def _sanitize_xgb_params(xgb_params: dict) -> dict:
    params = dict(xgb_params)
    params["eval_metric"] = "logloss"
    booster = str(params.get("booster", "gbtree")).lower()
    if booster != "dart":
        params.pop("rate_drop", None)
        params.pop("skip_drop", None)
        params.pop("one_drop", None)
        params.pop("sample_type", None)
        params.pop("normalize_type", None)
    return params


def _xgb_params_from_config(cfg: TrainConfig) -> dict:
    params = GAXGBoostFeatureSelector().xgb_params.copy()
    params["seed"] = int(cfg.random_state)
    if cfg.n_estimators is not None and int(cfg.n_estimators) > 0:
        params["n_estimators"] = int(cfg.n_estimators)
    if cfg.xgb_booster is not None:
        params["booster"] = str(cfg.xgb_booster)
    if str(params.get("booster", "gbtree")).lower() == "dart":
        params.update(DART_DEFAULTS)
    if cfg.xgb_rate_drop is not None:
        params["rate_drop"] = float(cfg.xgb_rate_drop)
    if cfg.xgb_skip_drop is not None:
        params["skip_drop"] = float(cfg.xgb_skip_drop)
    if cfg.xgb_one_drop is not None:
        params["one_drop"] = int(bool(cfg.xgb_one_drop))
    if cfg.xgb_sample_type is not None:
        params["sample_type"] = str(cfg.xgb_sample_type)
    if cfg.xgb_normalize_type is not None:
        params["normalize_type"] = str(cfg.xgb_normalize_type)
    return _sanitize_xgb_params(params)


def _derive_session_date(index: pd.Index, *, tz: str) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Feature matrix must use DatetimeIndex for session derivation.")
    idx = index
    if idx.tz is not None:
        idx = idx.tz_convert(tz)
    return idx.normalize().date


def _build_feature_matrix_with_labels(cfg: TrainConfig) -> pd.DataFrame:
    clean = normalize_ticker(cfg.ticker)
    processed_root = Path(cfg.processed_root) if cfg.processed_root else None
    if cfg.ga_model_root:
        ga_model_root = Path(cfg.ga_model_root)
    else:
        ga_model_root = REPO_ROOT / "Data" / "models" / "ga_xgboost" / cfg.dataset_name
    ga_summary_path = ga_model_root / "training_run_summary.json"
    agent_cfg = AgentFeatureConfig(
        ticker=clean,
        dataset_name=cfg.dataset_name,
        model_name="ga_xgboost",
        processed_root=processed_root,
        model_root=ga_model_root,
        pivot_label_dir=cfg.pivot_label_dir,
        tb_label_dir=cfg.tb_label_dir,
        include_pivot_probs=True,
        include_tb_probs=True,
        include_state_placeholders=False,
        include_vix_features=bool(cfg.include_vix_features),
        drop_na=False,
        tz=cfg.session_tz,
    )
    feat_df = build_agent_feature_matrix(config=agent_cfg)
    if not {"open", "high", "low", "close"}.issubset(feat_df.columns):
        raise KeyError("Agent feature matrix must include open/high/low/close columns.")

    label_cols = ["open", "high", "low", "close"]
    for col in (
        "p_pivot_long",
        "p_pivot_short",
        "p_tb_long",
        "p_tb_short",
        "tb_long_label",
        "tb_short_label",
        "trend_phase_label",
        "trend_phase_ignition",
        "trend_phase_expansion",
        "trend_phase_m",
        "trend_phase_a",
    ):
        if col in feat_df.columns and col not in label_cols:
            label_cols.append(col)
    label_df = feat_df[label_cols].copy()
    label_df[cfg.atr_col] = ta.atr(
        label_df["high"], label_df["low"], label_df["close"], length=14
    )
    label_df["session_date"] = _derive_session_date(feat_df.index, tz=cfg.session_tz)
    label_df = build_meta_entry_labels(
        label_df,
        atr_col=cfg.atr_col,
        a_tp=cfg.a_tp,
        b_sl=cfg.b_sl,
        use_next_open=cfg.use_next_open,
        cost_bps=cfg.cost_bps,
        day_col="session_date",
        thresholds_summary_path=ga_summary_path,
        use_summary_thresholds=True,
    )
    label_df = build_meta_exit_labels(
        label_df,
        atr_col=cfg.atr_col,
        enter_long_col="y_enter_long",
        enter_short_col="y_enter_short",
        a_tp=cfg.a_tp,
        b_sl=cfg.b_sl,
        use_next_open=cfg.use_next_open,
        cost_bps=cfg.cost_bps,
        K=cfg.hazard_k,
        day_col="session_date",
    )
    for col in ALL_TARGETS:
        feat_df[col] = label_df[col].to_numpy(dtype=np.int8)
    return feat_df


def _select_feature_columns(df: pd.DataFrame, *, targets: list[str]) -> list[str]:
    excluded = set(targets) | {"timestamp"}
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns found for training.")
    return cols


def _predict_proba(
    model: xgb.Booster | None,
    X: np.ndarray,
    *,
    constant_prob: float | None = None,
) -> np.ndarray:
    if X.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if model is None:
        p = 0.0 if constant_prob is None else float(constant_prob)
        return np.full(X.shape[0], p, dtype=np.float32)
    dmat = xgb.DMatrix(X)
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None and int(best_iter) >= 0:
        return model.predict(dmat, iteration_range=(0, int(best_iter) + 1)).astype(np.float32)
    return model.predict(dmat).astype(np.float32)


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {"n": float(y_true.size)}
    if y_true.size == 0:
        return out
    pred = (probs >= 0.5).astype(np.int8)
    out["pos_rate"] = float(np.mean(y_true))
    out["acc"] = float(np.mean(pred == y_true))
    out["f1"] = float(f1_score(y_true, pred, zero_division=0))
    try:
        out["logloss"] = float(log_loss(y_true, probs, labels=[0, 1]))
    except ValueError:
        out["logloss"] = float("nan")
    try:
        out["auc"] = (
            float(roc_auc_score(y_true, probs))
            if len(np.unique(y_true)) > 1
            else float("nan")
        )
    except ValueError:
        out["auc"] = float("nan")
    try:
        out["ap"] = (
            float(average_precision_score(y_true, probs))
            if len(np.unique(y_true)) > 1
            else float("nan")
        )
    except ValueError:
        out["ap"] = float("nan")
    return out


def _train_one_target(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    splits: dict[str, np.ndarray],
    xgb_params: dict,
    early_stopping_rounds: int,
) -> dict:
    X = _sanitize_feature_matrix_for_xgboost(df[feature_cols].to_numpy(dtype=np.float32))
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    valid_rows = ((y == 0) | (y == 1))

    idx_train = splits["train"][valid_rows[splits["train"]]]
    idx_val = splits["val"][valid_rows[splits["val"]]]
    idx_test = splits["test"][valid_rows[splits["test"]]]

    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    train_rows = int(idx_train.size)
    constant_prob: float | None = None
    model: xgb.Booster | None = None
    evals_result: dict[str, dict[str, list[float]]] = {}

    params_local = dict(xgb_params)
    params_local["scale_pos_weight"] = (neg / max(pos, 1)) if train_rows > 0 else 1.0
    params_local = _sanitize_xgb_params(params_local)
    num_boost_round = int(params_local.pop("n_estimators", 100))
    n_jobs = params_local.pop("n_jobs", None)
    if n_jobs is not None:
        params_local["nthread"] = int(n_jobs)

    if train_rows == 0:
        constant_prob = 0.0
    elif pos == 0 or neg == 0:
        constant_prob = float(np.mean(y_train)) if train_rows > 0 else 0.0
    else:
        dtrain = xgb.DMatrix(X[idx_train], label=y_train)
        evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]
        early_stop = None
        if idx_val.size > 0:
            dval = xgb.DMatrix(X[idx_val], label=y_val)
            evals.append((dval, "val"))
            if int(early_stopping_rounds) > 0:
                early_stop = int(early_stopping_rounds)
        model = xgb.train(
            params_local,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stop,
            evals_result=evals_result,
            verbose_eval=False,
        )

    probs_full = np.full(len(df), np.nan, dtype=np.float32)
    probs_full[valid_rows] = _predict_proba(model, X[valid_rows], constant_prob=constant_prob)
    probs_val = _predict_proba(model, X[idx_val], constant_prob=constant_prob)
    probs_test = _predict_proba(model, X[idx_test], constant_prob=constant_prob)

    train_metrics = _binary_metrics(y_train, _predict_proba(model, X[idx_train], constant_prob=constant_prob))
    val_metrics = _binary_metrics(y_val, probs_val)
    test_metrics = _binary_metrics(y_test, probs_test)

    return {
        "target_col": target_col,
        "model": model,
        "params": params_local,
        "num_boost_round": num_boost_round,
        "constant_prob": constant_prob,
        "evals_result": evals_result,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "probs_full": probs_full,
        "probs_test": probs_test,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_pos": pos,
        "train_neg": neg,
    }


def _save_target_artifacts(
    *,
    out_dir: Path,
    result: dict,
    feature_cols: list[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = result["model"]
    if model is not None:
        model.save_model(str(out_dir / "xgb_model.json"))
    np.save(out_dir / "p_full.npy", result["probs_full"])
    np.save(out_dir / "p_test.npy", result["probs_test"])
    np.save(out_dir / "idx_train.npy", result["idx_train"])
    np.save(out_dir / "idx_val.npy", result["idx_val"])
    np.save(out_dir / "idx_test.npy", result["idx_test"])
    (out_dir / "feature_columns.txt").write_text("\n".join(feature_cols))

    meta = {
        "target_col": result["target_col"],
        "xgb_params": result["params"],
        "num_boost_round": int(result["num_boost_round"]),
        "constant_prob": result["constant_prob"],
        "train_pos": int(result["train_pos"]),
        "train_neg": int(result["train_neg"]),
        "train_metrics": result["train_metrics"],
        "val_metrics": result["val_metrics"],
        "test_metrics": result["test_metrics"],
        "evals_result": result["evals_result"],
        "best_iteration": (
            int(result["model"].best_iteration)
            if result["model"] is not None and getattr(result["model"], "best_iteration", None) is not None
            else None
        ),
        "best_score": (
            float(result["model"].best_score)
            if result["model"] is not None and getattr(result["model"], "best_score", None) is not None
            else None
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train meta XGBoost heads (no GA masking).")
    parser.add_argument("--ticker", type=str, default=TrainConfig.ticker)
    parser.add_argument("--dataset-name", type=str, default=TrainConfig.dataset_name)
    parser.add_argument("--x-filename", type=str, default=TrainConfig.x_filename)
    parser.add_argument(
        "--targets",
        type=str,
        default=TrainConfig.targets,
        help="Targets: all, enter, exit, or comma-separated explicit heads.",
    )
    parser.add_argument("--processed-root", type=str, default=None)
    parser.add_argument("--ga-model-root", type=str, default=None)
    parser.add_argument("--model-root", type=str, default=None)
    parser.add_argument("--pivot-label-dir", type=str, default=TrainConfig.pivot_label_dir)
    parser.add_argument("--tb-label-dir", type=str, default=TrainConfig.tb_label_dir)
    parser.add_argument(
        "--include-vix-features",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.include_vix_features,
    )
    parser.add_argument("--a-tp", type=float, default=TrainConfig.a_tp)
    parser.add_argument("--b-sl", type=float, default=TrainConfig.b_sl)
    parser.add_argument("--cost-bps", type=float, default=TrainConfig.cost_bps)
    parser.add_argument("--hazard-k", type=int, default=TrainConfig.hazard_k)
    parser.add_argument(
        "--use-next-open",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.use_next_open,
    )
    parser.add_argument("--xgb-booster", "--booster", choices=["gbtree", "dart"], default=None)
    parser.add_argument("--xgb-rate-drop", type=float, default=None)
    parser.add_argument("--xgb-skip-drop", type=float, default=None)
    parser.add_argument("--xgb-one-drop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--xgb-sample-type", choices=["uniform", "weighted"], default=None)
    parser.add_argument("--xgb-normalize-type", choices=["tree", "forest"], default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--early-stopping-rounds", type=int, default=TrainConfig.early_stopping_rounds)
    parser.add_argument("--random-state", type=int, default=TrainConfig.random_state)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        ticker=args.ticker,
        dataset_name=args.dataset_name,
        x_filename=args.x_filename,
        processed_root=args.processed_root,
        ga_model_root=args.ga_model_root,
        model_root=args.model_root,
        targets=args.targets,
        pivot_label_dir=args.pivot_label_dir,
        tb_label_dir=args.tb_label_dir,
        include_vix_features=bool(args.include_vix_features),
        a_tp=float(args.a_tp),
        b_sl=float(args.b_sl),
        cost_bps=float(args.cost_bps),
        hazard_k=int(args.hazard_k),
        use_next_open=bool(args.use_next_open),
        xgb_booster=args.xgb_booster,
        xgb_rate_drop=args.xgb_rate_drop,
        xgb_skip_drop=args.xgb_skip_drop,
        xgb_one_drop=args.xgb_one_drop,
        xgb_sample_type=args.xgb_sample_type,
        xgb_normalize_type=args.xgb_normalize_type,
        n_estimators=args.n_estimators,
        early_stopping_rounds=int(args.early_stopping_rounds),
        random_state=int(args.random_state),
    )
    targets = _resolve_targets(cfg.targets)
    print(f"[META-XGB] Targets={targets}")

    frame = _build_feature_matrix_with_labels(cfg)
    feature_cols = _select_feature_columns(frame, targets=targets)
    print(
        f"[META-XGB] Matrix rows={len(frame)}, feature_cols={len(feature_cols)} "
        f"(includes swing/tb probs + structure features)"
    )

    processed_root = Path(cfg.processed_root) if cfg.processed_root else None
    splits = _load_split_indices(
        cfg.ticker,
        cfg.dataset_name,
        cfg.x_filename,
        processed_root=processed_root,
    )
    print(
        "[META-XGB] Split sizes: "
        f"train={splits['train'].size}, val={splits['val'].size}, test={splits['test'].size}"
    )

    base_params = _xgb_params_from_config(cfg)
    print(
        f"[META-XGB] XGBoost objective={base_params.get('objective', 'binary:logistic')} "
        f"eval_metric={base_params.get('eval_metric', 'logloss')} "
        f"booster={base_params.get('booster', 'gbtree')}"
    )

    if cfg.model_root:
        model_root = Path(cfg.model_root)
    else:
        model_root = REPO_ROOT / "Data" / "models" / "meta_xgboost"
    out_dataset_root = model_root / cfg.dataset_name
    out_dataset_root.mkdir(parents=True, exist_ok=True)

    summary_targets: dict[str, dict] = {}
    for target_col in targets:
        result = _train_one_target(
            df=frame,
            feature_cols=feature_cols,
            target_col=target_col,
            splits=splits,
            xgb_params=base_params,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        artifact_dir = _save_target_artifacts(
            out_dir=out_dataset_root / target_col,
            result=result,
            feature_cols=feature_cols,
        )
        print(
            f"[META-XGB] {target_col}: "
            f"train_pos={result['train_pos']}, train_neg={result['train_neg']}, "
            f"val_f1={result['val_metrics'].get('f1', float('nan')):.4f}, "
            f"test_f1={result['test_metrics'].get('f1', float('nan')):.4f}, "
            f"saved={artifact_dir}"
        )
        summary_targets[target_col] = {
            "artifact_dir": str(artifact_dir),
            "train_metrics": result["train_metrics"],
            "val_metrics": result["val_metrics"],
            "test_metrics": result["test_metrics"],
            "train_pos": int(result["train_pos"]),
            "train_neg": int(result["train_neg"]),
        }

    log_paths = log_training_run(
        run_name="meta_xgboost_train",
        output_dir=out_dataset_root,
        hyperparameters={**asdict(cfg), "targets": targets, "feature_count": len(feature_cols)},
        train_metrics={k: v["train_metrics"] for k, v in summary_targets.items()},
        validation_metrics={k: v["val_metrics"] for k, v in summary_targets.items()},
        best_validation_metrics={
            k: {"test_f1": v["test_metrics"].get("f1"), "test_auc": v["test_metrics"].get("auc")}
            for k, v in summary_targets.items()
        },
        artifacts={
            "model_root": str(out_dataset_root),
            "targets": {k: v["artifact_dir"] for k, v in summary_targets.items()},
        },
        extra={
            "ticker": normalize_ticker(cfg.ticker),
            "dataset_name": cfg.dataset_name,
            "rows": int(len(frame)),
        },
    )
    print(f"[META-XGB] Saved training summary: {log_paths['latest_path']}")


if __name__ == "__main__":
    main()
