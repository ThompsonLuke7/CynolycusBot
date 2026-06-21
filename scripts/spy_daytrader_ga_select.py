"""GA feature selection for the SPY day trader, per swing-pivot label.

Runs GAXGBoostFeatureSelector (the established pre-training step) over the full
SPY context feature set and writes the selected <=max_features subset to JSON.
The last `holdout_frac` of rows is kept OUT of GA entirely so the downstream
competition's test split stays untouched by selection.

Example (long model, 300-feature cap):
    .venv/bin/python scripts/spy_daytrader_ga_select.py --label long_swing_label
    .venv/bin/python scripts/spy_daytrader_ga_select.py --label short_swing_label

Smoke test first (tiny GA, ~minutes) to confirm wiring:
    .venv/bin/python scripts/spy_daytrader_ga_select.py --label long_swing_label \
        --generations 3 --population 8 --max-features 50 --out /tmp/ga_smoke_long.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.spy_intraday.Models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector

DEFAULT_MATRIX = REPO_ROOT / "Data/processed/spy/training_export/spy_daytrader_context_matrix.parquet"
DEFAULT_MANIFEST = REPO_ROOT / "Data/processed/spy/training_export/spy_daytrader_context_manifest.json"
DEFAULT_OUT_DIR = REPO_ROOT / "Data/processed/spy/training_export/feature_selection"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="binary target column, e.g. long_swing_label / short_swing_label")
    ap.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--max-features", type=int, default=300)
    ap.add_argument("--holdout-frac", type=float, default=0.15,
                    help="tail fraction kept out of GA (matches competition test split)")
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--fitness", default="logloss_penalized")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    feats: list[str] = manifest["feature_columns"]
    ts_col = manifest.get("timestamp_column", "timestamp")

    cols = list(dict.fromkeys(feats + [args.label, ts_col]))
    df = pd.read_parquet(args.matrix, columns=cols)
    if ts_col not in df.columns:
        df = df.reset_index()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

    y_raw = pd.to_numeric(df[args.label], errors="coerce")
    df = df.assign(_y=(y_raw == 1).astype("Int64"))
    df = df.dropna(subset=["_y"]).sort_values(ts_col, kind="mergesort").reset_index(drop=True)

    # keep the competition's test tail out of selection
    n = len(df)
    cut = int(n * (1.0 - args.holdout_frac))
    train = df.iloc[:cut]
    X = (train[feats].apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan).to_numpy(np.float32))
    y = train["_y"].to_numpy(np.int64)

    pos = int(y.sum())
    print(f"label={args.label}  rows(GA)={len(train):,} / total={n:,}  "
          f"positives={pos} ({pos/len(y):.3%})  features_in={len(feats)}  cap={args.max_features}")
    if pos < 50:
        raise SystemExit(f"Too few positives ({pos}) for {args.label}; check the label column.")

    selector = GAXGBoostFeatureSelector(
        population_size=args.population,
        generations=args.generations,
        max_features=args.max_features,
        fitness_metric=args.fitness,
        random_state=args.seed,
    )
    selector.fit(X, y)

    mask = np.asarray(selector.best_mask_, dtype=bool)
    selected = [f for f, keep in zip(feats, mask) if keep]
    print(f"selected {len(selected)} / {len(feats)} features  (GA best score={selector.best_score_:.5f})")

    out = args.out or (DEFAULT_OUT_DIR / f"selected_{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": args.label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_features_in": len(feats),
        "n_selected": len(selected),
        "max_features": args.max_features,
        "holdout_frac": args.holdout_frac,
        "ga_best_score": float(selector.best_score_),
        "ga_params": {
            "population_size": args.population,
            "generations": args.generations,
            "fitness_metric": args.fitness,
            "random_state": args.seed,
        },
        "matrix": str(args.matrix.relative_to(REPO_ROOT)),
        "features": selected,
    }, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
