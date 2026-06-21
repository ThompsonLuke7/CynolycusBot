"""Build a Colab competition bundle for the SPY day trader using a GA-selected
feature subset and the correct swing-pivot label.

Takes the JSON written by spy_daytrader_ga_select.py, rewrites the context
manifest's feature_columns to the selected subset, and ships the canonical
colab_competition.py (the fixed NDCG-based best-model selector).

Example:
    .venv/bin/python scripts/spy_daytrader_make_reduced_bundle.py \
        --selected Data/processed/spy/training_export/feature_selection/selected_long_swing_label.json
    .venv/bin/python scripts/spy_daytrader_make_reduced_bundle.py \
        --selected Data/processed/spy/training_export/feature_selection/selected_short_swing_label.json

Then in Colab, upload the produced bundle and run with:
    SPY_TARGET=<label>  (printed at the end of this script)
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = REPO_ROOT / "Data/processed/spy/training_export"
BASE_MANIFEST = EXPORT_DIR / "spy_daytrader_context_manifest.json"
MATRIX = EXPORT_DIR / "spy_daytrader_context_matrix.parquet"
COLAB_SCRIPT = REPO_ROOT / "strategies/spy_intraday/Models/colab/spy_daytrader_train_colab.py"
CANONICAL_COMPETITION = REPO_ROOT / "strategies/model_training/colab_competition.py"
MANIFEST_NAME = "spy_daytrader_context_manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", type=Path, required=True, help="selected_<label>.json from spy_daytrader_ga_select.py")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sel = json.loads(args.selected.read_text())
    label = sel["label"]
    feats = sel["features"]
    if not feats:
        raise SystemExit("selected feature list is empty")

    manifest = json.loads(BASE_MANIFEST.read_text())
    missing = [f for f in feats if f not in set(manifest["feature_columns"])]
    if missing:
        raise SystemExit(f"{len(missing)} selected features not in base manifest, e.g. {missing[:5]}")

    manifest["feature_columns"] = feats
    manifest["feature_selection"] = {
        "method": "GAXGBoostFeatureSelector",
        "target_label": label,
        "n_selected": len(feats),
        "n_features_in": sel.get("n_features_in"),
        "source": str(args.selected.relative_to(REPO_ROOT)),
        "ga_best_score": sel.get("ga_best_score"),
    }

    out = args.out or (EXPORT_DIR / f"spy_daytrader_{label}_colab_bundle.tgz")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
        shutil.copy2(COLAB_SCRIPT, tmp / "spy_daytrader_train_colab.py")
        shutil.copy2(CANONICAL_COMPETITION, tmp / "colab_competition.py")
        with tarfile.open(out, "w:gz") as tar:
            tar.add(tmp / MANIFEST_NAME, arcname=MANIFEST_NAME)
            tar.add(MATRIX, arcname=manifest.get("matrix_file", MATRIX.name))
            tar.add(tmp / "spy_daytrader_train_colab.py", arcname="spy_daytrader_train_colab.py")
            tar.add(tmp / "colab_competition.py", arcname="colab_competition.py")

    print(f"wrote {out}  ({len(feats)} features, label={label})")
    print(f"  -> in Colab set:  SPY_TARGET={label}")


if __name__ == "__main__":
    main()
