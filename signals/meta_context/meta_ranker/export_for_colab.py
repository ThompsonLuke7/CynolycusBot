"""
Bundle the Meta Ranker training matrix + manifest + trainer for Colab.

Usage: python meta_context/meta_ranker/export_for_colab.py
Produces: meta_context/meta_ranker/meta_ranker_colab_bundle.tgz
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
HARNESS = REPO_ROOT / "strategies" / "model_training" / "colab_competition.py"
# The RESEARCH matrix, not the live rolling window. The live file is scored by
# the DEPLOYED base models, so training on it would fit on in-sample base
# predictions — the exact leak the walk-forward OOF exists to prevent.
FILES = ["meta_ranker_matrix_research.parquet", "manifest.json", "manifest_upside.json",
         "meta_ranker_train_colab.py"]
BUNDLE = HERE / "meta_ranker_colab_bundle.tgz"

# The theme block is dropped from the TRAINING feature set by default. Measured
# 2026-09-12/13 (research/execution_quality/24_horizon_thesis_experiments.md §4):
# with-theme test rho +0.0029 vs no-theme +0.0197, i.e. the block contributes
# -0.0167 [-0.0222, -0.0111]. Three of the twelve columns are also 100% NaN.
#
# The columns are dropped from the manifest copy written INTO the bundle; the
# on-disk manifest is never modified, because it describes the DEPLOYED model,
# which was trained on all 92 features. Editing it in place would leave the
# repo claiming a feature set no live model has. The on-disk manifest is
# rewritten by build_meta_ranker_matrix.py when a new model is actually
# deployed. Pass --keep-themes to train with them anyway.
MANIFESTS = ["manifest.json", "manifest_upside.json"]


def manifest_without_themes(manifest: dict) -> dict:
    """Manifest with the theme block removed from `feature_columns`, and a note
    saying so. Untouched when the manifest declares no theme block."""
    # The measured block is NOT just `theme_context_columns`. The ablation
    # (scripts/horizon_thesis/theme_lookahead_audit.py::THEME_FEATURES) removed
    # 13 columns: those 11 plus `within_theme_mom_rank` and
    # `theme_crowding_frac`, which live in cross_context_columns but are
    # computed WITHIN a theme and so inherit the same unstable membership
    # (83-88% of tickers change primary theme in 6-9 days). Dropping only the
    # declared block would leave two of the ablated columns in and no longer
    # match the -0.0167 measurement.
    declared = list(manifest.get("theme_context_columns") or [])
    cross_derived = [c for c in (manifest.get("cross_context_columns") or [])
                     if "theme" in c.lower()]
    feature_set = set(manifest.get("feature_columns", []))
    theme_cols = [c for c in dict.fromkeys(declared + cross_derived) if c in feature_set]
    if not theme_cols:
        return manifest
    out = dict(manifest)
    out["feature_columns"] = [c for c in manifest["feature_columns"]
                              if c not in set(theme_cols)]
    out["dropped_feature_blocks"] = {
        "theme_context": {
            "columns": theme_cols,
            "reason": "ablation rho -0.0167 [-0.0222, -0.0111]; 3 are 100% NaN",
            "evidence": "research/execution_quality/24_horizon_thesis_experiments.md §4",
        }
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-themes", action="store_true",
                    help="train with the theme block despite its measured -0.0167 rho")
    args = ap.parse_args()
    missing = [f for f in FILES if not (HERE / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing}; run build_meta_ranker_matrix.py first")
    if not HARNESS.exists():
        raise FileNotFoundError(f"missing competition harness at {HARNESS}")
    # Ship the shared harness next to the trainer (the trainer imports it top-level).
    shutil.copy2(HARNESS, HERE / HARNESS.name)
    staged = Path(tempfile.mkdtemp(prefix="meta_bundle_"))
    try:
        for name in MANIFESTS:
            manifest = json.loads((HERE / name).read_text())
            if not args.keep_themes:
                manifest = manifest_without_themes(manifest)
                kept = len(manifest["feature_columns"])
                dropped = manifest.get("dropped_feature_blocks", {}).get("theme_context", {})
                print(f"{name}: {kept} features "
                      f"(dropped {len(dropped.get('columns', []))} theme columns)")
            (staged / name).write_text(json.dumps(manifest, indent=2))
        with tarfile.open(BUNDLE, "w:gz") as tar:
            for f in FILES:
                # The trainer expects `meta_ranker_matrix.parquet` inside the
                # bundle; ship the research file under that name so Colab needs
                # no changes. Manifests come from the staging dir, not HERE.
                if f in MANIFESTS:
                    tar.add(staged / f, arcname=f)
                    continue
                arc = "meta_ranker_matrix.parquet" if f.startswith("meta_ranker_matrix_research") else f
                tar.add(HERE / f, arcname=arc)
            tar.add(HERE / HARNESS.name, arcname=HARNESS.name)
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    mb = BUNDLE.stat().st_size / 1e6
    print(f"wrote {BUNDLE}  ({mb:.1f} MB)")
    with tarfile.open(BUNDLE) as tar:
        print("contents:", tar.getnames())


if __name__ == "__main__":
    main()
