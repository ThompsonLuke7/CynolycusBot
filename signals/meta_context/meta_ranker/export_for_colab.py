"""
Bundle the Meta Ranker training matrix + manifest + trainer for Colab.

Usage: python meta_context/meta_ranker/export_for_colab.py
Produces: meta_context/meta_ranker/meta_ranker_colab_bundle.tgz
"""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
HARNESS = REPO_ROOT / "strategies" / "model_training" / "colab_competition.py"
FILES = ["meta_ranker_matrix.parquet", "manifest.json", "manifest_upside.json", "meta_ranker_train_colab.py"]
BUNDLE = HERE / "meta_ranker_colab_bundle.tgz"


def main():
    missing = [f for f in FILES if not (HERE / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing}; run build_meta_ranker_matrix.py first")
    if not HARNESS.exists():
        raise FileNotFoundError(f"missing competition harness at {HARNESS}")
    # Ship the shared harness next to the trainer (the trainer imports it top-level).
    shutil.copy2(HARNESS, HERE / HARNESS.name)
    with tarfile.open(BUNDLE, "w:gz") as tar:
        for f in FILES:
            tar.add(HERE / f, arcname=f)
        tar.add(HERE / HARNESS.name, arcname=HARNESS.name)
    mb = BUNDLE.stat().st_size / 1e6
    print(f"wrote {BUNDLE}  ({mb:.1f} MB)")
    with tarfile.open(BUNDLE) as tar:
        print("contents:", tar.getnames())


if __name__ == "__main__":
    main()
