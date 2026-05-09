"""
Import a Colab-trained booster bundle back into the repo.

Usage:
    python -m momentum_expansion.models.import_from_colab path/to/momentum_model_bundle.tgz
"""
from __future__ import annotations

import argparse
import logging
import shutil
import tarfile
from pathlib import Path

from momentum_expansion.config.momentum_config import MODELS_DIR

logger = logging.getLogger(__name__)


def import_bundle(bundle: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(MODELS_DIR)
    logger.info("Booster bundle extracted -> %s", MODELS_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    args = p.parse_args()
    import_bundle(args.bundle)
