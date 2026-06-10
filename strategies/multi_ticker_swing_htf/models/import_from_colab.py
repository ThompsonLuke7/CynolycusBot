"""
Import a Colab-trained HTF swing booster bundle back into the repo.

Usage:
    python -m multi_ticker_swing_htf.models.import_from_colab path/to/htf_swing_model_bundle.tgz
"""
from __future__ import annotations

import argparse
import logging
import tarfile
from pathlib import Path

from strategies.multi_ticker_swing_htf.config import MODELS_DIR

logger = logging.getLogger(__name__)


def import_bundle(bundle: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        try:
            tar.extractall(MODELS_DIR, filter="data")
        except TypeError:
            tar.extractall(MODELS_DIR)
    logger.info("HTF booster bundle extracted -> %s", MODELS_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    args = p.parse_args()
    import_bundle(args.bundle)
