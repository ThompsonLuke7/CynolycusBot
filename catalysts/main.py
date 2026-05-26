"""CLI for the unified catalyst layer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from catalysts.config import CATALYST_FEATURE_MATRIX_PATH
from catalysts.pipeline import build_catalyst_features, build_catalyst_records, build_catalyst_scores


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified catalyst pipeline.")
    parser.add_argument("--stage", choices=["records", "scores", "features", "all"], required=True)
    parser.add_argument("--timestamps-csv", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.stage in {"records", "all"}:
        build_catalyst_records()
    if args.stage in {"scores", "all"}:
        build_catalyst_scores()
    if args.stage in {"features", "all"}:
        if not args.timestamps_csv:
            raise ValueError("--timestamps-csv is required for catalyst features")
        timestamps = _read_table(args.timestamps_csv)
        build_catalyst_features(
            timestamps,
            output_path=Path(args.output) if args.output else CATALYST_FEATURE_MATRIX_PATH,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
