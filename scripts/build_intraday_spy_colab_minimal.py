from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "colab_intraday_spy_minimal.zip"


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file())
    raise FileNotFoundError(f"Missing required path: {path}")


def build_zip(output_path: Path) -> tuple[int, int]:
    include_paths = [
        REPO_ROOT / "strategies" / "spy_intraday" / "Models" / "ga_xgboost",
        REPO_ROOT / "strategies" / "spy_intraday" / "Features",
        REPO_ROOT / "Data" / "load_data.py",
        REPO_ROOT / "Data" / "retrieve_data.py",
        REPO_ROOT / "Data" / "plots" / "plots.py",
        REPO_ROOT / "strategies" / "spy_intraday" / "Policy" / "training_logging.py",
        REPO_ROOT / "Data" / "processed" / "spy" / "datasets" / "10min_shift1",
        REPO_ROOT / "Data" / "processed" / "spy" / "splits" / "10min_shift1",
        REPO_ROOT / "Data" / "processed" / "spy" / "stats" / "norm_stats_10min_shift1_X_10min_shift1_tree_train.json",
        REPO_ROOT / "Data" / "models" / "ga_xgboost" / "10min" / "long" / "swing",
        REPO_ROOT / "Data" / "models" / "ga_xgboost" / "10min" / "short" / "swing",
    ]

    files: list[Path] = []
    for path in include_paths:
        files.extend(_iter_files(path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_bytes = 0
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, arcname=path.relative_to(REPO_ROOT))
            total_bytes += path.stat().st_size
    return len(files), total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a minimal Colab zip for intraday SPY swing retraining.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output zip path.",
    )
    args = parser.parse_args()
    file_count, total_bytes = build_zip(args.output.resolve())
    print(f"Wrote {args.output} with {file_count} files ({total_bytes / (1024 * 1024):.1f} MiB uncompressed input).")


if __name__ == "__main__":
    main()
