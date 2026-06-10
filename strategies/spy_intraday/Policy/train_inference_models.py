from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[3]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()


def _run_cmd(args: list[str]) -> None:
    print(f"[train_inference_models] Running: {' '.join(args)}")
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GA-XGB models for inference (full-fit by default)."
    )
    parser.add_argument(
        "--processed-root",
        required=True,
        help="Processed data root (contains datasets/<dataset>).",
    )
    parser.add_argument(
        "--model-root",
        required=True,
        help="Output model root (will create ga_xgboost/<dataset>).",
    )
    parser.add_argument("--dataset-name", default="15min")
    parser.add_argument("--label-modes", default="pivot,tb")
    parser.add_argument("--refresh-masks", action="store_true")
    parser.add_argument("--full-fit", action="store_true", default=True)
    parser.add_argument("--no-full-fit", dest="full_fit", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    processed_root = Path(args.processed_root)
    model_root = Path(args.model_root)
    label_modes = [s.strip() for s in args.label_modes.split(",") if s.strip()]
    if not label_modes:
        raise SystemExit("No label modes specified.")

    for label_mode in label_modes:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "strategies" / "spy_intraday" / "Models" / "ga_xgboost" / "train.py"),
            "--label-mode",
            label_mode,
            "--processed-root",
            str(processed_root),
            "--model-root",
            str(model_root),
        ]
        if args.full_fit:
            cmd.append("--full-fit")
        if args.refresh_masks:
            cmd.append("--refresh-masks")
        _run_cmd(cmd)


if __name__ == "__main__":
    main()
