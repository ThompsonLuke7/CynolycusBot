from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _read_frame(path_like: str | Path) -> pd.DataFrame:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file: {path}")


def _derive_intent(trace: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(trace["timestamp"], errors="coerce")
    if "action_dir_idx" in trace.columns:
        idx = pd.to_numeric(trace["action_dir_idx"], errors="coerce")
        d = pd.Series(0.0, index=trace.index)
        d[idx == 1] = 1.0
        d[idx == 2] = -1.0
        out["htf_dir"] = d
    else:
        out["htf_dir"] = np.sign(pd.to_numeric(trace.get("action", 0.0), errors="coerce").fillna(0.0))
    if "action_mag" in trace.columns:
        out["htf_conf"] = pd.to_numeric(trace["action_mag"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        out["htf_conf"] = pd.to_numeric(trace.get("action", 0.0), errors="coerce").abs().fillna(0.0).clip(0.0, 1.0)
    if "convex_atr_scale" in trace.columns:
        out["htf_atr_pct"] = pd.to_numeric(trace["convex_atr_scale"], errors="coerce")
    else:
        out["htf_atr_pct"] = np.nan
    out["htf_expected_edge"] = np.nan
    out = out[out["timestamp"].notna()].copy()
    return out


def _run_segment_eval(
    *,
    agent_eval_script: Path,
    model_path: Path,
    segment_csv: Path,
    trace_out: Path,
) -> None:
    cmd = [
        sys.executable,
        str(agent_eval_script),
        "--data-csv",
        str(segment_csv),
        "--model-path",
        str(model_path),
        "--plot-tail",
        "50",
        "--plot-out",
        str(trace_out.with_suffix(".png")),
        "--trace-out",
        str(trace_out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Segment eval failed for model {model_path}:\n{proc.stdout}\n{proc.stderr}"
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build walk-forward OOF 15m intent file for 1m execution training.")
    p.add_argument("--matrix-path", required=True, help="15m agent matrix (csv/parquet) with timestamp/features.")
    p.add_argument(
        "--manifest",
        required=True,
        help="CSV with columns: start_ts,end_ts,model_path[,fold_id].",
    )
    p.add_argument(
        "--out-path",
        default="Data/processed/spy/execution_agent/htf_intent_oof.parquet",
    )
    p.add_argument("--agent-eval-script", default="strategies/spy_intraday/Policy/Agent/run_eval.py")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    matrix = _read_frame(args.matrix_path)
    if "timestamp" not in matrix.columns:
        raise ValueError("matrix must contain timestamp column.")
    matrix["timestamp"] = pd.to_datetime(matrix["timestamp"], errors="coerce")
    matrix = matrix[matrix["timestamp"].notna()].copy().sort_values("timestamp")

    manifest = pd.read_csv(args.manifest)
    needed = {"start_ts", "end_ts", "model_path"}
    if not needed.issubset(set(manifest.columns)):
        raise ValueError(f"manifest must include {sorted(needed)}")
    manifest["start_ts"] = pd.to_datetime(manifest["start_ts"], errors="coerce")
    manifest["end_ts"] = pd.to_datetime(manifest["end_ts"], errors="coerce")
    manifest = manifest.dropna(subset=["start_ts", "end_ts"]).copy()
    if manifest.empty:
        raise ValueError("Manifest is empty after timestamp parse.")

    eval_script = Path(args.agent_eval_script)
    if not eval_script.exists():
        raise FileNotFoundError(eval_script)

    chunks: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="intent_oof_") as td:
        tmp = Path(td)
        for i, row in manifest.iterrows():
            start_ts = row["start_ts"]
            end_ts = row["end_ts"]
            model_path = Path(str(row["model_path"]))
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            seg = matrix[(matrix["timestamp"] >= start_ts) & (matrix["timestamp"] < end_ts)].copy()
            if seg.empty:
                continue
            seg_csv = tmp / f"seg_{i:04d}.csv"
            trace_csv = tmp / f"seg_trace_{i:04d}.csv"
            seg.to_csv(seg_csv, index=False)
            _run_segment_eval(
                agent_eval_script=eval_script,
                model_path=model_path,
                segment_csv=seg_csv,
                trace_out=trace_csv,
            )
            trace = pd.read_csv(trace_csv)
            intent = _derive_intent(trace)
            intent["fold_id"] = row["fold_id"] if "fold_id" in manifest.columns else i
            intent["model_path"] = str(model_path)
            chunks.append(intent)

    if not chunks:
        raise RuntimeError("No OOF intent rows produced.")
    out = pd.concat(chunks, axis=0, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".csv"}:
        out.to_csv(out_path, index=False)
    else:
        out.to_parquet(out_path, index=False)
    print(f"Saved OOF intent: {out_path} rows={len(out):,}")


if __name__ == "__main__":
    main()

