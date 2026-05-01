from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from run_phase4_focused_model_competition import _command


SELECTED_RUNS = [
    ("nonshift_setup_area", 0.42, 0.15, 2.0, 12, 1.0, 0.8),
    ("nonshift_setup_area", 0.42, 0.20, 2.0, 12, 1.0, 0.8),
    ("shift1_setup_area", 0.42, 0.20, 2.0, 16, 1.5, 1.0),
    ("shift1_setup_area", 0.42, 0.20, 2.0, 12, 1.0, 0.8),
    ("nonshift_swing", 0.42, 0.15, 2.0, 16, 1.5, 1.0),
]


def _with_trace_output(cmd: list[str]) -> list[str]:
    out = list(cmd)
    idx = out.index("--plot-top-n")
    out[idx + 1] = "1"
    return out


def main() -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp/cynolycus_matplotlib").resolve()))
    for i, run in enumerate(SELECTED_RUNS, start=1):
        cmd, out_dir = _command(run)
        cmd = _with_trace_output(cmd)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[traces] {i}/{len(SELECTED_RUNS)} {out_dir.name}", flush=True)
        started = time.monotonic()
        with (out_dir / "trace_refresh.log").open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=540, env=env)
        print(f"[traces] done {out_dir.name} code={proc.returncode} elapsed={time.monotonic() - started:.1f}s", flush=True)
        if proc.returncode != 0:
            sys.exit(int(proc.returncode))


if __name__ == "__main__":
    main()
