from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


OUT_ROOT = Path("Data/models/ga_xgboost/model_competition_phase4_focused")
EXECUTION_1M = "Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet"
TIMEOUT_SECONDS = 540

MODELS = {
    "nonshift_swing": {
        "dataset": "10min",
        "x_filename": "X_10min_tree.parquet",
        "single_label_dir": "swing_single",
        "note": "clean OOS non-shift swing model",
    },
    "nonshift_setup_area": {
        "dataset": "10min",
        "x_filename": "X_10min_tree.parquet",
        "single_label_dir": "swing_support_single",
        "note": "setup-area/support model; full-fit probabilities, optimistic",
    },
    "shift1_setup_area": {
        "dataset": "10min_shift1",
        "x_filename": "X_10min_shift1_tree.parquet",
        "single_label_dir": "swing_support_single",
        "note": "shifted one-bar setup-area/support model",
    },
}

RUNS = [
    ("nonshift_swing", 0.42, 0.15, None, 12, 1.0, 0.8),
    ("nonshift_swing", 0.42, 0.15, 2.0, 12, 1.0, 0.8),
    ("nonshift_swing", 0.42, 0.15, None, 16, 1.5, 1.0),
    ("nonshift_swing", 0.42, 0.15, 2.0, 16, 1.5, 1.0),
    ("nonshift_setup_area", 0.42, 0.15, None, 12, 1.0, 0.8),
    ("nonshift_setup_area", 0.42, 0.15, 2.0, 12, 1.0, 0.8),
    ("nonshift_setup_area", 0.42, 0.20, None, 12, 1.0, 0.8),
    ("nonshift_setup_area", 0.42, 0.20, 2.0, 12, 1.0, 0.8),
    ("shift1_setup_area", 0.42, 0.20, 2.0, 12, 1.0, 0.8),
    ("shift1_setup_area", 0.42, 0.15, 2.0, 12, 1.0, 0.8),
    ("shift1_setup_area", 0.35, 0.20, 2.0, 12, 1.0, 0.8),
    ("shift1_setup_area", 0.50, 0.15, 2.0, 12, 1.0, 0.8),
    ("shift1_setup_area", 0.42, 0.20, None, 12, 1.0, 0.8),
    ("shift1_setup_area", 0.42, 0.20, 2.0, 16, 1.5, 1.0),
]


def _run_name(
    model_name: str,
    long_thr: float,
    short_thr: float,
    lag: float | None,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
) -> str:
    lag_tag = "nolag" if lag is None else f"lag{int(lag)}"
    return f"{model_name}_l{long_thr:.2f}_s{short_thr:.2f}_{lag_tag}_h{horizon_bars}_tp{tp_atr:.1f}_sl{sl_atr:.1f}"


def _command(run: tuple[str, float, float, float | None, int, float, float]) -> tuple[list[str], Path]:
    model_name, long_thr, short_thr, lag, horizon_bars, tp_atr, sl_atr = run
    model = MODELS[model_name]
    out_dir = OUT_ROOT / _run_name(model_name, long_thr, short_thr, lag, horizon_bars, tp_atr, sl_atr)
    cmd = [
        sys.executable,
        "Models/ga_xgboost/analyze_phase4_triggers.py",
        "--ticker",
        "SPY",
        "--dataset-name",
        model["dataset"],
        "--x-filename",
        model["x_filename"],
        "--single-label-dir",
        model["single_label_dir"],
        "--long-setup-threshold",
        str(long_thr),
        "--short-setup-threshold",
        str(short_thr),
        "--splits",
        "test",
        "--skip-default-variants",
        "--include-asym-post-setup",
        "--asym-long-policy-filter",
        "break_prev_stop_1m_body_and_close",
        "--asym-short-policy-filter",
        "break_prev_stop_1m_body_and_close",
        "--post-setup-max-bars",
        "2,4,6",
        "--cooldown-bars",
        "12",
        "--horizon-bars",
        str(horizon_bars),
        "--tp-atr",
        str(tp_atr),
        "--sl-atr",
        str(sl_atr),
        "--use-1m-execution",
        "--execution-1m-path",
        EXECUTION_1M,
        "--setup-side-filter",
        "none",
        "--plot-top-n",
        "0",
        "--tail",
        "400",
        "--output-dir",
        str(out_dir),
    ]
    if lag is not None:
        cmd.extend(["--max-entry-lag-minutes", str(lag)])
    return cmd, out_dir


def _collect() -> pd.DataFrame:
    rows = []
    for scoreboard in OUT_ROOT.glob("*/best_phase4_trigger_scoreboard.csv"):
        df = pd.read_csv(scoreboard)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["run_name"] = scoreboard.parent.name
        row["run_dir"] = str(scoreboard.parent)
        row["model_name"] = scoreboard.parent.name.split("_l", 1)[0]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["total_ev_atr", "total_trades"], ascending=[False, False])


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "manifest.json").write_text(json.dumps({"models": MODELS, "runs": RUNS}, indent=2))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp/cynolycus_matplotlib").resolve()))
    failures = []
    for i, run in enumerate(RUNS, start=1):
        cmd, out_dir = _command(run)
        if (out_dir / "best_phase4_trigger_scoreboard.csv").exists():
            print(f"[focused] {i}/{len(RUNS)} skip {out_dir.name}", flush=True)
            continue
        print(f"[focused] {i}/{len(RUNS)} {out_dir.name}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with (out_dir / "run.log").open("w") as log:
            try:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=TIMEOUT_SECONDS, env=env)
                code = int(proc.returncode)
            except subprocess.TimeoutExpired:
                log.write(f"\n[focused] timed out after {TIMEOUT_SECONDS}s\n")
                code = 124
        elapsed = time.monotonic() - started
        print(f"[focused] done {out_dir.name} code={code} elapsed={elapsed:.1f}s", flush=True)
        if code != 0:
            failures.append({"run": out_dir.name, "code": code, "elapsed_seconds": elapsed})
        _collect().to_csv(OUT_ROOT / "focused_competition_best_scoreboards.csv", index=False)

    summary = _collect()
    summary.to_csv(OUT_ROOT / "focused_competition_best_scoreboards.csv", index=False)
    if failures:
        (OUT_ROOT / "failures.json").write_text(json.dumps(failures, indent=2))
    cols = [
        "run_name",
        "variant",
        "mode",
        "total_ev_atr",
        "total_trades",
        "long_ev_atr",
        "short_ev_atr",
        "long_win_rate",
        "short_win_rate",
        "post_setup_max_bars",
        "max_entry_lag_minutes",
    ]
    print(summary.head(20)[[c for c in cols if c in summary.columns]].to_string(index=False) if not summary.empty else "no rows")
    print(f"[focused] wrote {OUT_ROOT / 'focused_competition_best_scoreboards.csv'}")


if __name__ == "__main__":
    main()
