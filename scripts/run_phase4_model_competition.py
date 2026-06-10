from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


OUT_ROOT = Path("Data/models/ga_xgboost/model_competition_phase4")
EXECUTION_1M = "Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet"
COMMAND_TIMEOUT_SECONDS = 540

MODELS = [
    {
        "name": "nonshift_swing",
        "dataset": "10min",
        "x_filename": "X_10min_tree.parquet",
        "single_label_dir": "swing_single",
        "note": "clean OOS non-shift swing model",
    },
    {
        "name": "nonshift_setup_area",
        "dataset": "10min",
        "x_filename": "X_10min_tree.parquet",
        "single_label_dir": "swing_support_single",
        "note": "setup-area/support model; full-fit probabilities, optimistic",
    },
    {
        "name": "shift1_setup_area",
        "dataset": "10min_shift1",
        "x_filename": "X_10min_shift1_tree.parquet",
        "single_label_dir": "swing_support_single",
        "note": "shifted one bar setup-area/support model",
    },
]

THRESHOLDS = [(0.42, 0.15), (0.42, 0.20), (0.35, 0.20), (0.50, 0.15)]
LAG_SETTINGS = [None, 2.0]
HORIZON_SETTINGS = [(12, 1.0, 0.8), (16, 1.5, 1.0)]
ENTRY_POLICIES = ",".join(
    [
        "break_prev_stop_1m_confirm",
        "break_prev_stop_1m_momentum",
        "break_prev_stop_1m_body_and_close",
    ]
)


def _run(cmd: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp/cynolycus_matplotlib").resolve()))
    started = time.monotonic()
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
                env=env,
            )
            code = int(proc.returncode)
        except subprocess.TimeoutExpired:
            log.write(f"\n[competition] timed out after {COMMAND_TIMEOUT_SECONDS}s\n")
            code = 124
    return code, time.monotonic() - started


def _command(model: dict[str, str], threshold: tuple[float, float], lag: float | None, horizon: tuple[int, float, float]) -> tuple[list[str], Path]:
    long_thr, short_thr = threshold
    horizon_bars, tp_atr, sl_atr = horizon
    lag_tag = "nolag" if lag is None else f"lag{int(lag)}"
    run_name = f"{model['name']}_l{long_thr:.2f}_s{short_thr:.2f}_{lag_tag}_h{horizon_bars}_tp{tp_atr:.1f}_sl{sl_atr:.1f}"
    out_dir = OUT_ROOT / run_name
    cmd = [
        sys.executable,
        "strategies/spy_intraday/Models/ga_xgboost/analyze_phase4_triggers.py",
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
        ENTRY_POLICIES,
        "--asym-short-policy-filter",
        ENTRY_POLICIES,
        "--post-setup-max-bars",
        "2,4",
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
        run_dir = scoreboard.parent
        df = pd.read_csv(scoreboard)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["run_dir"] = str(run_dir)
        row["run_name"] = run_dir.name
        row["model_name"] = run_dir.name.split("_l", 1)[0]
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["total_ev_atr", "total_trades"], ascending=[False, False])


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"models": MODELS, "thresholds": THRESHOLDS, "lags": LAG_SETTINGS, "horizons": HORIZON_SETTINGS}
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    failures = []
    total = len(MODELS) * len(THRESHOLDS) * len(LAG_SETTINGS) * len(HORIZON_SETTINGS)
    i = 0
    for model in MODELS:
        for threshold in THRESHOLDS:
            for lag in LAG_SETTINGS:
                for horizon in HORIZON_SETTINGS:
                    i += 1
                    cmd, out_dir = _command(model, threshold, lag, horizon)
                    if (out_dir / "best_phase4_trigger_scoreboard.csv").exists():
                        print(f"[competition] {i}/{total} skip {out_dir.name}", flush=True)
                        continue
                    print(f"[competition] {i}/{total} {out_dir.name}", flush=True)
                    code, elapsed = _run(cmd, out_dir / "run.log")
                    print(f"[competition] done {out_dir.name} code={code} elapsed={elapsed:.1f}s", flush=True)
                    if code != 0:
                        failures.append({"run": out_dir.name, "code": code, "elapsed_seconds": elapsed})
                        print(f"[competition] failed {out_dir.name} code={code}", flush=True)
                    _collect().to_csv(OUT_ROOT / "competition_best_scoreboards.csv", index=False)
    summary = _collect()
    summary_path = OUT_ROOT / "competition_best_scoreboards.csv"
    summary.to_csv(summary_path, index=False)
    top_cols = [
        "run_name",
        "variant",
        "mode",
        "total_ev_atr",
        "total_trades",
        "long_ev_atr",
        "short_ev_atr",
        "long_win_rate",
        "short_win_rate",
        "long_setup_threshold",
        "short_setup_threshold",
        "post_setup_max_bars",
        "max_entry_lag_minutes",
        "horizon_bars",
    ]
    available_cols = [c for c in top_cols if c in summary.columns]
    print("\n[competition] top results")
    print(summary.head(25)[available_cols].to_string(index=False) if not summary.empty else "no successful runs")
    if failures:
        (OUT_ROOT / "failures.json").write_text(json.dumps(failures, indent=2))
        print(f"[competition] failures={len(failures)}")
    print(f"[competition] wrote {summary_path}")


if __name__ == "__main__":
    main()
