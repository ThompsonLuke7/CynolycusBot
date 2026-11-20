# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # CynolycusBot Stage 3 - final booster for the momentum-slot pick (Colab)
#
# **What this does.** Trains the pick from the Stage 3 validation (`L2_20__pit`:
# the 20-session vol-neutral label, 82 point-in-time features) ONE more time, on
# the last 24 months before the sealed test window, with exactly the walk-forward
# recipe (same rows, same inner early-stopping split, same parameters, seeds 42
# and 43). The boosters score the live decision-state archive forward in time
# (shadow only: nothing trades on them). Training stops 35 days before
# 2026-05-15, so both the sealed test and everything after it stay
# out-of-sample for these boosters.
#
# **How to run** (same folder as the Stage 3 run):
# 1. `MyDrive/cynolycus_stage3/` must still hold `stage3_matrix.parquet`,
#    `stage3_manifest.json` and `stage3_train_colab.py`; add this file.
# 2. Runtime: GPU (T4 is enough). Cell 1 mounts Drive; cell 2 runs
#    `%run /content/drive/MyDrive/cynolycus_stage3/stage3_final_colab.py`.
# 3. Each finished seed is saved to `MyDrive/cynolycus_stage3/final/` at once;
#    rerunning skips it. About 10 minutes, mostly copying the matrix.
# 4. When it prints `FINAL DONE`, download
#    `MyDrive/cynolycus_stage3/final/stage3_final.tar` and hand it back.

# %%
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ARM = "L2_20__pit"
TEST_START = pd.Timestamp("2026-05-15", tz="UTC")   # sealed test; training ends embargo days before it
SEEDS = (42, 43)
DRIVE_DIR = Path("/content/drive/MyDrive/cynolycus_stage3")


def _trainer(folder: Path):
    """Import the Stage 3 trainer sitting next to this file (its loader + split are reused verbatim)."""
    here = Path(__file__).resolve().parent if "__file__" in globals() else folder
    for base in (here, folder, Path("/content"), Path(".")):
        path = base / "stage3_train_colab.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("stage3_train_colab", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["stage3_train_colab"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("stage3_train_colab.py not found next to this script or in MyDrive/cynolycus_stage3/")


def main() -> None:
    T = _trainer(DRIVE_DIR)
    assert T.STAGE == "stage3", f"expected the Stage 3 trainer, got STAGE={T.STAGE!r}"
    work, _ = T.locate_bundle()
    manifest = json.loads((work / T.MANIFEST_NAME).read_text())
    arm = next(a for a in manifest["arms"] if a["name"] == ARM)
    feats = manifest["feature_sets"][arm["features"]]
    out = (DRIVE_DIR if DRIVE_DIR.parent.exists() else work) / "final"
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if shutil.which("nvidia-smi") else "cpu"

    wf = manifest["walk_forward"]
    train_end = TEST_START - pd.Timedelta(days=arm["embargo_days"])
    fold = {"train_start": train_end - pd.DateOffset(months=wf["train_months"]), "train_end": train_end,
            "test_start": TEST_START, "test_end": TEST_START}
    print(f"{ARM}: target={arm['target']} features={len(feats)} train {fold['train_start'].date()}.."
          f"{train_end.date()} device={device}")

    data = T.load_matrix(work / manifest["matrix"], feats, [arm["target"]],
                         min_names_per_bar=int(manifest.get("min_names_per_bar", 1)))
    col_idx = [data["cols"][c] for c in feats]
    y = data["y"][arm["target"]]
    ok = T.complete_rows(data["X"], col_idx) & ~np.isnan(y) & data["full_bar"]
    tr, va, _ = T.fold_indices(data["ts"], ok, fold, embargo_days=arm["embargo_days"],
                               inner_val_frac=wf["inner_val_frac"])
    print(f"train rows {len(tr):,}  early-stopping rows {len(va):,}  positives {float(np.mean(y[tr])):.3f}")
    if len(tr) < 50_000 or len(va) < 5_000:
        raise RuntimeError("too few rows for the final fit; is this the Stage 3 matrix?")
    X_tr, X_va = data["X"][np.ix_(tr, col_idx)], data["X"][np.ix_(va, col_idx)]

    import xgboost as xgb

    seeds_meta = []
    for seed in SEEDS:
        path = out / f"stage3_final_{ARM}_s{seed}.json"
        meta_path = out / f"stage3_final_{ARM}_s{seed}.meta.json"
        if path.exists() and meta_path.exists():
            print(f"seed {seed}: already saved - skipping")
            seeds_meta.append(json.loads(meta_path.read_text()))
            continue
        t0 = time.time()
        p = dict(manifest["xgb_params"])
        early = int(p.pop("early_stopping_rounds", 60))
        model = xgb.XGBClassifier(**p, objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                  device=device, n_jobs=-1, random_state=seed, early_stopping_rounds=early,
                                  verbosity=0)
        model.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], verbose=False)
        best = int(getattr(model, "best_iteration", -1))
        booster = model.get_booster()
        booster.feature_names = list(feats)
        booster.set_attr(best_iteration=str(best))
        booster.save_model(str(path))
        va_pred = model.predict_proba(X_va)[:, 1]
        base = float(np.mean(y[tr]))
        ll = float(-np.mean(y[va] * np.log(np.clip(va_pred, 1e-7, 1)) + (1 - y[va]) * np.log(np.clip(1 - va_pred, 1e-7, 1))))
        ll0 = float(-np.mean(y[va] * np.log(base) + (1 - y[va]) * np.log(1 - base)))
        meta = {"seed": seed, "best_iteration": best, "val_logloss": ll, "val_logloss_base_rate": ll0,
                "seconds": round(time.time() - t0, 1)}
        meta_path.write_text(json.dumps(meta, indent=2))
        seeds_meta.append(meta)
        print(f"seed {seed}: best_iter={best} val logloss {ll:.5f} vs base rate {ll0:.5f} ({meta['seconds']}s)")

    info = {
        "arm": arm, "features": list(feats), "rule": manifest.get("rule"), "selection": "621d0acaa1cd",
        "train_start": str(fold["train_start"]), "train_end": str(train_end), "test_start": str(TEST_START),
        "train_rows": int(len(tr)), "val_rows": int(len(va)), "xgb_params": manifest["xgb_params"],
        "seeds": seeds_meta, "matrix": manifest["matrix"], "matrix_rows": manifest.get("rows"),
        "scoring": "mean of the seeds' P(y=1), each booster up to its best_iteration attribute",
    }
    (out / "stage3_final_meta.json").write_text(json.dumps(info, indent=2, default=str))
    tar_path = out / "stage3_final.tar"
    with tarfile.open(tar_path, "w") as tar:
        for p in sorted(out.glob("stage3_final_*.json")):
            tar.add(p, arcname=p.name)
    print("FINAL DONE ->", tar_path)


# %%
if __name__ == "__main__":
    main()
