# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # CynolycusBot Stage 2 - label arms for the 4H models (Colab)
#
# **What this does.** Trains one XGBoost classifier per *arm* with the deployed
# walk-forward recipe (24-month train window, 6-month test folds) and saves each
# arm's out-of-fold scores. An arm = a target + a feature set. Half the arms are
# controls trained on the same labels shuffled within each bar; a real arm only
# counts if it beats its control.
#
# **How to run.**
# 1. Runtime -> Change runtime type -> GPU (T4 is enough; High-RAM if offered).
# 2. Put these three files in your Drive at `MyDrive/cynolycus_stage2/`:
#    `stage2_matrix.parquet` (2.7 GB), `stage2_manifest.json`, and this script.
#    (Uploading them to the Colab session folder also works, but Drive survives
#    a disconnect.)
# 3. Run all cells. The first pass is a ~1 minute smoke test; if it prints
#    `SMOKE OK` the full run starts by itself.
# 4. Every finished fold is saved to `MyDrive/cynolycus_stage2/out/`. If Colab
#    disconnects, just Run all again: finished folds and arms are skipped.
# 5. When it prints `ALL ARMS DONE`, download
#    `MyDrive/cynolycus_stage2/out/stage2_results.tar` and hand it back.
#
# Nothing here is run locally (project rule: training happens on Colab).

# %%
# !pip install -q "xgboost>=2.0" pyarrow pandas

from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MANIFEST_NAME = "stage2_manifest.json"
BUNDLE_NAME = "stage2_colab_bundle.tar"          # optional: the same files in one tar
DRIVE_DIR = Path("/content/drive/MyDrive/cynolycus_stage2")


# %%
def date_folds(ts: pd.Series, *, train_months: int, embargo_days: int, test_months: int) -> list[dict]:
    """Walk-forward folds, newest first then reversed -- identical to
    ``colab_competition.date_folds`` (kept inline so this file runs on its own)."""
    ts = pd.to_datetime(ts, utc=True)
    date_min, date_max = ts.min(), ts.max()
    folds, test_end = [], date_max
    while True:
        test_start = test_end - pd.DateOffset(months=test_months)
        train_end = test_start - pd.Timedelta(days=embargo_days)
        train_start = train_end - pd.DateOffset(months=train_months)
        if train_start < date_min:
            break
        folds.append(dict(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end))
        test_end = test_start
    return list(reversed(folds))


def full_bar_rows(ts: pd.Series, min_names: int) -> np.ndarray:
    """True for rows on bars that hold a real cross-section.

    ~1% of rows sit on stray timestamps (illiquid names whose first print of the
    day came late), a handful of tickers each. The live models rank a full
    cross-section, so those rows are excluded from every arm alike.
    """
    stamps = ts.dt.tz_convert(None).to_numpy() if ts.dt.tz is not None else ts.to_numpy()
    codes, counts = np.unique(stamps, return_inverse=True, return_counts=True)[1:]
    return counts[codes] >= int(min_names)


def load_matrix(path: Path, feature_cols: list[str], target_cols: list[str], *, ticker_frac: float = 1.0,
                min_names_per_bar: int = 1) -> dict:
    """Read the matrix column by column into ONE float32 block (no pandas copy).

    Peak memory ~= rows x features x 4 bytes. ``ticker_frac`` < 1 keeps a
    deterministic subset of tickers (smoke test). ``full_bar`` is judged on the
    WHOLE matrix before any subsetting, so a ticker subset keeps the same bars.
    """
    pf = pq.ParquetFile(path)
    ids = pf.read(columns=["timestamp", "ticker"]).to_pandas()
    ts = pd.to_datetime(ids["timestamp"], utc=True)
    ticker = ids["ticker"].astype(str).to_numpy()
    full_bar = full_bar_rows(ts, min_names_per_bar)
    keep = np.ones(len(ids), dtype=bool)
    if ticker_frac < 1.0:
        names = np.unique(ticker)
        keep = np.isin(ticker, names[:: max(1, int(round(1 / ticker_frac)))])
    n = int(keep.sum())
    X = np.empty((n, len(feature_cols)), dtype=np.float32)
    for j, col in enumerate(feature_cols):
        v = pf.read(columns=[col]).column(0).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        X[:, j] = v[keep]
    X[~np.isfinite(X)] = np.nan
    y = {c: pf.read(columns=[c]).column(0).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)[keep]
         for c in target_cols}
    return {"X": X, "cols": {c: j for j, c in enumerate(feature_cols)}, "ts": ts[keep].reset_index(drop=True),
            "ticker": ticker[keep], "y": y, "full_bar": full_bar[keep]}


def complete_rows(X: np.ndarray, col_idx: list[int], chunk: int = 500_000) -> np.ndarray:
    """True where every listed feature is present (deployed training drops the rest)."""
    ok = np.empty(X.shape[0], dtype=bool)
    for a in range(0, X.shape[0], chunk):
        ok[a:a + chunk] = ~np.isnan(X[a:a + chunk][:, col_idx]).any(axis=1)
    return ok


def fold_indices(ts: pd.Series, ok: np.ndarray, fold: dict, *, embargo_days: int, inner_val_frac: float):
    """(train, val, test) row indices for one fold.

    The last ``inner_val_frac`` of the train window's days are held out for early
    stopping, with the same embargo between inner-train and validation as
    between train and test -- forward-looking labels must not straddle a cut.
    """
    in_train = ok & (ts >= fold["train_start"]).to_numpy() & (ts <= fold["train_end"]).to_numpy()
    test = np.flatnonzero(ok & (ts >= fold["test_start"]).to_numpy() & (ts < fold["test_end"]).to_numpy())
    days = pd.DatetimeIndex(ts[in_train].dt.normalize().unique()).sort_values()
    if len(days) < 30:
        return np.array([], int), np.array([], int), test
    cut = days[min(max(1, int(len(days) * (1.0 - inner_val_frac))), len(days) - 1)]
    val = np.flatnonzero(in_train & (ts >= cut).to_numpy())
    train = np.flatnonzero(in_train & (ts < cut - pd.Timedelta(days=embargo_days)).to_numpy())
    return train, val, test


def xgb_fit_predict(X_tr, y_tr, X_va, y_va, X_te, *, seed: int, params: dict, device: str):
    """Fit one classifier, return (test scores, best iteration, gain importance)."""
    import xgboost as xgb

    p = dict(params)
    early = int(p.pop("early_stopping_rounds", 60))
    model = xgb.XGBClassifier(
        **p, objective="binary:logistic", eval_metric="logloss", tree_method="hist", device=device,
        n_jobs=-1, random_state=seed, early_stopping_rounds=early, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    scores = model.predict_proba(X_te)[:, 1].astype(np.float32)
    return scores, int(getattr(model, "best_iteration", -1)), model.feature_importances_.astype(np.float32)


def run_arm(arm: dict, data: dict, manifest: dict, out_dir: Path, *, seeds: list[int], fit_predict=xgb_fit_predict,
            device: str = "cpu", max_folds: int | None = None, min_train: int = 50_000,
            min_val: int = 5_000, log=print) -> dict | None:
    """Walk-forward OOF for one arm. Resumable: finished folds/arms are skipped.

    Writes ``oof_<arm>.parquet`` (timestamp, ticker, score, fold),
    ``importance_<arm>.csv`` and ``summary_<arm>.json``.
    """
    name = arm["name"]
    final = out_dir / f"oof_{name}.parquet"
    if final.exists():
        log(f"[{name}] already done - skipping")
        return None
    fold_dir = out_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    feats = manifest["feature_sets"][arm["features"]]
    col_idx = [data["cols"][c] for c in feats]
    y = data["y"][arm["target"]]
    ok = complete_rows(data["X"], col_idx) & ~np.isnan(y)
    ok &= data["full_bar"]
    wf = manifest["walk_forward"]
    folds = date_folds(data["ts"], train_months=wf["train_months"], embargo_days=arm["embargo_days"],
                       test_months=wf["test_months"])
    if max_folds:
        folds = folds[-max_folds:]
    log(f"[{name}] target={arm['target']} features={arm['features']}({len(feats)}) usable rows={int(ok.sum()):,} "
        f"positives={float(np.nanmean(y[ok])):.3f} folds={len(folds)} embargo={arm['embargo_days']}d")
    summary, gains = [], []
    for k, fold in enumerate(folds):
        part = fold_dir / f"{name}__f{k:02d}.parquet"
        if part.exists():
            summary.append(json.loads((fold_dir / f"{name}__f{k:02d}.json").read_text()))
            continue
        tr, va, te = fold_indices(data["ts"], ok, fold, embargo_days=arm["embargo_days"],
                                  inner_val_frac=wf["inner_val_frac"])
        if len(tr) < min_train or len(va) < min_val or len(te) == 0:
            log(f"[{name}] fold {k}: skipped (train={len(tr)} val={len(va)} test={len(te)})")
            continue
        t0 = time.time()
        X_tr, X_va, X_te = (data["X"][np.ix_(i, col_idx)] for i in (tr, va, te))
        score = np.zeros(len(te), dtype=np.float32)
        best = []
        for seed in seeds:
            s, it, gain = fit_predict(X_tr, y[tr], X_va, y[va], X_te, seed=seed,
                                      params=manifest["xgb_params"], device=device)
            score += s / len(seeds)
            best.append(it)
            gains.append(gain)
        del X_tr, X_va, X_te
        pd.DataFrame({"timestamp": data["ts"].iloc[te].to_numpy(), "ticker": data["ticker"][te],
                      "score": score, "fold": np.int16(k)}).to_parquet(part, index=False)
        rec = {"fold": k, "test_start": str(fold["test_start"].date()), "test_end": str(fold["test_end"].date()),
               "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)), "best_iteration": best,
               "seconds": round(time.time() - t0, 1)}
        (fold_dir / f"{name}__f{k:02d}.json").write_text(json.dumps(rec))
        summary.append(rec)
        log(f"[{name}] fold {k} {rec['test_start']}..{rec['test_end']} train={len(tr):,} test={len(te):,} "
            f"best_iter={best} {rec['seconds']}s")
    parts = sorted(fold_dir.glob(f"{name}__f*.parquet"))
    if not parts:
        log(f"[{name}] no folds produced")
        return None
    pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(final, index=False)
    if gains:
        pd.DataFrame({"feature": feats, "gain": np.mean(gains, axis=0)}).sort_values(
            "gain", ascending=False).to_csv(out_dir / f"importance_{name}.csv", index=False)
    out = {"arm": arm, "folds": summary, "rows": int(sum(r["test"] for r in summary))}
    (out_dir / f"summary_{name}.json").write_text(json.dumps(out, indent=2))
    return out


def pack_results(out_dir: Path) -> Path:
    tar_path = out_dir / "stage2_results.tar"
    with tarfile.open(tar_path, "w") as tar:
        for p in sorted(out_dir.glob("oof_*.parquet")) + sorted(out_dir.glob("summary_*.json")) + \
                sorted(out_dir.glob("importance_*.csv")):
            tar.add(p, arcname=p.name)
    return tar_path


def locate_bundle() -> tuple[Path, Path]:
    """(folder holding the manifest + matrix, output folder). Uses Drive when mounted.

    Looks for ``stage2_manifest.json`` next to ``stage2_matrix.parquet`` in the
    Drive folder, the Colab session folder, or the current folder; failing that,
    for ``stage2_colab_bundle.tar`` in the same places and extracts it.
    """
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
    except Exception as exc:  # not on Colab, or Drive declined
        print("Drive not mounted:", exc)
    places = [DRIVE_DIR, Path("/content"), Path(".")]
    work = next((p for p in places if (p / MANIFEST_NAME).exists()), None)
    if work is None:
        bundle = next((p / BUNDLE_NAME for p in places if (p / BUNDLE_NAME).exists()), None)
        if bundle is None:
            raise FileNotFoundError(
                f"Put {MANIFEST_NAME} and stage2_matrix.parquet (or {BUNDLE_NAME}) in one of: "
                f"{[str(p) for p in places]}"
            )
        work = Path("/content/stage2_work") if Path("/content").exists() else Path("stage2_work")
        work.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle) as tar:
            tar.extractall(work)
    out = (DRIVE_DIR if DRIVE_DIR.parent.exists() else work) / "out"
    out.mkdir(parents=True, exist_ok=True)
    return _local_copy(work), out


def _local_copy(work: Path) -> Path:
    """On Colab, read the matrix from local disk, not through the Drive mount
    (118 column reads of a 2.7 GB file over Drive are slow and can stall)."""
    local = Path("/content/stage2_work")
    if not Path("/content").exists() or work.resolve() == local.resolve():
        return work
    local.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((work / MANIFEST_NAME).read_text())
    for name in (MANIFEST_NAME, manifest["matrix"]):
        src, dst = work / name, local / name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            print(f"copying {name} to local disk ({src.stat().st_size / 1e9:.2f} GB) ...")
            shutil.copy2(src, dst)
    return local


def main() -> None:
    work, out_dir = locate_bundle()
    manifest = json.loads((work / "stage2_manifest.json").read_text())
    device = "cuda" if shutil.which("nvidia-smi") else "cpu"
    arms = manifest["arms"]
    only = [a.strip() for a in os.environ.get("ARMS", "").split(",") if a.strip()]
    if only:
        arms = [a for a in arms if a["name"] in only]
    all_feats = list(dict.fromkeys(c for a in arms for c in manifest["feature_sets"][a["features"]]))
    targets = sorted({a["target"] for a in arms})
    seeds = [int(s) for s in os.environ.get("SEEDS", "42,43").split(",")]
    min_names = int(manifest.get("min_names_per_bar", 1))
    print(f"device={device} arms={[a['name'] for a in arms]} features={len(all_feats)} seeds={seeds} out={out_dir}")

    if os.environ.get("SKIP_SMOKE") != "1":
        smoke_dir = Path("/content/stage2_smoke") if Path("/content").exists() else Path("stage2_smoke")
        shutil.rmtree(smoke_dir, ignore_errors=True)
        smoke_dir.mkdir(parents=True)
        small = load_matrix(work / manifest["matrix"], all_feats, targets, ticker_frac=0.05,
                            min_names_per_bar=min_names)
        t0 = time.time()
        res = run_arm(arms[0], small, manifest, smoke_dir, seeds=seeds[:1], device=device, max_folds=1)
        assert res and res["rows"] > 0, "smoke test produced no out-of-fold rows"
        print(f"SMOKE OK ({time.time() - t0:.0f}s on 5% of tickers, one fold)")
        del small

    data = load_matrix(work / manifest["matrix"], all_feats, targets, min_names_per_bar=min_names)
    print(f"matrix loaded: {data['X'].shape[0]:,} rows x {data['X'].shape[1]} features "
          f"({data['X'].nbytes / 1e9:.1f} GB), {data['ts'].min().date()} .. {data['ts'].max().date()}")
    for arm in arms:
        run_arm(arm, data, manifest, out_dir, seeds=seeds[:1] if arm.get("control") else seeds, device=device)
        pack_results(out_dir)          # keep the tar current so a partial run is still usable
    print("ALL ARMS DONE ->", pack_results(out_dir))


# %%
if __name__ == "__main__":
    main()
