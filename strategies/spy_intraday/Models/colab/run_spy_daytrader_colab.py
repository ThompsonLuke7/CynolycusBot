# SPY day-trader competition — Colab driver
# Trains BOTH swing targets (long + short), each running all 4 families
# (xgb/lgbm × classifier/ranker) with the magnitude-aware settings baked into the
# patched trainer: classifiers get scale_pos_weight(auto) + MFE sample-weights,
# rankers get graded relevance. Per-target output is checkpointed to Drive and the
# loop resumes (skips) any target already finished. Paste cells into Colab top-to-bottom.

# %% [1] GPU + deps -----------------------------------------------------------
get_ipython().system("nvidia-smi -L || echo 'no GPU (CPU is fine, just slower)'")
get_ipython().system("pip -q install xgboost lightgbm")

# %% [2] mount Drive ----------------------------------------------------------
from google.colab import drive
from pathlib import Path
drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/cynolycus/spy_daytrader")   # <-- adjust if you like
DRIVE.mkdir(parents=True, exist_ok=True)
print("Drive output dir:", DRIVE)

# %% [3] get the training bundle (cache on Drive, upload once) -----------------
import shutil
BUNDLE = Path("/content/spy_daytrader_context_colab_bundle.tgz")
drive_bundle = DRIVE / BUNDLE.name
if drive_bundle.exists():
    shutil.copy(drive_bundle, BUNDLE)
    print("loaded bundle from Drive:", drive_bundle)
else:
    from google.colab import files
    print("Upload spy_daytrader_context_colab_bundle.tgz ...")
    up = files.upload()
    shutil.move(next(iter(up)), BUNDLE)
    shutil.copy(BUNDLE, drive_bundle)   # cache so future sessions skip the upload
    print("cached bundle to Drive:", drive_bundle)

# %% [4] extract bundle so the trainer + engine are importable ----------------
import tarfile, os
with tarfile.open(BUNDLE) as t:
    t.extractall("/content")
os.chdir("/content")
print("extracted:", sorted(p.name for p in Path("/content").glob("*.py")))

# %% [5] train both targets, checkpoint each to Drive, resume if rerun --------
import subprocess
TARGETS = ["long_swing_label", "short_swing_label"]
KNOBS = dict(
    SCALE_POS_WEIGHT="auto",   # neg/pos imbalance for the ~7% positive base rate
    MAGNITUDE_GAIN="1.0",      # strength of the big-move up-weighting (raise to sharpen)
    GRADED_RELEVANCE="1",      # ranker target graded by MFE quantile
    GRADED_BINS="4",
    N_SEEDS="5", TOP_K="5", RANK_GROUP="date",
    MODEL_FAMILIES="xgb_classifier,xgb_ranker,lgbm_classifier,lgbm_ranker",
)
for tgt in TARGETS:
    out = DRIVE / f"spy_daytrader_{tgt}_model_competition_bundle.tgz"
    if out.exists():
        print("skip (already done):", tgt)
        continue
    env = dict(os.environ, SPY_BUNDLE=str(BUNDLE), SPY_TARGET=tgt, **KNOBS)
    print(f"\n===== training {tgt} =====")
    subprocess.run(["python", "spy_daytrader_train_colab.py"], env=env, check=True)
    src = Path(f"spy_daytrader_{tgt}_model_competition_bundle.tgz")
    shutil.copy(src, out)                 # checkpoint immediately (don't lose it to a disconnect)
    print("saved ->", out)
print("\nAll targets complete. Result bundles are in", DRIVE)

# %% [6] (optional) peek at the winners ---------------------------------------
import json
for tgt in TARGETS:
    b = DRIVE / f"spy_daytrader_{tgt}_model_competition_bundle.tgz"
    if not b.exists():
        continue
    with tarfile.open(b) as t:
        meta = json.loads(t.extractfile("competition_meta.json").read())
    print(tgt, "-> best:", meta["best"]["family"], "seed", meta["best"]["seed"],
          "| magnitude:", meta.get("magnitude_column"), "graded:", meta.get("graded_relevance"),
          "scale_pos_weight:", meta.get("scale_pos_weight"))
