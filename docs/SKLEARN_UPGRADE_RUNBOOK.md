# scikit-learn 1.5.2 -> 1.6.1 upgrade runbook

Prepared 2026-08-10 05:45 ET. **Run after a market close, never inside the live
window.** Every step is reversible; the rollback is a single `pip install`.

## Why

`requirements.txt` pinned `scikit-learn>=1.5,<1.6`. That pin is inverted
against reality:

* It caused a production outage. umap-learn 0.5.12 and hdbscan 0.8.44 both call
  `check_array(ensure_all_finite=...)`, the name scikit-learn adopted in 1.6.
  Under 1.5.2 both raise `TypeError`, which killed stage 4 of the 2026-08-03
  weekly refresh and left the Meta Ranker on theme features stamped 2026-07-20
  for the whole 08-03..08-07 trading week.
* It protects nothing. All 30 estimator artifacts loaded live were pickled by a
  *newer* scikit-learn than the pin allows — 28 on 1.6.1, 2 on 1.9.0 (the
  `oof_ranker_20260618` isotonic calibrators). Loading them already crosses a
  version boundary; the pin only controls which way.
* `umap-learn` and `hdbscan` were never listed in `requirements.txt` at all,
  which is how they silently drifted onto a release that needs 1.6.

1.6.1 is chosen over 1.9.0 because it matches 28 of the 30 artifacts and is the
smallest step that satisfies umap/hdbscan. The two 1.9.0 calibrators will still
emit `InconsistentVersionWarning`; they were checked on 2026-08-10 under 1.5.2
and predict monotonically within [0, 1]. Parity is what gates the change.

## Pre-flight

```bash
cd /home/luket/repos/CynolycusBot
git rev-parse --abbrev-ref HEAD                  # must be main
ps -ef | rg 'combined_server|nightly_|weekly_refresh' | rg -v rg   # must be empty
cat Data/runtime/live_data_jobs.lock             # must be empty (lock free)
```

Do not proceed while the server or any heavy data job is running — the runners
are fresh subprocesses and would pick up a half-swapped interpreter.

## Step 1 — baseline (already captured)

`Data/runtime/sklearn_parity_before_1.5.2.json` holds a fingerprint of all 157
pickled artifacts scored on fixed seeded input under 1.5.2 / numpy 2.2.6.

**The baseline can only be captured on 1.5.2, and it is not in git** (`Data/**`
is gitignored). Once step 2 runs, re-capturing means downgrading first. If the
file is missing or models changed since 2026-08-10, re-capture *before*
upgrading:

```bash
PYTHONPATH=. .venv/bin/python scripts/check_sklearn_upgrade_parity.py \
  --save Data/runtime/sklearn_parity_before_1.5.2.json
```

## Step 2 — upgrade

Verified by `pip install --dry-run` on 2026-08-10: this touches exactly one
package. numpy, scipy, joblib and threadpoolctl are all already satisfied.

```bash
.venv/bin/pip install 'scikit-learn==1.6.1'
```

## Step 3 — parity gate

```bash
PYTHONPATH=. .venv/bin/python scripts/check_sklearn_upgrade_parity.py \
  --compare Data/runtime/sklearn_parity_before_1.5.2.json
```

* exit 0 — every artifact byte-identical. Continue.
* exit 1 — a model moved. **Roll back** (below) and investigate the named files.
* exit 2 — the artifact set changed since the baseline. Re-capture on 1.5.2
  first, or the comparison is meaningless.

The checker is self-tested: it was confirmed to return 1 on an injected 1e-6
drift and 2 on a set mismatch, so exit 0 is a real signal rather than silence.

## Step 4 — the checks the parity harness cannot cover

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q          # expect 933 passed, 0 failed
PYTHONPATH=. .venv/bin/python -m pytest themes/dynamic_theme/tests/ -q
```

The theme tests matter most here: they exercise `load_umap`/`load_hdbscan`, and
under 1.6.1 the shim must go inert (`_needs_shim()` returns False) while UMAP
and HDBSCAN still run. sentence-transformers also depends on scikit-learn but
computes BGE embeddings in torch, so it is not expected to move; the theme
tests cover that path end to end.

## Step 5 — write the pins down

Only after steps 3 and 4 pass:

```diff
-scikit-learn>=1.5,<1.6
+scikit-learn>=1.6,<1.7
+umap-learn==0.5.12
+hdbscan==0.8.44
```

Pinning umap-learn and hdbscan explicitly is the part that stops this
recurring — the outage happened because they were unpinned and free to move.

## Step 6 — optional cleanup

`themes/dynamic_theme/sklearn_compat.py` self-disables above scikit-learn 1.6,
so after this upgrade it is inert, not harmful. Removing it is tidiness, not a
requirement, and it is the safety net if the pin is ever moved back. If you do
remove it, revert the two call sites to plain `import umap` / `import hdbscan`:

* `themes/dynamic_theme/stages/step03_cluster.py`
* `themes/dynamic_theme/emerging.py`

and delete `themes/dynamic_theme/tests/test_sklearn_compat.py`.

## Rollback

```bash
.venv/bin/pip install 'scikit-learn==1.5.2'
```

Then revert `requirements.txt` if step 5 was already applied. The shim keeps the
theme pipeline working on 1.5.2, so a rollback is safe to leave in place
indefinitely.
