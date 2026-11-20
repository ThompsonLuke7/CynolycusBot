"""Follow-up 3: robust seed stability at the chosen co-movement weight (w=0.2).

Two-seed estimates proved noisy (identical inputs gave 0.86 and 0.58), so this uses
6 UMAP seeds and all 15 pairs, for HDBSCAN eom/5 vs leaf/5. Co-movement as of
2026-10-05, as in comovement_sweep.py. Writes seed_stability.csv.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

import comovement_sweep as cs  # noqa: E402
from themes.dynamic_theme import config as C  # noqa: E402
from themes.dynamic_theme.sklearn_compat import load_hdbscan, load_umap  # noqa: E402
from themes.dynamic_theme.stages.step02_embed import _build_comovement_vectors  # noqa: E402

SEEDS = [42, 7, 123, 2024, 99, 314]


def main() -> None:
    tickers = pd.read_csv(HERE / "b2_tickers.csv").iloc[:, 0].astype(str).tolist()
    mat = cs.blend(np.load(HERE / "b2_embeddings.npy"), _build_comovement_vectors(tickers, as_of=cs.AS_OF), tickers, 0.2)
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
    layouts = {s: load_umap().UMAP(n_components=C.UMAP_N_COMPONENTS, n_neighbors=C.UMAP_N_NEIGHBORS,
                                   min_dist=C.UMAP_MIN_DIST, metric=C.UMAP_METRIC, random_state=s,
                                   low_memory=False).fit_transform(mat) for s in SEEDS}
    rows = []
    for method in ("eom", "leaf"):
        prim, noise, ngroups = {}, [], []
        for s, red in layouts.items():
            labels = load_hdbscan().HDBSCAN(min_cluster_size=5, min_samples=C.HDBSCAN_MIN_SAMPLES,
                                            metric=C.HDBSCAN_METRIC, cluster_selection_method=method).fit(red).labels_
            ids = sorted(set(labels) - {-1})
            cents = np.stack([norm[labels == i].mean(axis=0) for i in ids])
            cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-10
            prim[s] = np.array(ids)[np.argmax(norm @ cents.T, axis=1)]
            noise.append(float((labels == -1).mean()))
            ngroups.append(len(ids))
        aris = [adjusted_rand_score(prim[a], prim[b]) for a, b in combinations(SEEDS, 2)]
        rows.append({"method": method, "pairs": len(aris), "ari_mean": round(float(np.mean(aris)), 3),
                     "ari_min": round(float(np.min(aris)), 3), "ari_max": round(float(np.max(aris)), 3),
                     "groups_mean": round(float(np.mean(ngroups)), 1), "noise_mean": round(float(np.mean(noise)), 3)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / "seed_stability.csv", index=False)


if __name__ == "__main__":
    main()
